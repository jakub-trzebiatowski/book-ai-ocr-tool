"""Synthesize chapter audio using Gemini TTS."""

from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from google.genai import Client
from google.genai.types import Part, Content, SpeechConfig, VoiceConfig, PrebuiltVoiceConfig, \
    GenerateContentConfig, GenerateContentResponse, JobState

from book_ai_ocr_tool.models import ChapterContent

PROJECT_ID = "book-digitizer-482317"
LOCATION_ID = "us-central1"

AUDIO_MIME = "audio/L16;codec=pcm;rate=24000"

FINAL_JOB_STATES = (
    JobState.JOB_STATE_SUCCEEDED,
    JobState.JOB_STATE_FAILED,
    JobState.JOB_STATE_CANCELLED,
    JobState.JOB_STATE_EXPIRED,
)

# Defaults tuned for quick previews; allow overrides via CLI.
DEFAULT_TTS_MODEL = "gemini-2.5-flash-preview-tts"

NARRATOR_TAG = "narrator"

@dataclass
class TTSConfig:
    chapters_dir: Path
    chapter_id: str
    output_dir: Path
    voice_map_path: Path
    model: str = DEFAULT_TTS_MODEL


def parse_args(argv: Sequence[str]) -> TTSConfig:
    parser = argparse.ArgumentParser(description="Synthesize audiobook audio from a chapter JSON using Gemini TTS")
    parser.add_argument("--chapters-dir", required=True, type=Path, help="Directory containing chapter_XXX.json files")
    parser.add_argument("--chapter-id", required=True, type=str,
                        help="ID of the chapter to process (e.g., 'chapter_001')")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory where wave files will be written")
    parser.add_argument("--voice-map", required=True, dest="voice_map_path", type=Path,
                        help="Path to JSON file mapping tags to voices")
    parser.add_argument("--model", default=DEFAULT_TTS_MODEL,
                        help=f"Gemini TTS model to use (default: {DEFAULT_TTS_MODEL})")

    args = parser.parse_args(argv)

    return TTSConfig(
        chapters_dir=args.chapters_dir,
        output_dir=args.output_dir,
        voice_map_path=args.voice_map_path,
        chapter_id=args.chapter_id,
        model=args.model,
    )


def load_api_key(env_var: str) -> str:
    key = os.getenv(env_var)
    if not key:
        raise RuntimeError(f"Environment variable {env_var} is not set")
    return key


def load_voice_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_chapter(path: Path) -> ChapterContent:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return ChapterContent(**data)


def slugify_tag(tag: str | None) -> str:
    if not tag:
        return "untagged"
    slug_chars: list[str] = []
    for char in tag.lower():
        if char.isalnum():
            slug_chars.append(char)
        elif slug_chars and slug_chars[-1] != "-":
            slug_chars.append("-")
    slug = "".join(slug_chars).strip("-")
    return slug or "tagged"


def generate_content_tts(
        client: Client,
        text: str,
        voice: str,
) -> bytes:
    if not text.strip():
        raise ValueError("Cannot synthesize empty text")

    response = client.models.generate_content(
        model=DEFAULT_TTS_MODEL,
        contents=Content(
            role="user",
            parts=[
                Part.from_text(text=text),
            ],
        ),
        config=GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(
                        voice_name=voice,
                    ),
                ),
            ),
        ),
    )

    audio_data = _extract_inline_data_bytes(
        response=response,
        expected_mime_type=AUDIO_MIME,
    )

    return audio_data


def write_wav_file(path: Path, audio_bytes: bytes) -> None:
    ensure_dir(path.parent)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(24000)  # 24kHz sample rate
        wf.writeframes(audio_bytes)


def main(argv: Iterable[str] | None = None) -> int:
    config = parse_args(list(argv) if argv is not None else sys.argv[1:])

    chapter_json_path = config.chapters_dir / f"{config.chapter_id}.json"
    if not chapter_json_path.exists():
        print(f"Chapter JSON file not found: {chapter_json_path}", file=sys.stderr)
        return 2

    try:
        voice_map = load_voice_map(config.voice_map_path)
    except FileNotFoundError:
        print(f"Voice map file not found: {config.voice_map_path}", file=sys.stderr)
        return 3
    except json.JSONDecodeError:
        print(f"Failed to parse voice map JSON: {config.voice_map_path}", file=sys.stderr)
        return 3

    narrator_voice_name = voice_map.get(NARRATOR_TAG)

    if narrator_voice_name is None:
        raise RuntimeError(f"No voice mapping found for narrator tag '{NARRATOR_TAG}'")

    try:
        chapter = load_chapter(chapter_json_path)
    except Exception as exc:  # pragma: no cover - file dependent
        print(f"Failed to load chapter {chapter_json_path.name}: {exc}", file=sys.stderr)
        return 5

    chapter_output_dir = config.output_dir / config.chapter_id
    ensure_dir(chapter_output_dir)

    def build_paragraph_audio_file_path(index: int) -> Path:
        filename = f"{index:03d}.wav"
        return chapter_output_dir / filename

    print(f"Synthesizing chapter '{config.chapter_id}': {chapter.title}")

    client = Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION_ID,
    )

    title_audio_bytes = generate_content_tts(
        client=client,
        text=chapter.title,
        voice=narrator_voice_name,
    )

    title_wav_path = chapter_output_dir / "000_title.wav"

    write_wav_file(
        path=title_wav_path,
        audio_bytes=title_audio_bytes,
    )

    for paragraph_idx, paragraph in enumerate(chapter.paragraphs, start=1):
        paragraph_wav_path = build_paragraph_audio_file_path(
            index=paragraph_idx,
        )

        if paragraph_wav_path.exists():
            print(f"  skipping existing file: {paragraph_wav_path.relative_to(config.output_dir)}")
            continue

        voice = voice_map.get(paragraph.tag)

        if voice is None:
            raise RuntimeError(f"No voice mapping found for tag '{paragraph.tag}'")

        paragraph_audio_bytes = generate_content_tts(
            client=client,
            text=paragraph.text,
            voice=voice,
        )

        write_wav_file(
            path=paragraph_wav_path,
            audio_bytes=paragraph_audio_bytes,
        )

    return 0


def _extract_inline_data_bytes(
        response: GenerateContentResponse,
        expected_mime_type: str,
) -> bytes:
    candidates = response.candidates

    if candidates is None or len(candidates) == 0:
        raise RuntimeError("No candidate for inline response")

    content = candidates[0].content

    if content is None:
        raise RuntimeError("No content for inline response")

    parts = content.parts

    if parts is None or len(parts) == 0:
        raise RuntimeError("No part for inline response")

    part = parts[0]

    inline_data = part.inline_data

    if inline_data is None:
        raise RuntimeError("No inline data for inline response")

    if inline_data.mime_type != expected_mime_type:
        raise RuntimeError(f"Inline data does not match expected mime type: {expected_mime_type}")

    data = inline_data.data

    if data is None:
        raise RuntimeError("No data for inline response")

    return data


if __name__ == "__main__":
    raise SystemExit(main())
