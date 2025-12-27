import argparse
import json
import os
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Sequence

from google import genai
from google.genai import types
from PIL import Image

from book_ai_ocr_tool.models import ImageOCRResult

PROJECT_ID = "book-digitizer-482317"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_API_KEY_ENV = "GOOGLE_API_KEY"
SYSTEM_INSTRUCTIONS = (
    "You are an OCR assistant. Process book page images, return structured JSON matching the ImageOCRResult schema. "
    "Use proper Unicode typography, never treat page numbers as body paragraphs, ignore footnotes unless explicitly requested."
)


@dataclass
class OCRConfig:
    input_dir: Path
    output_dir: Path
    prompt_file: Path
    model: str = DEFAULT_MODEL
    api_key_env: str = DEFAULT_API_KEY_ENV
    limit: int | None = None


@dataclass
class ImageHandle:
    input_image_path: Path
    output_dir: Path

    @property
    def output_json_path(self) -> Path:
        """Compute output JSON path from input image stem."""
        return self.output_dir / f"{self.input_image_path.stem}.json"

    def output_exists(self) -> bool:
        """Check if the output JSON file already exists."""
        return self.output_json_path.exists()

    def to_png_bytes(self) -> bytes:
        """Convert the input image to PNG bytes."""
        return image_to_png_bytes(self.input_image_path)

    def write_result(self, result: ImageOCRResult):
        """Write OCR result to output JSON file."""

        with self.output_json_path.open("w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)

        return


def parse_args(argv: Sequence[str]) -> OCRConfig:
    parser = argparse.ArgumentParser(description="OCR book pages using Gemini with structured output")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing page images")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write OCR results")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model to use (default: gemini-2.0-flash)")
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help="Environment variable containing the Gemini API key (default: GOOGLE_API_KEY)",
    )
    parser.add_argument(
        "--prompt-file",
        required=True,
        type=Path,
        help="Path to a text file containing the user prompt for the OCR task",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of images to process (useful for testing)",
    )
    args = parser.parse_args(argv)
    return OCRConfig(input_dir=args.input_dir, output_dir=args.output_dir, model=args.model,
                     api_key_env=args.api_key_env, prompt_file=args.prompt_file, limit=args.limit)


def find_images(directory: Path) -> List[Path]:
    files: List[Path] = []
    for entry in directory.iterdir():
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(entry)
    return sorted(files)


def image_to_png_bytes(path: Path) -> bytes:
    with Image.open(path) as img:
        img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_prompt(prompt_file: Path) -> str:
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    with prompt_file.open("r", encoding="utf-8") as f:
        return f.read().strip()


def call_gemini(
        client: genai.Client,
        model: str,
        image_png: bytes,
        user_prompt_text: str,
) -> ImageOCRResult:
    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=ImageOCRResult.model_json_schema(),
        ),
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=SYSTEM_INSTRUCTIONS),
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=user_prompt_text),
                    types.Part.from_bytes(data=image_png, mime_type="image/png"),
                ],
            ),
        ],
    )

    if not response.text:
        raise RuntimeError("Gemini response payload was empty")

    return ImageOCRResult.model_validate_json(response.text)


def load_api_key(env_var: str) -> str:
    key = os.getenv(env_var)
    if not key:
        raise RuntimeError(f"Environment variable {env_var} is not set")
    return key


def ocr_directory(config: OCRConfig) -> int:
    if not config.input_dir.exists() or not config.input_dir.is_dir():
        print(f"Input directory not found: {config.input_dir}", file=sys.stderr)
        return 2

    ensure_output_dir(config.output_dir)
    images = find_images(config.input_dir)
    if not images:
        print("No images found to process.", file=sys.stderr)
        return 3

    try:
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location="us-central1"
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"Failed to initialize Gemini client: {exc}", file=sys.stderr)
        return 4

    try:
        user_prompt_text = load_prompt(config.prompt_file)
    except Exception as exc:  # pragma: no cover - file dependent
        print(f"Failed to load prompt: {exc}", file=sys.stderr)
        return 6

    failures = 0
    processed = 0

    for idx, image_path in enumerate(images, start=1):
        if config.limit is not None and processed >= config.limit:
            print(f"Reached limit of {config.limit} processed images, stopping.")
            break

        handle = ImageHandle(input_image_path=image_path, output_dir=config.output_dir)

        # Skip processing if output already exists
        if handle.output_exists():
            print(f"Skipped page {idx}: {image_path.name} (output already exists)")
            continue

        try:
            image_png = handle.to_png_bytes()
            result = call_gemini(client, config.model, image_png, user_prompt_text)

            handle.write_result(result)

            processed += 1

            print(f"Processed image {idx}: {image_path.name}")
        except Exception as exc:  # pragma: no cover - network and file dependent
            failures += 1
            print(f"Failed page {idx} ({image_path.name}): {exc}", file=sys.stderr)

    if failures:
        print(f"Completed with {failures} failures", file=sys.stderr)
        return 5

    return 0


def main(argv: Iterable[str] | None = None) -> int:
    config = parse_args(list(argv) if argv is not None else sys.argv[1:])
    return ocr_directory(config)


if __name__ == "__main__":
    raise SystemExit(main())
