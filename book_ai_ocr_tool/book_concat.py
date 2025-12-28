import argparse
import sys
import wave
from pathlib import Path
from typing import Iterable, Tuple


AudioParams = Tuple[int, int, int, int, str, str]


def _read_wav(path: Path) -> Tuple[AudioParams, bytes]:
    with wave.open(str(path), "rb") as wav_file:
        params: AudioParams = wav_file.getparams()
        frames = wav_file.readframes(params[3])
    return params, frames


def _assert_compatible(reference: AudioParams, candidate: AudioParams, path: Path) -> None:
    ref_core = reference[:3] + reference[4:]
    cand_core = candidate[:3] + candidate[4:]
    if ref_core != cand_core:
        raise ValueError(
            f"Audio params mismatch for {path.name}: expected channels/sample_width/rate/compression {ref_core}, got {cand_core}"
        )


def _silence(duration_seconds: float, params: AudioParams) -> bytes:
    frame_count = int(round(duration_seconds * params[2]))
    if frame_count <= 0:
        return b""
    total_samples = frame_count * params[0]
    if params[1] == 1 and params[4] == "NONE":
        # 8-bit PCM silence is midpoint 0x80
        return bytes([0x80]) * total_samples
    return b"\x00" * (total_samples * params[1])


def concatenate_parts(wav_paths: Iterable[Path], output_path: Path) -> Path:
    ordered_parts = sorted(wav_paths)
    if not ordered_parts:
        raise ValueError("No .wav files found to concatenate")

    combined = bytearray()
    first_params: AudioParams | None = None
    for idx, wav_path in enumerate(ordered_parts):
        params, frames = _read_wav(wav_path)
        if first_params is None:
            first_params = params
            combined.extend(frames)
            continue

        _assert_compatible(first_params, params, wav_path)
        gap_seconds = 1.0 if idx == 1 else 0.5
        combined.extend(_silence(gap_seconds, first_params))
        combined.extend(frames)

    assert first_params is not None  # for type checkers
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as out_wav:
        nchannels, sampwidth, framerate, _, comptype, compname = first_params
        out_wav.setparams((nchannels, sampwidth, framerate, 0, comptype, compname))
        out_wav.writeframes(combined)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate chapter audio part .wav files.")
    parser.add_argument("--chapters-wav-dir", required=True, type=Path, help="Directory containing one subdirectory per chapter with .wav parts")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory where concatenated .wav files will be written")

    args = parser.parse_args()

    input_dir: Path = args.chapters_wav_dir
    output_dir: Path = args.output_dir
    if not input_dir.is_dir():
        parser.error(f"--chapters-wav-dir must be a directory: {input_dir}")

    chapter_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    if not chapter_dirs:
        parser.error(f"No chapter directories found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0

    for chapter_dir in chapter_dirs:
        wav_parts = sorted(chapter_dir.glob("*.wav"))
        if not wav_parts:
            print(f"Warning: no .wav files found in {chapter_dir}, skipping", file=sys.stderr)
            continue

        output_path = output_dir / f"{chapter_dir.name}.wav"
        try:
            concatenate_parts(wav_parts, output_path)
        except Exception as exc:  # keep CLI concise per chapter
            print(f"Error processing {chapter_dir.name}: {exc}", file=sys.stderr)
            continue

        success_count += 1
        print(f"Wrote concatenated audio to {output_path}")

    if success_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
