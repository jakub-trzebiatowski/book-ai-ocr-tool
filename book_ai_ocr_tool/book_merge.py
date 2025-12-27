import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from book_ai_ocr_tool.models import ImageOCRResult, ChapterContent, ImageOCRParagraph


@dataclass
class MergeConfig:
    input_dir: Path
    output_dir: Path


def parse_args(argv: Sequence[str]) -> MergeConfig:
    parser = argparse.ArgumentParser(description="Merge OCR JSONs from book pages into chapters")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing OCR JSON files")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write merged chapter JSONs")
    args = parser.parse_args(argv)
    return MergeConfig(input_dir=args.input_dir, output_dir=args.output_dir)


def find_ocr_jsons(directory: Path) -> List[Path]:
    """Find all JSON files in directory, sorted by name."""
    files: List[Path] = []
    for entry in directory.iterdir():
        if entry.is_file() and entry.suffix.lower() == ".json":
            files.append(entry)
    return sorted(files)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_ocr_result(json_path: Path) -> ImageOCRResult:
    """Load an OCR result JSON file."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return ImageOCRResult(**data)


def append_paragraphs_with_tag_merge(
    existing: List[ImageOCRParagraph],
    new_paragraphs: List[ImageOCRParagraph],
) -> None:
    for paragraph in new_paragraphs:
        paragraph_copy = ImageOCRParagraph(**paragraph.model_dump())
        if existing and existing[-1].tag == paragraph_copy.tag:
            existing[-1].text = f"{existing[-1].text.rstrip()}\n\n{paragraph_copy.text.lstrip()}"
        else:
            existing.append(paragraph_copy)


def merge_chapters(ocr_results: List[ImageOCRResult]) -> List[ChapterContent]:
    """Merge OCR results by chapter title, tracking current chapter and handling missing titles."""
    chapters: List[ChapterContent] = []
    current_chapter_title: str | None = None
    current_chapter_paragraphs: List[ImageOCRParagraph] = []

    for result in ocr_results:
        for page in result.pages:
            page_title = page.chapter_title

            # Determine the title for this page
            if page_title:
                title_to_use = page_title
                print("Page has chapter title:", title_to_use)
            elif current_chapter_title is not None:
                # Continue with current chapter
                title_to_use = current_chapter_title
                print("Page has no chapter title, continuing with current chapter:", title_to_use)
            else:
                # Page has no chapter title and we haven't set one yet - fail
                raise ValueError("First page or pages after chapter boundary must have a chapter_title set")

            # Check if chapter changed
            if title_to_use != current_chapter_title:
                print("Starting new chapter:", title_to_use)
                # Save previous chapter if it exists
                if current_chapter_title is not None and current_chapter_paragraphs:
                    chapters.append(ChapterContent(title=current_chapter_title, paragraphs=current_chapter_paragraphs))
                # Start new chapter
                current_chapter_title = title_to_use
                current_chapter_paragraphs = []

            append_paragraphs_with_tag_merge(current_chapter_paragraphs, page.paragraphs)

    # Fail if we ended with an unknown chapter
    if current_chapter_title is None:
        raise ValueError("No pages with chapter_title were found")

    # Don't forget the last chapter
    if current_chapter_paragraphs:
        chapters.append(ChapterContent(title=current_chapter_title, paragraphs=current_chapter_paragraphs))

    return chapters


def merge_directory(config: MergeConfig) -> int:
    if not config.input_dir.exists() or not config.input_dir.is_dir():
        print(f"Input directory not found: {config.input_dir}", file=sys.stderr)
        return 2

    ensure_output_dir(config.output_dir)
    json_files = find_ocr_jsons(config.input_dir)

    if not json_files:
        print("No JSON files found to process.", file=sys.stderr)
        return 3

    # Load all OCR results
    try:
        ocr_results = [load_ocr_result(json_file) for json_file in json_files]
        print(f"Loaded {len(json_files)} OCR result files")
    except Exception as exc:  # pragma: no cover - file dependent
        print(f"Failed to load OCR results: {exc}", file=sys.stderr)
        return 4

    # Merge chapters
    try:
        chapters = merge_chapters(ocr_results)
        print(f"Merged into {len(chapters)} chapters")
    except Exception as exc:  # pragma: no cover - logic error
        print(f"Failed to merge chapters: {exc}", file=sys.stderr)
        return 5

    # Clean the output directory

    for existing_file in config.output_dir.iterdir():
        if existing_file.is_file():
            try:
                existing_file.unlink()
                print(f"Removed existing file: {existing_file.name}")
            except Exception as exc:  # pragma: no cover - file dependent
                print(f"Failed to remove existing file {existing_file.name}: {exc}", file=sys.stderr)

    # Write output files
    failures = 0
    for idx, chapter in enumerate(chapters, start=1):
        try:
            # Create sequential filename
            output_filename = f"chapter_{idx:03d}.json"
            output_path = config.output_dir / output_filename

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(chapter.model_dump(), f, ensure_ascii=False, indent=2)

            print(f"Wrote {output_filename}: {chapter.title}")
        except Exception as exc:  # pragma: no cover - file dependent
            failures += 1
            print(f"Failed to write chapter {idx} '{chapter.title}': {exc}", file=sys.stderr)

    if failures:
        print(f"Completed with {failures} failures", file=sys.stderr)
        return 6

    return 0


def main(argv: Iterable[str] | None = None) -> int:
    config = parse_args(list(argv) if argv is not None else sys.argv[1:])
    return merge_directory(config)


if __name__ == "__main__":
    raise SystemExit(main())

