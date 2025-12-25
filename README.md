# book-ai-ocr-tool

CLI to OCR book page images using OpenAI GPT structured output.

## Setup

```bash
uv sync
```

Ensure your API key is available in `OPENAI_API_KEY` (or set `--api-key-env`).

## Usage

```bash
python book_ocr.py --input-dir path/to/images --output-dir path/to/output
```

Options:
- `--model` (default `gpt-5-mini`)
- `--api-key-env` (default `OPENAI_API_KEY`)

Outputs one JSON per input image containing page number, text, optional confidence and warnings.
