# dot.ocr OCR service

This directory hosts a FastAPI + CLI wrapper around the Hugging Face `rednote-hilab/dots.ocr` model. The `DotOCRInference` helper centralizes tokenizer/image-processor/model loading, handles 4-bit quantization, resizes large images, and exposes both a CLI (`ocr_infer.py`) and HTTP service (`app.py`).

## Quick start

1. Create a Python 3.10/3.11 virtual environment and activate it:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Run inference against the bundled sample:
   ```powershell
   python ocr_infer.py samples/test.png
   ```
4. Launch the FastAPI service:
   ```powershell
   set DOT_OCR_LOAD_IN_4BIT=1
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
   The service inherits defaults from `inference.create_pipeline()`, but you can override `DOT_OCR_MAX_NEW_TOKENS`, `DOT_OCR_DO_SAMPLE`, and `DOT_OCR_LOAD_IN_4BIT` via environment variables.

## Files

- `app.py`: FastAPI entry point that accepts image uploads and reuses the cached inference pipeline.
- `ocr_infer.py`: CLI script with flags for sampling, 4-bit control, and max image size; defaults to the `samples/test.png`.
- `inference.py`: Shared inference helper that configures Hugging Face components, resizes images, builds prompts, and keeps any cached pipelines.
- `samples/`: Stores the included `test.png` for quick smoke tests.
- `docs/dot_ocr_deployment.md`: Deployment troubleshooting notes gathered during initial setup.

## Notes

- Download and inference require GPU drivers if `cuda` is selected; the helper falls back to CPU when GPU is unavailable.
- Large images are resized to a 1024 px long edge by default (`DotOCRInference.max_image_size`). Override via `ocr_infer.py --max-image-size` or `create_pipeline`.
- Keep an eye on the deployment notes in `docs/` for issues encountered while setting up the original environment.
