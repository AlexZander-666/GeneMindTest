# dot.ocr OCR service

This directory hosts a FastAPI + CLI wrapper around the Hugging Face `rednote-hilab/dots.ocr` model. The `DotOCRInference` helper centralizes tokenizer/image-processor/model loading, handles 4-bit quantization, resizes large images, and exposes both a CLI (`dot_ocr_service.cli`) and HTTP service (`dot_ocr_service.api`).

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
3. Make the `src/` package discoverable (choose one):
   - Activate it per-shell:
     ```powershell
     set PYTHONPATH=src
     ```
   - Or install the local package in editable mode (adds `dot_ocr_service` to the interpreter permanently for this venv):
     ```powershell
     pip install -e .
     ```
4. Run inference against the bundled sample:
   ```powershell
   python -m dot_ocr_service.cli samples/test.png
   ```
5. Launch the FastAPI service:
   ```powershell
   set DOT_OCR_LOAD_IN_4BIT=1
   set DOT_OCR_MAX_IMAGE_SIZE=1024
   uvicorn dot_ocr_service.api:app --host 0.0.0.0 --port 8000
   ```
   The service inherits defaults from `inference.create_pipeline()`, but you can override `DOT_OCR_MAX_NEW_TOKENS`, `DOT_OCR_DO_SAMPLE`, `DOT_OCR_LOAD_IN_4BIT`, and `DOT_OCR_MAX_IMAGE_SIZE` via environment variables.

## Files

- `src/dot_ocr_service/api.py`: FastAPI entry point that accepts image uploads and reuses the cached inference pipeline.
- `src/dot_ocr_service/cli.py`: CLI script with flags for sampling, 4-bit control, and max image size; defaults to the `samples/test.png`.
- `src/dot_ocr_service/inference.py`: Shared inference helper that configures Hugging Face components, resizes images, builds prompts, and keeps any cached pipelines.
- `output/`: CLI generated markdown exports (git-ignored).
- `samples/`: Stores the included `test.png` for quick smoke tests.
- `docs/dot_ocr_deployment.md`: Deployment troubleshooting notes gathered during initial setup.

## Notes

- Download and inference require GPU drivers if `cuda` is selected; the helper falls back to CPU when GPU is unavailable.
- Large images are resized to a 1024 px long edge by default (`DotOCRInference.max_image_size`). Override via `python -m dot_ocr_service.cli --max-image-size`/`DOT_OCR_MAX_IMAGE_SIZE` or directly via `create_pipeline`.
- Keep an eye on the deployment notes in `docs/` for issues encountered while setting up the original environment.
