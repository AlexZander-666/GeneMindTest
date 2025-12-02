"""dot_ocr_service package."""

from .inference import DEFAULT_PROMPT, MODEL_ID, DotOCRInference, create_pipeline

__all__ = ["DotOCRInference", "create_pipeline", "DEFAULT_PROMPT", "MODEL_ID"]
