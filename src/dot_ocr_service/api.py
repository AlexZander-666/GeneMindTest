import io
import os

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from .inference import DEFAULT_PROMPT, MODEL_ID, create_pipeline

LOAD_IN_4BIT = os.environ.get("DOT_OCR_LOAD_IN_4BIT", "1") != "0"
MAX_NEW_TOKENS = int(os.environ.get("DOT_OCR_MAX_NEW_TOKENS", "2048"))
DO_SAMPLE = os.environ.get("DOT_OCR_DO_SAMPLE", "0") == "1"
MAX_IMAGE_SIZE = int(os.environ.get("DOT_OCR_MAX_IMAGE_SIZE", "1024"))

pipeline = create_pipeline(
    load_in_4bit=LOAD_IN_4BIT,
    max_image_size=MAX_IMAGE_SIZE,
)
device = pipeline.device

app = FastAPI(
    title="dots.ocr OCR Service",
    description="小红书 dots.ocr 模型 HTTP 接口（GPU 优先）",
    version="1.0.0",
)


def ocr_pil_image(image: Image.Image, prompt: str = DEFAULT_PROMPT) -> str:
    """
    对一张 PIL 图片做 OCR，返回识别文本。
    """
    return pipeline.run(
        image,
        prompt=prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
    )


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """上传一张图片，返回识别出的文字。"""
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        text = ocr_pil_image(image)
        return JSONResponse({
            "code": 0,
            "msg": "ok",
            "text": text,
            "device": str(device)
        })
    except Exception as e:
        import traceback
        return JSONResponse({
            "code": 1,
            "msg": f"error: {str(e)}",
            "traceback": traceback.format_exc()
        }, status_code=500)


@app.get("/")
async def root():
    return {
        "service": "dots.ocr",
        "status": "running",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model": MODEL_ID
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "cuda_available": torch.cuda.is_available()}
