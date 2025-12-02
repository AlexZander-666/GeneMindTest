import os
from functools import lru_cache
from types import MethodType
from typing import Optional, Sequence

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen2VLImageProcessor,
)

# 配置 Hugging Face 环境，加速模型下载
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# 优化 CUDA 内存分配，减少碎片化
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = "rednote-hilab/dots.ocr"
DEFAULT_PROMPT = (
    "Please extract all text content from this image. "
    "Output only the text without any formatting."
)


def _disable_vision_bf16(model) -> None:
    """
    dots.ocr 的视觉塔默认会把输入强制转换为 bf16。
    在 4bit 量化 + RTX 30 系列 GPU 上会导致卷积 bias (fp16) 与输入 dtype 不匹配。
    这里通过给 vision_tower.forward 打补丁，确保量化推理统一保持 fp16。
    """
    try:
        # vision_tower 直接在 DotsOCRForCausalLM 上，不在 model.model 里
        vision_tower = getattr(model, "vision_tower", None)
        if vision_tower is None:
            # 兼容其他可能的模型结构
            base_model = getattr(model, "model", model)
            vision_tower = getattr(base_model, "vision_tower", None)

        if vision_tower is None or getattr(vision_tower, "_bf16_disabled", False):
            return

        original_forward = vision_tower.forward

        def forward_without_bf16(self, hidden_states, grid_thw, bf16=None):
            if hidden_states.dtype != torch.float16:
                hidden_states = hidden_states.to(torch.float16)
            return original_forward(hidden_states, grid_thw, bf16=False)

        vision_tower.forward = MethodType(forward_without_bf16, vision_tower)
        vision_tower._bf16_disabled = True
        print("Successfully patched vision_tower to disable bf16")
    except AttributeError:
        pass


class DotOCRInference:
    """
    封装 dots.ocr 推理流程，统一处理 tokenizer、image_processor 与模型输入。
    """

    def __init__(self, load_in_4bit: bool = True, max_image_size: int = 1024):
        """
        初始化推理管道。

        Args:
            load_in_4bit: 是否使用 4bit 量化加载模型，节省显存。
            max_image_size: 图像长边的最大尺寸（像素），超过此尺寸的图片会被等比缩放。
                           设为 0 或负数表示不限制。默认 1024，适合 6GB 显存的 GPU。
        """
        self.max_image_size = max_image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

        print("Loading image processor...")
        self.image_processor = Qwen2VLImageProcessor.from_pretrained(
            MODEL_ID, trust_remote_code=True
        )

        print("Loading model...")
        self.model = self._load_model(load_in_4bit=load_in_4bit)
        self.model.eval()
        self.image_token_id = getattr(self.model.config, "image_token_id", None)
        tokenizer_image_token = getattr(self.tokenizer, "image_token", None)
        if tokenizer_image_token is None and self.image_token_id is not None:
            tokenizer_image_token = self.tokenizer.convert_ids_to_tokens([self.image_token_id])[0]
        self.image_token = tokenizer_image_token or "<|imgpad|>"
        print("Model loaded successfully!")

    def _load_model(self, load_in_4bit: bool):
        if self.device.type == "cuda":
            if load_in_4bit:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    quantization_config=quant_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
                _disable_vision_bf16(model)
                return model

            return AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        return model.to(self.device)

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """
        将图像长边缩放到 max_image_size，保持宽高比。
        这是解决 OOM 的关键：减少视觉 token 数量可大幅降低 attention 显存占用。
        """
        if self.max_image_size <= 0:
            return image
        w, h = image.size
        if max(w, h) <= self.max_image_size:
            return image
        scale = self.max_image_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        print(f"Resizing image from {w}x{h} to {new_w}x{new_h} (max_size={self.max_image_size})")
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def run(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        *,
        max_new_tokens: int = 2048,
        do_sample: bool = False,
        **generate_kwargs,
    ) -> str:
        # 缩放图像以防止 OOM
        image = self._resize_image(image)

        image_inputs, image_token_counts = self._prepare_image_inputs(image)
        prompt = prompt or DEFAULT_PROMPT
        text = self._build_prompt(prompt, image_token_counts)
        text_inputs = self._prepare_text_inputs(text)
        inputs = {**text_inputs, **image_inputs}

        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        gen_kwargs.update(generate_kwargs)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        input_length = inputs["input_ids"].shape[1]
        generated_ids_trimmed = generated_ids[:, input_length:]

        output_text = self.tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text

    def run_from_path(self, image_path: str, **kwargs) -> str:
        image = Image.open(image_path).convert("RGB")
        return self.run(image, **kwargs)

    def _build_prompt(self, prompt: str, image_token_counts: Optional[Sequence[int]] = None) -> str:
        placeholder = self._build_image_placeholders(image_token_counts)
        prompt_text = f"{prompt}\n{placeholder}" if placeholder else prompt
        messages = [{"role": "user", "content": prompt_text}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _prepare_image_inputs(self, image: Image.Image):
        image_inputs = self.image_processor(images=image, return_tensors="pt")
        image_token_counts = self._count_image_tokens(image_inputs.get("image_grid_thw"))
        image_inputs = self._to_device(image_inputs)
        return image_inputs, image_token_counts

    def _prepare_text_inputs(self, prompt_text: str):
        text_inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=True)
        return self._to_device(text_inputs)

    def _count_image_tokens(self, image_grid_thw):
        if image_grid_thw is None:
            return []
        grid_tensor = torch.as_tensor(image_grid_thw)
        merge_length = self.image_processor.merge_size ** 2
        counts = (grid_tensor.prod(dim=1) // merge_length).tolist()
        return [max(0, int(count)) for count in counts]

    def _build_image_placeholders(self, image_token_counts: Optional[Sequence[int]]) -> str:
        if not image_token_counts:
            return ""
        token = getattr(self, "image_token", "<|imgpad|>")
        segments = []
        for count in image_token_counts:
            if count <= 0:
                continue
            segments.append(" ".join([token] * count))
        return "\n".join(segments)

    def _to_device(self, batch):
        return {
            k: (v.to(self.device) if hasattr(v, "to") else v)
            for k, v in batch.items()
        }


@lru_cache(maxsize=4)
def create_pipeline(load_in_4bit: bool = True, max_image_size: int = 1024) -> DotOCRInference:
    """
    创建（或复用缓存的）推理实例，避免重复加载大模型。
    cache key 由 load_in_4bit 和 max_image_size 决定，可在不同配置间切换。

    Args:
        load_in_4bit: 是否使用 4bit 量化，默认 True。
        max_image_size: 图像长边最大尺寸，默认 1024。设为 0 表示不限制。
    """
    return DotOCRInference(load_in_4bit=load_in_4bit, max_image_size=max_image_size)
