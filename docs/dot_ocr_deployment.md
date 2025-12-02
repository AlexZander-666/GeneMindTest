# dot.ocr OCR 模型部署问题总结

## Executive Summary
- 成功在 Windows + RTX 3060 (6 GiB) 环境下启用了 `dot.ocr`，并梳理出可复用的 inference 模块供 CLI/服务共享。
- 避免安装 `dots-ocr` 包、改用 Transformers + Hugging Face checkpoint，绕开了 Windows 上的 `flash-attn` 编译难题。
- 引入 4-bit 量化、图像尺寸限制与 CUDA 内存配置后，显存压力大幅下降，6 GiB GPU 也能跑通 inference。

## 项目概况
- **目标**：在本地 Windows 上部署 `rednote-hilab/dots.ocr` OCR 模型，提供 FastAPI + CLI 双通道推理接口，支持通过 `curl` 上传图片。
- **技术栈**：Python 3.10、PyTorch (CUDA 12.1)、Transformers、FastAPI/Uvicorn、BitsAndBytes 4-bit、RTX 3060 Laptop GPU (6 GiB)。
- **模型来源**：Hugging Face `rednote-hilab/dots.ocr`（主模型）与 `.base` 轻量版本；搭配 Qwen2-VL tokenizer + `Qwen2VLImageProcessor`。
- **部署亮点**：统一 `DotOCRInference` 模块，CLI/服务共享；环境变量控制 4-bit/FP16/CPU；增加图像预处理与 dtype 补丁以稳定低显存下的推理。

## 关键问题与应对

### 一、环境与依赖
1. **Python 版本冲突**：3.13.9 无法安装 PyTorch。**处理**：重建 Python 3.10 虚拟环境。
2. **镜像源 SSL 错误**：多数国内源因证书导致连接失败。**处理**：PyTorch 使用官方 `https://download.pytorch.org/whl/cu121`，其余包用清华镜像或备用源。
3. **dots-ocr 包不可用**：PyPI 无包，GitHub 依赖 `flash-attn` 需 Windows 本地编译。**处理**：直接 `pip install transformers accelerate qwen-vl-utils`，通过 `AutoModelForCausalLM` 加载 Hugging Face 权重，绕过 `dots-ocr` 包。
4. **模型下载超时**：Hugging Face 下载在国内经常失败。**处理**：设定 `HF_ENDPOINT` 为镜像，`HF_HUB_DOWNLOAD_TIMEOUT=300`，保证模型与 tokenizer 可靠拉取。

### 二、模型组件与推理流程
5. **AutoProcessor 加载失败（缺 video_processor）**：dots.ocr 没有 video component，`AutoProcessor` 抛出 `TypeError`。**处理**：分别加载 `AutoTokenizer` + `Qwen2VLImageProcessor`，手动合并输出。
6. **image_grid_thw 等张量丢失导致 `NoneType`**：只传 `pixel_values` 入 `model.generate()`。**处理**：保留 processor 返回的所有键，和文本 input 一起 `to(device)` 并合并传入。
7. **Prompt 与 img_mask 不匹配**：未插入 `<|imgpad|>`，`vision_embeddings` 与 `img_mask.sum()` 断言失败。**处理**：根据 `image_grid_thw` 计算所需 image token 个数，在 prompt 末尾补足 `<|imgpad|>`。
8. **代码重复与参数漂移**：CLI (`dot_ocr_service.cli`) 与 FastAPI (`dot_ocr_service.api`) 若各自维护 tokenizer/model，4-bit/FP16 控制容易漂移。**处理**：抽象 `DotOCRInference` 模块（`dot_ocr_service.inference`），CLI/FastAPI 共享、由参数/环境变量控制 dtype/量化，避免重复 patch。

### 三、量化、显存与性能
9. **CUDA OOM（Vision Tower attention）**：`device_map="auto"` 让全部参数上 GPU；高 token 数 attention 爆显存。**处理**：启用 `BitsAndBytesConfig(load_in_4bit=True)`、限长 `max_new_tokens`、图像最长边 ≤ 1024px，必要时退到 CPU。
10. **4-bit bfloat16/bias dtype 不一致**：`conv` 层要求 activation & bias 同 dtype。**处理**：将 `bnb_4bit_compute_dtype` 强制为 `torch.float16` + `bnb_4bit_use_double_quant=True`，确保 bias 与激活一致。
11. **vision_tower bf16 补丁失效**：属性挂在模型自身，补丁未命中。**处理**：补丁函数检查 `model.vision_tower` 与 `model.model.vision_tower`，在 patch 中强制 `hidden_states` 转 `float16` 并 `bf16=False`，并加 `_bf16_disabled` 标记。
12. **4-bit 量化后仍 OOM（attention activation 未量化）**：量化只针对权重，activation 仍 FP16，占用大量显存。**处理**：增加图像缩放（长边 ≤ `max_image_size`）、设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、引入 `--max-image-size` CLI 参数，使 RTX 3060 在 6 GiB 显存下也能跑通。

## 经验与建议
- **先确认依赖兼容性**：PyTorch 官方支持 Python 3.8-3.12，优选 3.10/3.11；核心框架用官方源，其余包用国内镜像。
- **绕开不可用包**：未发布的 `dots-ocr` 可由 transformers + qwen 工具链直接替代，避免 Windows 编译问题。
- **多模态模型要手动控制组件**：AutoProcessor 可能不适用，务必分别请求 tokenizer/image_processor，并合并输出。
- **多输入张量不可漏传**：`image_grid_thw`、`img_mask` 等必须与 prompt 一起传入，防止 `NoneType`/断言错误。
- **统一推理模块**：抽象 `DotOCRInference`，CLI + FastAPI 共享，在一个点控制 dtype、量化、patch，更易维护。
- **量化只是权重**：attention activation 仍是显存瓶颈，需从输入尺寸、token 数、内存分配等维度优化。
- **CPU fallback 也要准备**：显存有限的设备可通过 `CUDA_VISIBLE_DEVICES=` 或 `device="cpu"`，配合 batch/size 降低压力。
- **持续记录显存策略**：对于 RTX 3060 这类 6 GiB GPU，4-bit + 图像长边控制 + `expandable_segments` 已成为部署最佳实践。
