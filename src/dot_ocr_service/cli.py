import argparse
from datetime import datetime
from pathlib import Path

from .inference import DEFAULT_PROMPT, create_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE = PROJECT_ROOT / "samples" / "test.png"
OUTPUT_DIR = PROJECT_ROOT / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dots.ocr 推理脚本")
    parser.add_argument(
        "image_path",
        nargs="?",
        default=str(DEFAULT_SAMPLE),
        help="待识别的图片路径，默认 samples/test.png",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="自定义提示词，不填则使用默认 OCR 提示",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="生成文本的最大 token 数，默认为 2048",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="启用采样生成（默认关闭，即贪心搜索）",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="关闭 4bit 量化加载，显存足够时可使用",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1024,
        help="图像长边最大尺寸（像素），超过会等比缩放。默认 1024，设为 0 表示不限制",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image_path).expanduser()
    if not image_path.exists():
        raise FileNotFoundError(
            f"找不到图片文件 '{image_path}'，"
            "请检查路径或指定正确的文件"
        )

    pipeline = create_pipeline(
        load_in_4bit=not args.no_4bit,
        max_image_size=args.max_image_size,
    )
    print(f"\n处理图片: {image_path}")
    result = pipeline.run_from_path(
        str(image_path),
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
    )

    print("\n" + "=" * 50)
    print("OCR 结果:")
    print("=" * 50)
    print(result)

    # 确保输出目录存在
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 生成输出文件路径
    image_name = image_path.stem
    output_file = OUTPUT_DIR / f"{image_name}.md"

    # 生成 MD 内容
    md_content = f"""---
source: {image_path.name}
processed_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
---

# OCR 识别结果

{result}
"""

    # 写入文件
    output_file.write_text(md_content, encoding='utf-8')
    print(f"\n结果已保存至: {output_file}")


if __name__ == "__main__":
    main()
