"""
Inference script for the fine-tuned Qwen3-VL drawing feature detection model.

Usage:
    # Single image (uses the trained detection prompt automatically)
    python src/inference.py \
        --model_path outputs/qwen3-vl-3b-lora/final \
        --base_model merve/qwen3-vl-3b-llava-1pct \
        --image path/to/drawing.png

    # Single image with merged model (no --base_model needed)
    python src/inference.py \
        --model_path outputs/qwen3-vl-3b-merged \
        --image path/to/drawing.png

    # Batch inference on a directory — saves results.json
    python src/inference.py \
        --model_path outputs/qwen3-vl-3b-merged \
        --image_dir data/IM_D03_PT_5K \
        --output results.json
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Prompts (must match training format) ──────────────────────────────────────

# System prompt — matches train_config.yaml system_prompt
DEFAULT_SYSTEM_PROMPT = (
    "你是一个专业的工业图纸分析助手。你的任务是识别和分析工程图纸中的工件特征，"
    "包括但不限于：螺纹孔、圆角、矩形孔、圆孔、长圆孔、倒角、槽等特征。"
    "对于每种特征，请输出其类型、规格尺寸（如M8、R3）以及在图纸中的位置坐标。"
    "请根据图纸内容给出准确、完整的分析结果。"
)

# Detection prompt — matches prepare_data.py HUMAN_PROMPT (without <image>)
# The image is injected separately as a content part in predict().
DETECTION_PROMPT = (
    "你是一个专门用于分析工程图纸的AI视觉系统。\n\n"
    "任务：\n对提供的工程图纸图像进行目标检测。\n\n"
    "对于每个检测到的对象，提取：\n"
    "- category（类别）\n"
    "- bbox（图像坐标边界框）\n\n"
    "需要提取的类别有：\n"
    "- Threaded Hole, Threaded Hole Group\n"
    "- Fillet, Fillet Group\n"
    "- Round Hole, Round Hole Group\n"
    "- Slotted Hole, Slotted Hole Group\n"
    "- Rectangular Hole, Rectangular Hole Group\n"
    "其中，Group 仅为描述性特征，Group 内包含的特征仍需单独列举\n\n"
    "边界框格式（整数）：\n[x_min, y_min, x_max, y_max]\n\n"
    "规则：\n"
    "- 检测所有可见的符号、标注、尺寸、表格、标题栏以及几何元素。\n"
    "- 使用像素坐标。\n"
    "- bbox 必须由四个整数值组成，且满足 x_min < x_max 且 y_min < y_max。\n"
    "- 不要输出解释。\n"
    "- 不要输出 Markdown 格式。\n"
    "- 仅输出严格 json。\n"
    "- 不要添加任何额外字段。\n"
    "- 若未检测到对象，返回空数组。\n"
    "- 输出必须严格遵守以下格式。\n\n"
    "输出示例（禁止修改）\n"
    "[\n"
    "    {\n"
    '        "category": "Round Hole",\n'
    '        "bbox": [x_min, y_min, x_max, y_max]\n'
    "    },\n"
    "    ...\n"
    "]\n\n"
    "图纸："
)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(
    model_path: str,
    base_model: Optional[str] = None,
    device: str = "cuda",
    dtype: str = "bf16",
    use_flash_attn: bool = False,
):
    """
    Load the fine-tuned model.

    Detects automatically whether model_path is a LoRA adapter or a full
    (merged) model. For LoRA adapters, --base_model must be provided.
    """
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "device_map": "auto",
        "attn_implementation": attn_impl,
    }

    is_lora = (Path(model_path) / "adapter_config.json").exists()

    if is_lora:
        if base_model is None:
            raise ValueError(
                "--model_path contains a LoRA adapter but --base_model is not set."
            )
        logger.info(f"Loading base model: {base_model}")
        model = AutoModelForVision2Seq.from_pretrained(base_model, **model_kwargs)
        logger.info(f"Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        logger.info("LoRA weights merged")
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    else:
        logger.info(f"Loading full model: {model_path}")
        model = AutoModelForVision2Seq.from_pretrained(model_path, **model_kwargs)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model.eval()
    return model, processor


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(
    model,
    processor,
    image_path: str,
    prompt: str = DETECTION_PROMPT,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_new_tokens: int = 1024,
    min_pixels: int = 256,
    max_pixels: int = 768,
) -> str:
    """
    Run inference on a single image and return the raw model output string.

    min_pixels / max_pixels are in 28x28-patch units, matching the training
    convention (total pixels = n_patches * 28 * 28).
    """
    min_pix = min_pixels * 28 * 28
    max_pix = max_pixels * 28 * 28

    # Strip <image> placeholder — the image is injected as a content part below
    prompt_text = prompt.replace("<image>", "").rstrip()

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{os.path.abspath(image_path)}",
                    "min_pixels": min_pix,
                    "max_pixels": max_pix,
                },
                {"type": "text", "text": prompt_text},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


# ── Output parsing and display ────────────────────────────────────────────────

def parse_detections(raw: str) -> Optional[list]:
    """Try to parse the model output as a JSON detection list. Returns None on failure."""
    text = raw.strip()
    # Model may wrap output in a markdown code block
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def print_detections(image_name: str, detections: list):
    """Print detections as an ASCII table."""
    print(f"\n  Image : {image_name}")
    print(f"  Found : {len(detections)} detection(s)")
    if not detections:
        return
    print(f"  {'#':<4} {'Category':<22} {'Size':<10} {'BBox'}")
    print(f"  {'-'*4} {'-'*22} {'-'*10} {'-'*30}")
    for i, d in enumerate(detections, 1):
        cat  = d.get("category", "?")
        size = d.get("size", "")
        bbox = d.get("bbox", [])
        print(f"  {i:<4} {cat:<22} {size:<10} {bbox}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inference with fine-tuned Qwen3-VL drawing feature detection model"
    )
    parser.add_argument(
        "--model_path", required=True,
        help="Path to fine-tuned model or LoRA adapter directory",
    )
    parser.add_argument(
        "--base_model", default=None,
        help="Base model path or HF repo (required if model_path is a LoRA adapter)",
    )
    parser.add_argument("--image",     default=None, help="Path to a single image")
    parser.add_argument("--image_dir", default=None, help="Directory of images for batch inference")
    parser.add_argument(
        "--output", default=None,
        help="Output JSON file path (batch mode default: inference_results.json)",
    )
    parser.add_argument("--dtype",    default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument(
        "--min_pixels", type=int, default=256,
        help="Min image resolution in 28x28-patch units (default: 256)",
    )
    parser.add_argument(
        "--max_pixels", type=int, default=768,
        help="Max image resolution in 28x28-patch units (default: 768, matches training)",
    )
    parser.add_argument(
        "--flash_attn", action="store_true",
        help="Use Flash Attention 2 (Linux/WSL only; not supported on native Windows)",
    )
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Specify --image for single inference or --image_dir for batch.")

    model, processor = load_model(
        model_path=args.model_path,
        base_model=args.base_model,
        dtype=args.dtype,
        use_flash_attn=args.flash_attn,
    )

    predict_kwargs = dict(
        model=model,
        processor=processor,
        prompt=DETECTION_PROMPT,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        max_new_tokens=args.max_new_tokens,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    if args.image:
        # ── Single image ──
        logger.info(f"Processing: {args.image}")
        raw = predict(image_path=args.image, **predict_kwargs)

        detections = parse_detections(raw)
        if detections is not None:
            print_detections(Path(args.image).name, detections)
        else:
            print("\n[Warning] Output is not valid JSON. Raw model output:")
            print(raw)

        if args.output:
            out = {
                "dataitem_name": args.image,
                "result": detections if detections is not None else [],
                "raw": raw,
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved to {args.output}")

    else:
        # ── Batch mode ──
        image_dir = Path(args.image_dir)
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
        image_files = sorted(f for f in image_dir.rglob("*") if f.suffix.lower() in exts)

        output_path = Path(args.output or "inference_results.json")
        results = []

        with tqdm(total=len(image_files), unit="img", dynamic_ncols=True) as pbar:
            for img_path in image_files:
                pbar.set_description(img_path.name[:40])
                try:
                    raw = predict(image_path=str(img_path), **predict_kwargs)
                    detections = parse_detections(raw)
                    results.append({
                        "dataitem_name": str(img_path.relative_to(image_dir)),
                        "result": detections if detections is not None else [],
                        "raw": raw,
                    })
                except Exception as e:
                    results.append({
                        "dataitem_name": str(img_path.relative_to(image_dir)),
                        "error": str(e),
                    })

                # Write after every image so a crash doesn't lose prior results
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

                pbar.update(1)

        logger.info(f"Done. {len(results)} results saved to {output_path}")


if __name__ == "__main__":
    main()
