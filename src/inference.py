"""
Inference script for the fine-tuned Qwen3-VL model.

Usage:
    # Single image inference
    python src/inference.py \
        --model_path outputs/qwen3-vl-3b-lora/final \
        --base_model Qwen/Qwen3-VL-3B \
        --image path/to/drawing.png \
        --prompt "请识别这张图纸中的所有工件特征。"

    # Batch inference on a directory of images
    python src/inference.py \
        --model_path outputs/qwen3-vl-3b-lora/final \
        --base_model Qwen/Qwen3-VL-3B \
        --image_dir path/to/images/ \
        --output results.json
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "你是一个专业的工业图纸分析助手。你的任务是识别和分析工程图纸中的工件特征，"
    "包括但不限于：尺寸标注、形位公差、表面粗糙度、螺纹、孔、槽、倒角、圆角等特征。"
    "请根据图纸内容给出准确、详细的分析结果。"
)


def load_model(
    model_path: str,
    base_model: Optional[str] = None,
    device: str = "cuda",
    dtype: str = "bf16",
    use_flash_attn: bool = True,
):
    """
    Load the fine-tuned model.

    If model_path contains LoRA adapter, base_model must be provided.
    If model_path is a full model, base_model is not needed.
    """
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        model_kwargs["attn_implementation"] = "sdpa"

    # Check if this is a LoRA adapter or full model
    adapter_config_path = Path(model_path) / "adapter_config.json"
    is_lora = adapter_config_path.exists()

    if is_lora:
        if base_model is None:
            raise ValueError(
                "model_path contains a LoRA adapter but --base_model is not specified. "
                "Please provide the base model path."
            )
        logger.info(f"Loading base model: {base_model}")
        model = AutoModelForVision2Seq.from_pretrained(
            base_model, **model_kwargs
        )
        logger.info(f"Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()  # Merge LoRA weights for faster inference
        logger.info("LoRA adapter merged into base model")

        # Load processor from adapter dir, fallback to base model
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    else:
        logger.info(f"Loading full model: {model_path}")
        model = AutoModelForVision2Seq.from_pretrained(
            model_path, **model_kwargs
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model.eval()
    return model, processor


def predict(
    model,
    processor,
    image_path: str,
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
    top_p: float = 0.9,
    min_pixels: int = 256,
    max_pixels: int = 1280,
) -> str:
    """Run inference on a single image."""
    # min_pixels / max_pixels = number of 28×28 visual-token patches.
    # Total pixel count = n_patches * 28 * 28 (Qwen-VL convention).
    min_pix = min_pixels * 28 * 28
    max_pix = max_pixels * 28 * 28

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
                {"type": "text", "text": prompt},
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
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )

    # Decode only the generated part
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text


def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned Qwen2.5-VL")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model or LoRA adapter")
    parser.add_argument("--base_model", type=str, default=None, help="Base model path (required for LoRA)")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory of images for batch inference")
    parser.add_argument("--prompt", type=str, default="请识别这张图纸中的所有工件特征，并详细描述。", help="Prompt")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="System prompt")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for batch results")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--no_flash_attn", action="store_true")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Either --image or --image_dir must be specified")

    # Load model
    model, processor = load_model(
        model_path=args.model_path,
        base_model=args.base_model,
        dtype=args.dtype,
        use_flash_attn=not args.no_flash_attn,
    )

    if args.image:
        # Single image inference
        logger.info(f"Processing image: {args.image}")
        result = predict(
            model, processor, args.image, args.prompt,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print("\n" + "=" * 60)
        print("Prediction Result:")
        print("=" * 60)
        print(result)
        print("=" * 60)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump({"image": args.image, "prediction": result}, f, ensure_ascii=False, indent=2)

    elif args.image_dir:
        # Batch inference
        image_dir = Path(args.image_dir)
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
        image_files = sorted([
            f for f in image_dir.rglob("*")
            if f.suffix.lower() in image_extensions
        ])

        logger.info(f"Found {len(image_files)} images in {image_dir}")
        results = []

        for i, img_path in enumerate(image_files):
            logger.info(f"[{i+1}/{len(image_files)}] Processing: {img_path.name}")
            try:
                result = predict(
                    model, processor, str(img_path), args.prompt,
                    system_prompt=args.system_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                results.append({
                    "image": str(img_path.relative_to(image_dir)),
                    "prediction": result,
                })
                print(f"  -> {result[:100]}...")
            except Exception as e:
                logger.error(f"  Failed: {e}")
                results.append({
                    "image": str(img_path.relative_to(image_dir)),
                    "error": str(e),
                })

        # Save results
        output_path = args.output or "inference_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
