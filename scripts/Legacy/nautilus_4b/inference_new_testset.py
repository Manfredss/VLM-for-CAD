"""
inference_new_testset.py — Run inference on new_testset_50 with custom prompt.

Usage:
  python3 inference_new_testset.py \
      --model_path Qwen/Qwen3.5-4B \
      --adapter_path /workspace/finetune/output/swift_4b_aug/.../checkpoint-3800 \
      --image_dir /workspace/finetune/data/data/new_testset_50 \
      --test_json /workspace/finetune/data/data/new_testset_50.json \
      --prompt_file /workspace/finetune/data/data/prompt5.txt \
      --output_file /workspace/finetune/output/infer_new50_aug.json \
      --load_in_4bit
"""

import os
import re
import json
import time
import argparse
import logging
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "你是一个专业的工业图纸分析助手。你的任务是识别和分析工程图纸中的工件特征，"
    "包括但不限于：螺纹孔、圆角、矩形孔、圆孔、长圆孔、倒角、槽等特征。"
    "对于每种特征，请输出其类型、规格尺寸（如M8、R3）以及在图纸中的位置坐标。"
    "请根据图纸内容给出准确、完整的分析结果。"
)


def load_prompt(prompt_file: str) -> str:
    """Load the detection prompt from a file, stripping the wrapping variable assignment."""
    text = Path(prompt_file).read_text(encoding="utf-8")
    # Strip FEATURE_PROMPT = \"\"\" ... \"\"\" wrapper if present
    match = re.search(r'"""(.*?)"""', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def load_model(model_path: str, adapter_path: str | None, load_in_4bit: bool):
    from transformers import AutoProcessor, BitsAndBytesConfig

    try:
        from transformers import AutoModelForImageTextToText as _ModelCls
    except ImportError:
        try:
            from transformers import AutoModelForVision2Seq as _ModelCls
        except ImportError:
            from transformers import AutoModelForCausalLM as _ModelCls

    logger.info(f"Loading processor: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    logger.info(f"Loading model: {model_path}  (4-bit={load_in_4bit})")
    model_kwargs = dict(
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model_kwargs["torch_dtype"] = torch.bfloat16

    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Flash Attention 2 enabled")
    except ImportError:
        model_kwargs["attn_implementation"] = "sdpa"

    model = _ModelCls.from_pretrained(model_path, **model_kwargs)

    if adapter_path:
        from peft import PeftModel
        logger.info(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        logger.info(f"  GPU {i} VRAM: {alloc:.2f} GB")

    return model, processor


def run_inference_single(model, processor, image_path: Path,
                         prompt: str, max_new_tokens: int) -> str:
    try:
        from qwen_vl_utils import process_vision_info
        _use_qwen_utils = True
    except ImportError:
        _use_qwen_utils = False

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{str(image_path)}"},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    if _use_qwen_utils:
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs if video_inputs else None,
            return_tensors="pt",
        )
    else:
        from PIL import Image as PILImage
        img = PILImage.open(image_path).convert("RGB")
        inputs = processor(text=[text], images=[img], return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.05,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_ids, skip_special_tokens=True)


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def parse_output(raw_text: str):
    text = _strip_think_tags(raw_text).strip()

    code_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if code_match:
        try:
            obj = json.loads(code_match.group(1))
            if isinstance(obj, list):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        try:
            obj = json.loads(arr_match.group())
            if isinstance(obj, list):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # Truncated JSON recovery
    last_brace = text.rfind("}")
    if last_brace >= 0:
        candidate = text[:last_brace + 1]
        first_bracket = candidate.find("[")
        if first_bracket >= 0:
            candidate = candidate[first_bracket:]
            if not candidate.rstrip().endswith("]"):
                candidate = candidate.rstrip().rstrip(",") + "]"
            try:
                obj = json.loads(candidate)
                if isinstance(obj, list):
                    logger.info(f"Recovered {len(obj)} items from truncated JSON")
                    return obj
            except (json.JSONDecodeError, ValueError):
                pass

    logger.warning(f"JSON parse failed. raw output: {text[:200]!r}")
    return []


def _deduplicate(detections: list) -> list:
    seen = set()
    result = []
    for item in detections:
        key = (
            item.get("category"),
            item.get("size", ""),
            tuple(item.get("bbox", item.get("bbox_2d", []))),
        )
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load prompt
    prompt = load_prompt(args.prompt_file)
    logger.info(f"Prompt loaded ({len(prompt)} chars)")

    # Load test JSON to get image list
    with open(args.test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    image_names = [item["dataitem_name"] for item in test_data]
    logger.info(f"Test set: {len(image_names)} images")

    # Load model
    model, processor = load_model(args.model_path, args.adapter_path, args.load_in_4bit)

    results = []
    t_start = time.time()

    for i, img_name in enumerate(image_names):
        img_path = image_dir / img_name
        if not img_path.exists():
            logger.warning(f"[{i+1}/{len(image_names)}] Image not found: {img_name}")
            results.append({"dataitem_name": img_name, "result": [], "raw": "[]", "_error": "image not found"})
            continue

        t0 = time.time()
        try:
            raw_text = run_inference_single(model, processor, img_path, prompt, args.max_new_tokens)
            parsed = _deduplicate(parse_output(raw_text))
            error = None
        except Exception as e:
            logger.error(f"[{i+1}/{len(image_names)}] ERROR on {img_name}: {e}")
            raw_text = ""
            parsed = []
            error = str(e)

        elapsed = time.time() - t0
        total_done = i + 1
        avg_time = (time.time() - t_start) / total_done
        eta_sec = avg_time * (len(image_names) - total_done)
        eta_str = f"{int(eta_sec // 3600)}h{int((eta_sec % 3600) // 60)}m"

        logger.info(
            f"[{total_done}/{len(image_names)}] {img_name} | "
            f"{len(parsed)} detections | {elapsed:.1f}s | ETA {eta_str}"
        )

        # Normalize fields
        normalized = []
        for f in parsed:
            normalized.append({
                "category": f.get("category", f.get("label", "")),
                "size": f.get("size", ""),
                "bbox": f.get("bbox", f.get("bbox_2d", [])),
            })

        raw_str = json.dumps(normalized, ensure_ascii=False)
        record = {"dataitem_name": img_name, "result": normalized, "raw": raw_str}
        if error:
            record["_error"] = error
        if not parsed and raw_text:
            record["_raw"] = raw_text
        results.append(record)

        # Save progress every 10 images
        if total_done % 10 == 0 or total_done == len(image_names):
            with open(output_file, "w", encoding="utf-8") as out_f:
                json.dump(results, out_f, ensure_ascii=False, indent=2)
            logger.info(f"  Progress saved -> {output_file}")

    # Final save
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)

    total_det = sum(len(r["result"]) for r in results)
    errors = sum(1 for r in results if "_error" in r)
    logger.info(f"Done. {len(results)} images, {total_det} detections, {errors} errors")
    logger.info(f"Output: {output_file}")


if __name__ == "__main__":
    main()
