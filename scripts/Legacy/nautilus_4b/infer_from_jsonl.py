"""
infer_from_jsonl.py — 从 JSONL 数据集直接推理（使用训练时完全相同的 prompt 格式）

用法:
  python3 infer_from_jsonl.py \
      --model_path Qwen/Qwen3.5-4B \
      --adapter_path /path/to/checkpoint-XXXX \
      --dataset /path/to/test_v2.jsonl \
      --output /path/to/results.jsonl \
      --max_new_tokens 2048
"""

import os
import re
import json
import sys
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


def load_model(model_path, adapter_path):
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as ModelCls
    except ImportError:
        try:
            from transformers import AutoModelForVision2Seq as ModelCls
        except ImportError:
            from transformers import AutoModelForCausalLM as ModelCls

    logger.info(f"Loading processor from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model_kwargs = dict(
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )

    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using Flash Attention 2")
    except ImportError:
        model_kwargs["attn_implementation"] = "sdpa"
        logger.info("Using SDPA attention")

    logger.info(f"Loading model: {model_path}")
    model = ModelCls.from_pretrained(model_path, **model_kwargs)

    if adapter_path:
        from peft import PeftModel
        logger.info(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        logger.info(f"  GPU {i} VRAM: {alloc:.2f} GB")

    return model, processor


def extract_image_paths(messages):
    """Extract image paths from messages content."""
    paths = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    img = part.get("image", "")
                    if img:
                        paths.append(img)
    return paths


def make_inference_messages(sample):
    """Build messages for inference: system + user only (no assistant ground truth)."""
    messages = sample.get("messages", [])
    infer_msgs = []
    for msg in messages:
        if msg["role"] == "assistant":
            break  # stop before ground truth
        infer_msgs.append(msg)
    return infer_msgs


def run_single(model, processor, sample, max_new_tokens, image_base_dir=None):
    """Run inference on a single sample using exact JSONL messages format."""
    try:
        from qwen_vl_utils import process_vision_info
        use_qwen_utils = True
    except ImportError:
        use_qwen_utils = False

    infer_msgs = make_inference_messages(sample)

    # Fix image paths: convert relative to absolute if needed
    if image_base_dir:
        for msg in infer_msgs:
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        img_path = part.get("image", "")
                        if img_path and not img_path.startswith(("/", "file://", "http")):
                            abs_path = os.path.join(image_base_dir, img_path)
                            part["image"] = f"file://{abs_path}"
                        elif img_path.startswith("/"):
                            part["image"] = f"file://{img_path}"

    # Apply chat template — match training config
    # Training used: enable_thinking=None, add_non_thinking_prefix=True
    # So we should let the model generate <think>\n\n</think>\n\n naturally
    try:
        text = processor.apply_chat_template(
            infer_msgs, tokenize=False, add_generation_prompt=True,
        )
    except Exception as e:
        logger.warning(f"apply_chat_template failed: {e}")
        raise

    if use_qwen_utils:
        image_inputs, video_inputs = process_vision_info(infer_msgs)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs if video_inputs else None,
            return_tensors="pt",
        )
    else:
        from PIL import Image as PILImage
        img_paths = extract_image_paths(infer_msgs)
        images = []
        for p in img_paths:
            p = p.replace("file://", "")
            images.append(PILImage.open(p).convert("RGB"))
        inputs = processor(text=[text], images=images if images else None, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = processor.tokenizer.decode(new_ids, skip_special_tokens=True)

    # Strip thinking tags
    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--image_base_dir", type=str, default=None,
                        help="Base dir for resolving relative image paths in JSONL")
    args = parser.parse_args()

    # Load dataset
    samples = []
    with open(args.dataset) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    logger.info(f"Loaded {len(samples)} samples from {args.dataset}")

    # Load model
    model, processor = load_model(args.model_path, args.adapter_path)

    # Resume from existing output
    done_names = set()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    done_names.add(rec.get("dataitem_name", ""))
                except json.JSONDecodeError:
                    pass
        logger.info(f"Resumed: {len(done_names)} samples already done")

    remaining = [(i, s) for i, s in enumerate(samples) if s.get("dataitem_name", f"sample_{i}") not in done_names]
    logger.info(f"Remaining: {len(remaining)} samples")

    total = len(remaining)
    t_start = time.time()

    with open(output_path, "a") as fout:
        for idx, (orig_i, sample) in enumerate(remaining):
            t0 = time.time()
            name = sample.get("dataitem_name", f"sample_{orig_i}")

            try:
                raw_text = run_single(model, processor, sample, args.max_new_tokens, args.image_base_dir)
                error = None
            except Exception as e:
                logger.error(f"[{idx+1}/{total}] ERROR {name}: {e}")
                raw_text = ""
                error = str(e)

            elapsed = time.time() - t0
            avg = (time.time() - t_start) / (idx + 1)
            eta = avg * (total - idx - 1)
            eta_str = f"{int(eta//3600)}h{int((eta%3600)//60)}m"

            # Count detections
            n_det = "err"
            if not error:
                try:
                    # Try to parse JSON from output
                    code_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw_text)
                    if code_match:
                        parsed = json.loads(code_match.group(1))
                    else:
                        parsed = json.loads(raw_text)
                    if isinstance(parsed, list):
                        n_det = len(parsed)
                except (json.JSONDecodeError, ValueError):
                    arr_match = re.search(r"\[[\s\S]*\]", raw_text)
                    if arr_match:
                        try:
                            parsed = json.loads(arr_match.group())
                            n_det = len(parsed) if isinstance(parsed, list) else "?"
                        except (json.JSONDecodeError, ValueError):
                            n_det = "parse_fail"

            logger.info(
                f"[{idx+1}/{total}] {name} | {n_det} det | "
                f"{elapsed:.1f}s | ETA {eta_str}"
            )

            record = {
                "dataitem_name": name,
                "response": raw_text,
            }
            if error:
                record["_error"] = error

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    logger.info(f"Done! Results: {args.output}")
    logger.info(f"Total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
