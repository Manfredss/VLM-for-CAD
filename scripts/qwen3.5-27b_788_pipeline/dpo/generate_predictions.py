"""
generate_predictions.py — Step 1 of the DPO pipeline.

For each TRAIN sample, generate K predictions with temperature sampling.
Output: JSONL with one line per (image, sample_idx), containing both raw turns.

We sample K times because DPO needs (chosen, rejected) pairs from the same
prompt. With temperature > 0 the SFT model produces a spread of qualities,
which we then score (step 2) and pair (step 3).
"""
import argparse
import json
import os
import re
import time
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _resolve_device_map():
    v = os.environ.get("INFERENCE_DEVICE_MAP", "auto").strip()
    if not v or v.lower() == "auto":
        return "auto"
    if v.lower() in ("none", "manual"):
        return None
    if v.lower() == "cpu" or v.lower().startswith("cuda:"):
        return {"": v}
    return v


def load_model(model_path, adapter_path):
    from transformers import AutoProcessor
    from peft import PeftModel
    try:
        from transformers import AutoModelForImageTextToText as _Cls
    except ImportError:
        from transformers import AutoModelForCausalLM as _Cls

    logger.info(f"Loading processor: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    logger.info(f"Loading base: {model_path}")
    os.environ.setdefault("INFERENCE_MODEL_PATH", model_path)
    kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True,
                  torch_dtype=torch.bfloat16)
    dm = _resolve_device_map()
    if dm is not None:
        kwargs["device_map"] = dm
    try:
        import flash_attn  # noqa
        kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        kwargs["attn_implementation"] = "sdpa"
    model = _Cls.from_pretrained(model_path, **kwargs)

    if adapter_path:
        logger.info(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    if dm is None:
        model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, processor


def _gen(model, processor, messages, max_new_tokens, temperature, top_p):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs,
                           videos=video_inputs if video_inputs else None,
                           return_tensors="pt")
    except ImportError:
        from PIL import Image
        paths = [c.get("image", "").replace("file://", "")
                 for m in messages if isinstance(m.get("content"), list)
                 for c in m["content"] if c.get("type") == "image"]
        imgs = [Image.open(p).convert("RGB") for p in paths]
        inputs = processor(text=[text], images=imgs if imgs else None,
                           return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True if temperature > 0 else False,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            repetition_penalty=1.05,
        )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_ids, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model_path", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N train samples (0 = all)")
    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args.model_path, args.adapter)

    samples = []
    with open(args.train_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[:args.limit]
    logger.info(f"Sampling {args.k}× over {len(samples)} train items")

    f_out = open(out_path, "a")  # append-mode for resumability
    done_keys = set()
    if out_path.stat().st_size > 0:
        with open(out_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_keys.add((rec["image"], rec["sample_idx"]))
                except: pass
        logger.info(f"resumed: {len(done_keys)} (image, sample) already done")

    t0 = time.time()
    for i, rec in enumerate(samples):
        image = Path(rec["images"][0]).name
        local_path = str(image_dir / image)
        # Reconstruct messages: same system + step1 user, then we sample step1 assistant
        # Then to sample step2 we feed our sampled step1 + step2 user.
        msgs_base = rec["messages"]  # [system, user1+image, asst1, user2, asst2]
        sys_msg = msgs_base[0]
        user1 = {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{local_path}"},
                {"type": "text", "text": msgs_base[1]["content"].replace("<image>", "")},
            ],
        }
        user2 = msgs_base[3]

        for k in range(args.k):
            if (image, k) in done_keys:
                continue
            try:
                # Turn 1
                msgs1 = [sys_msg, user1]
                raw_v = _gen(model, processor, msgs1,
                             args.max_new_tokens, args.temperature, args.top_p)
                # Turn 2 (continues with our sampled turn-1 response)
                msgs2 = msgs1 + [{"role": "assistant", "content": raw_v}, user2]
                raw_f = _gen(model, processor, msgs2,
                             args.max_new_tokens, args.temperature, args.top_p)
            except Exception as e:
                logger.error(f"[{i+1}/{len(samples)} k={k}] {image}: {e}")
                raw_v = ""; raw_f = ""

            out_rec = {
                "image": image,
                "sample_idx": k,
                "raw_views": raw_v,
                "raw_features": raw_f,
            }
            f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            f_out.flush()
            elapsed = time.time() - t0
            tot_done = sum(1 for _ in range(i+1)) * args.k  # rough
            avg = elapsed / max(1, (i*args.k + k + 1))
            eta = avg * (len(samples)*args.k - (i*args.k + k + 1))
            logger.info(f"[{i+1}/{len(samples)} k={k}] {image} | {len(raw_v)+len(raw_f)} chars"
                        f" | {avg:.1f}s/sample | ETA {int(eta/3600)}h{int(eta%3600/60)}m")

    f_out.close()
    logger.info(f"done -> {out_path}")


if __name__ == "__main__":
    main()
