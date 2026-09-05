"""
RAG-augmented inference wrapper for multi-step engineering drawing analysis.

Wraps the existing inference_swift.py pipeline, injecting retrieved context
into prompts before each inference step.

Usage:
  python inference_with_rag.py \
    --adapter_path /workspace/output/swift_27b_multistep_v2/checkpoint-1800 \
    --rag_config rag_config.yaml \
    --image_dir /workspace/data/5k \
    --output_file /workspace/output/results_rag.json

  # Without RAG (fallback to standard inference)
  python inference_with_rag.py \
    --adapter_path /workspace/output/swift_27b_multistep_v2/checkpoint-1800 \
    --no_rag
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add sibling directories to path
SCRIPT_DIR = Path(__file__).parent
MULTISTEP_DIR = SCRIPT_DIR.parent / "qwen3.5-27b_5k_view_multi_step"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(MULTISTEP_DIR))


def main():
    parser = argparse.ArgumentParser(description="RAG-augmented multi-step inference")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default="/workspace/data/5k")
    parser.add_argument("--output_file", type=str,
                        default="/workspace/output/results_rag.json")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--image_max_tokens", type=int, default=2560)
    parser.add_argument("--rag_config", type=str,
                        default=str(SCRIPT_DIR / "rag_config.yaml"))
    parser.add_argument("--no_rag", action="store_true",
                        help="Disable RAG, run standard inference")
    parser.add_argument("--benchmark_jsonl", type=str, default="")
    parser.add_argument("--train_dataset", type=str,
                        default="/workspace/data/train_multistep.jsonl")
    parser.add_argument("--val_dataset", type=str,
                        default="/workspace/data/val_multistep.jsonl")
    args = parser.parse_args()

    # Import from inference_swift
    from inference_swift import (
        SYSTEM_PROMPT, STEP1_USER_PROMPT, STEP2_USER_PROMPT,
        load_model, _generate, parse_output, _deduplicate,
        postprocess_features,
    )

    # Initialize RAG
    rag = None
    if not args.no_rag:
        try:
            from rag_retriever import DrawingRAG
            rag = DrawingRAG(config_path=args.rag_config)
            logger.info("RAG enabled")
        except Exception as e:
            logger.warning(f"RAG init failed, running without RAG: {e}")

    # Determine image list
    if args.benchmark_jsonl:
        image_paths = []
        with open(args.benchmark_jsonl) as f:
            for line in f:
                rec = json.loads(line.strip())
                image_paths.append(rec["images"][0])
    else:
        image_names = set()
        for ds_path in [args.train_dataset, args.val_dataset]:
            if not ds_path or not Path(ds_path).exists():
                continue
            with open(ds_path) as f:
                for line in f:
                    rec = json.loads(line.strip())
                    for img in rec.get("images", []):
                        image_names.add(Path(img).name)

        image_dir = Path(args.image_dir)
        image_paths = sorted([
            str(image_dir / name)
            for name in image_names
            if (image_dir / name).exists()
        ])

    logger.info(f"Found {len(image_paths)} images")
    if not image_paths:
        logger.error("No images found.")
        return

    # Set resolution
    os.environ["IMAGE_MAX_TOKEN_NUM"] = str(args.image_max_tokens)
    model, processor = load_model(args.model_path, args.adapter_path, args.load_in_4bit)

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = []
    t_start = time.time()

    for i, img_path in enumerate(image_paths, 1):
        img_name = Path(img_path).name
        t0 = time.time()

        try:
            # --- Step 1: Layout & View Detection ---
            step1_prompt = STEP1_USER_PROMPT
            if rag:
                ctx1 = rag.retrieve(img_path, step=1)
                if ctx1:
                    step1_prompt = rag.augment_prompt(step1_prompt, ctx1)

            messages_step1 = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{img_path}"},
                        {"type": "text", "text": step1_prompt},
                    ],
                },
            ]

            step1_raw = _generate(model, processor, messages_step1, args.max_new_tokens)
            step1_features = parse_output(step1_raw)
            step1_features = _deduplicate(step1_features)

            # --- Step 2: Feature Detection (with RAG using step 1 results) ---
            step2_prompt = STEP2_USER_PROMPT
            if rag:
                ctx2 = rag.retrieve(img_path, step=2,
                                    model_prediction=step1_features)
                if ctx2:
                    step2_prompt = rag.augment_prompt(step2_prompt, ctx2)

            messages_step2 = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{img_path}"},
                        {"type": "text", "text": STEP1_USER_PROMPT},
                    ],
                },
                {"role": "assistant", "content": step1_raw},
                {"role": "user", "content": step2_prompt},
            ]

            step2_raw = _generate(model, processor, messages_step2, args.max_new_tokens)
            step2_features = parse_output(step2_raw)
            step2_features = _deduplicate(step2_features)
            step2_features = postprocess_features(step2_features)

        except Exception as e:
            logger.error(f"[{i}/{len(image_paths)}] ERROR on {img_name}: {e}")
            step1_features, step2_features = [], []

        # Normalize and merge
        step1_norm = [
            {"category": f.get("category", ""), "size": "", "bbox": f.get("bbox", [])}
            for f in step1_features
        ]
        step2_norm = [
            {"category": f.get("category", ""), "size": f.get("size", ""),
             "bbox": f.get("bbox", [])}
            for f in step2_features
        ]
        merged = step1_norm + step2_norm

        elapsed = time.time() - t0
        avg = (time.time() - t_start) / i
        eta = avg * (len(image_paths) - i)

        logger.info(
            f"[{i}/{len(image_paths)}] {img_name} | "
            f"views={len(step1_features)} features={len(step2_features)} "
            f"total={len(merged)} | {elapsed:.1f}s | ETA {int(eta//60)}m"
        )

        results.append({
            "dataitem_name": img_name,
            "result": merged,
            "step1_count": len(step1_features),
            "step2_count": len(step2_features),
        })

        if i % 10 == 0 or i == len(image_paths):
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Done. {len(results)} images. Saved to {output_file}")


if __name__ == "__main__":
    main()
