"""
Prepare preference data for GRPO/DPO training.

Three modes:
  1. grpo:  Generate prompts with GT for online GRPO (model samples at train time)
  2. dpo:   Generate (chosen, rejected) pairs from inference results
  3. score: Score existing inference results and rank them

Input:
  - GT annotations: 5k_15feats_with_view_v2.json
  - (For DPO) Inference results from multiple checkpoints or sampling runs

Output:
  - JSONL in ms-swift format for GRPO or DPO training

Usage:
  # GRPO mode: prepare prompts + GT for online sampling
  python prepare_preference_data.py grpo \
    --gt_json /workspace/data/5k_15feats_with_view_v2.json \
    --image_dir /workspace/data/5k \
    --output /workspace/data/rlhf_grpo.jsonl

  # DPO mode: from inference results with different quality
  python prepare_preference_data.py dpo \
    --gt_json /workspace/data/5k_15feats_with_view_v2.json \
    --results_good results_best_ckpt.json \
    --results_bad results_worst_ckpt.json \
    --output /workspace/data/rlhf_dpo.jsonl

  # Score mode: rank inference results by reward
  python prepare_preference_data.py score \
    --gt_json /workspace/data/5k_15feats_with_view_v2.json \
    --results results.json \
    --config ../RLHF/rlhf_config.yaml
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from collections import Counter

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent))
from reward import DrawingRewardFunction, parse_json_output

# Add sibling dir for dataset prompts
MULTISTEP_DIR = Path(__file__).parent.parent / "qwen3.5-27b_5k_view_multi_step"
sys.path.insert(0, str(MULTISTEP_DIR))


VIEW_CATEGORIES = {
    "Title Block", "Notes", "Revision Table", "Bill of Materials",
    "Orthographic Projection - Front View",
    "Orthographic Projection - Top View",
    "Orthographic Projection - Left View",
    "Orthographic Projection - Right View",
    "Isometric View", "Auxiliary View", "Section View",
}

FEATURE_CATEGORIES = {
    "Threaded Hole", "Threaded Hole Group",
    "Round Hole", "Round Hole Group",
    "Pin Hole", "Pin Hole Group",
    "Counterbore Hole", "Counterbore Hole Group",
    "Rectangular Hole",
    "Fillet", "Fillet Group",
    "Chamfer", "Chamfer Group",
    "Threaded Shaft",
}


def load_prompts():
    """Import prompts from the multi-step training scripts."""
    try:
        from prepare_dataset_swift import SYSTEM_PROMPT, STEP1_USER_PROMPT, STEP2_USER_PROMPT
        return SYSTEM_PROMPT, STEP1_USER_PROMPT, STEP2_USER_PROMPT
    except ImportError:
        # Fallback: read directly
        from inference_swift import SYSTEM_PROMPT, STEP1_USER_PROMPT, STEP2_USER_PROMPT
        return SYSTEM_PROMPT, STEP1_USER_PROMPT, STEP2_USER_PROMPT


def normalize_bbox(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    return [
        max(0, min(1000, int(round(x1 / img_w * 1000)))),
        max(0, min(1000, int(round(y1 / img_h * 1000)))),
        max(0, min(1000, int(round(x2 / img_w * 1000)))),
        max(0, min(1000, int(round(y2 / img_h * 1000)))),
    ]


def split_annotations(item, img_w, img_h):
    """Split annotations into view and feature lists."""
    views, features = [], []
    for task in item.get("tasks", []):
        for tv in task.get("task_values", []):
            val = tv["value"]
            label = val["label"]
            bbox = normalize_bbox(tv["bbox"], img_w, img_h)
            if label in VIEW_CATEGORIES:
                views.append({"category": label, "bbox": bbox})
            elif label in FEATURE_CATEGORIES:
                features.append({
                    "category": label,
                    "size": val.get("size", ""),
                    "bbox": bbox,
                })
    return views, features


# =====================================================================
# GRPO mode: prompts + GT for online sampling
# =====================================================================
def prepare_grpo(gt_json, image_dir, output, seed=42):
    """
    Generate GRPO training data.

    For GRPO, each sample has the prompt + ground truth response.
    During training, ms-swift samples multiple completions and uses
    the reward function to rank them.

    Output format (ms-swift GRPO):
    {"messages": [...], "images": [...]}
    Same as SFT format — ms-swift handles the sampling internally.
    """
    from PIL import Image

    SYSTEM_PROMPT, STEP1_USER_PROMPT, STEP2_USER_PROMPT = load_prompts()

    with open(gt_json) as f:
        data = json.load(f)

    random.seed(seed)
    random.shuffle(data)

    image_dir = Path(image_dir)
    available = set(f for f in os.listdir(image_dir) if f.endswith(".png"))

    count = 0
    with open(output, "w", encoding="utf-8") as out:
        for item in data:
            name = item["dataitem_name"]
            if name not in available:
                continue

            img_path = str(image_dir / name)
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except Exception:
                continue

            views, features = split_annotations(item, img_w, img_h)

            step1_answer = json.dumps(views, ensure_ascii=False, indent=2)
            step2_answer = json.dumps(features, ensure_ascii=False, indent=2)

            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"<image>{STEP1_USER_PROMPT}"},
                    {"role": "assistant", "content": step1_answer},
                    {"role": "user", "content": STEP2_USER_PROMPT},
                    {"role": "assistant", "content": step2_answer},
                ],
                "images": [img_path],
            }
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1

    print(f"GRPO data: {count} samples -> {output}")


# =====================================================================
# DPO mode: chosen/rejected pairs
# =====================================================================
def prepare_dpo(gt_json, results_good, results_bad, image_dir, output,
                config_path=None, seed=42):
    """
    Generate DPO preference pairs.

    Compares predictions from two sources (e.g., best vs worst checkpoint)
    and creates (chosen, rejected) pairs based on reward scores.

    Output format (ms-swift DPO):
    {
      "messages": [system, user_step1],
      "chosen": "...",
      "rejected": "...",
      "images": [...]
    }
    """
    from PIL import Image

    SYSTEM_PROMPT, STEP1_USER_PROMPT, STEP2_USER_PROMPT = load_prompts()

    reward_fn = DrawingRewardFunction(config_path=config_path)

    with open(gt_json) as f:
        gt_data = json.load(f)
    gt_by_name = {item["dataitem_name"]: item for item in gt_data}

    with open(results_good) as f:
        good_results = json.load(f)
    with open(results_bad) as f:
        bad_results = json.load(f)

    good_by_name = {r["dataitem_name"]: r for r in good_results}
    bad_by_name = {r["dataitem_name"]: r for r in bad_results}

    image_dir = Path(image_dir)
    common_names = set(good_by_name) & set(bad_by_name) & set(gt_by_name)

    random.seed(seed)
    count = 0

    with open(output, "w", encoding="utf-8") as out:
        for name in sorted(common_names):
            gt_item = gt_by_name[name]
            img_path = str(image_dir / name)

            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except Exception:
                continue

            views, features = split_annotations(gt_item, img_w, img_h)
            gt_json_str = json.dumps(views + features, ensure_ascii=False)

            good_json = json.dumps(good_by_name[name]["result"], ensure_ascii=False)
            bad_json = json.dumps(bad_by_name[name]["result"], ensure_ascii=False)

            score_good = reward_fn(good_json, gt_json_str)
            score_bad = reward_fn(bad_json, gt_json_str)

            # Only create pair if there's a meaningful quality difference
            if abs(score_good - score_bad) < 0.05:
                continue

            chosen = good_json if score_good >= score_bad else bad_json
            rejected = bad_json if score_good >= score_bad else good_json

            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"<image>{STEP1_USER_PROMPT}"},
                ],
                "chosen": chosen,
                "rejected": rejected,
                "images": [img_path],
            }
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1

    print(f"DPO data: {count} preference pairs -> {output}")
    print(f"  (from {len(common_names)} common images, "
          f"filtered pairs with <0.05 score diff)")


# =====================================================================
# Score mode: rank existing results
# =====================================================================
def score_results(gt_json, results_path, config_path=None):
    """Score inference results and print per-image and aggregate stats."""
    reward_fn = DrawingRewardFunction(config_path=config_path)

    with open(gt_json) as f:
        gt_data = json.load(f)
    gt_by_name = {item["dataitem_name"]: item for item in gt_data}

    with open(results_path) as f:
        results = json.load(f)

    scores = []
    cat_scores = Counter()
    cat_counts = Counter()

    for entry in results:
        name = entry["dataitem_name"]
        if name not in gt_by_name:
            continue

        gt_item = gt_by_name[name]
        gt_features = []
        for task in gt_item.get("tasks", []):
            for tv in task.get("task_values", []):
                val = tv["value"]
                gt_features.append({
                    "category": val["label"],
                    "size": val.get("size", ""),
                    "bbox": tv["bbox"],
                })

        pred_json = json.dumps(entry["result"], ensure_ascii=False)
        gt_json_str = json.dumps(gt_features, ensure_ascii=False)

        detail = reward_fn.score_detailed(pred_json, gt_json_str)
        detail["image"] = name
        scores.append(detail)

        # Track per-category
        for feat in gt_features:
            cat = feat["category"]
            cat_counts[cat] += 1

    scores.sort(key=lambda x: x["total"])

    print(f"\n=== Reward Scores ({len(scores)} images) ===\n")

    # Aggregate
    for key in ["total", "category_f1", "bbox_iou", "size_accuracy", "completeness"]:
        vals = [s[key] for s in scores]
        avg = sum(vals) / len(vals)
        p10 = vals[int(len(vals) * 0.1)]
        p50 = vals[int(len(vals) * 0.5)]
        p90 = vals[int(len(vals) * 0.9)]
        print(f"  {key:20s}  avg={avg:.4f}  p10={p10:.4f}  p50={p50:.4f}  p90={p90:.4f}")

    # Bottom 10
    print(f"\n--- Bottom 10 (lowest reward) ---")
    for s in scores[:10]:
        print(f"  {s['image']:40s}  total={s['total']:.4f}  "
              f"cat_f1={s['category_f1']:.3f}  bbox={s['bbox_iou']:.3f}  "
              f"size={s['size_accuracy']:.3f}")


# =====================================================================
# CLI
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Prepare RLHF preference data")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # GRPO
    p_grpo = subparsers.add_parser("grpo", help="Prepare GRPO training data")
    p_grpo.add_argument("--gt_json", required=True)
    p_grpo.add_argument("--image_dir", default="/workspace/data/5k")
    p_grpo.add_argument("--output", default="/workspace/data/rlhf_grpo.jsonl")
    p_grpo.add_argument("--seed", type=int, default=42)

    # DPO
    p_dpo = subparsers.add_parser("dpo", help="Prepare DPO preference pairs")
    p_dpo.add_argument("--gt_json", required=True)
    p_dpo.add_argument("--results_good", required=True, help="Better model results")
    p_dpo.add_argument("--results_bad", required=True, help="Worse model results")
    p_dpo.add_argument("--image_dir", default="/workspace/data/5k")
    p_dpo.add_argument("--output", default="/workspace/data/rlhf_dpo.jsonl")
    p_dpo.add_argument("--config", default=str(Path(__file__).parent / "rlhf_config.yaml"))
    p_dpo.add_argument("--seed", type=int, default=42)

    # Score
    p_score = subparsers.add_parser("score", help="Score inference results")
    p_score.add_argument("--gt_json", required=True)
    p_score.add_argument("--results", required=True)
    p_score.add_argument("--config", default=str(Path(__file__).parent / "rlhf_config.yaml"))

    args = parser.parse_args()

    if args.mode == "grpo":
        prepare_grpo(args.gt_json, args.image_dir, args.output, args.seed)
    elif args.mode == "dpo":
        prepare_dpo(args.gt_json, args.results_good, args.results_bad,
                    args.image_dir, args.output, args.config, args.seed)
    elif args.mode == "score":
        score_results(args.gt_json, args.results, args.config)


if __name__ == "__main__":
    main()
