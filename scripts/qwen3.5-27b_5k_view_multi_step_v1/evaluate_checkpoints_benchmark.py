#!/usr/bin/env python3
"""
Evaluate a multi-step checkpoint on the benchmark dataset.
Computes P/R/F1 with IoU matching, per-category stats, size accuracy.
Reports metrics for Step 1 (views), Step 2 (features), and combined.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

from inference_swift import (
    load_model, run_multistep_inference, parse_output, _deduplicate,
    VIEW_CATEGORIES, FEATURE_CATEGORIES,
)


VIEW_CATEGORIES_SET = {
    "Title Block", "Notes", "Revision Table", "Bill of Materials",
    "Orthographic Projection - Front View",
    "Orthographic Projection - Top View",
    "Orthographic Projection - Left View",
    "Orthographic Projection - Right View",
    "Isometric View", "Auxiliary View", "Section View",
}

FEATURE_CATEGORIES_SET = {
    "Threaded Hole", "Threaded Hole Group",
    "Round Hole", "Round Hole Group",
    "Pin Hole", "Pin Hole Group",
    "Counterbore Hole", "Counterbore Hole Group",
    "Rectangular Hole",
    "Fillet", "Fillet Group",
    "Chamfer", "Chamfer Group",
    "Threaded Shaft",
}


def normalize_box(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        vals = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    mx = max(abs(v) for v in vals)
    if mx <= 1.5:
        vals = [v * 1000 for v in vals]
    vals = [max(0, min(1000, int(round(v)))) for v in vals]
    if vals[0] >= vals[2] or vals[1] >= vals[3]:
        return None
    return vals


def calculate_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter) if (aa + ab - inter) > 0 else 0.0


def evaluate_items(pred_items, gt_items, iou_threshold=0.4):
    """Evaluate predicted items against GT items."""
    stats = {
        "tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0,
        "size_correct": 0, "size_total": 0,
        "per_cat": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
    }

    matched_gt = set()
    for pred in pred_items:
        best_iou, best_idx, best_gt = -1, -1, None
        for i, gt in enumerate(gt_items):
            if i in matched_gt:
                continue
            iou = calculate_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou, best_idx, best_gt = iou, i, gt

        if best_iou >= iou_threshold and best_idx >= 0:
            matched_gt.add(best_idx)
            if pred["category"] == best_gt["category"]:
                stats["tp"] += 1
                stats["per_cat"][pred["category"]]["tp"] += 1
                if pred.get("size") is not None and best_gt.get("size") is not None:
                    stats["size_total"] += 1
                    if pred["size"] == best_gt["size"]:
                        stats["size_correct"] += 1
            else:
                stats["fp"] += 1
                stats["fn"] += 1
                stats["per_cat"][pred["category"]]["fp"] += 1
                stats["per_cat"][best_gt["category"]]["fn"] += 1
            stats["iou_sum"] += best_iou
        else:
            stats["fp"] += 1
            stats["per_cat"][pred["category"]]["fp"] += 1

    for i, gt in enumerate(gt_items):
        if i not in matched_gt:
            stats["fn"] += 1
            stats["per_cat"][gt["category"]]["fn"] += 1

    return stats


def extract_items(features, category_set=None):
    """Extract and normalize items, optionally filtering by category set."""
    items = []
    for f in features:
        cat = f.get("category", "")
        bbox = normalize_box(f.get("bbox_2d", f.get("bbox", [])))
        size = f.get("size", "")
        if cat and bbox:
            if category_set is None or cat in category_set:
                items.append({"category": cat, "bbox": bbox, "size": size})
    return items


def compute_metrics(stats):
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    size_acc = stats["size_correct"] / stats["size_total"] if stats["size_total"] > 0 else 0

    cat_f1_values = []
    per_category = {}
    for cat in sorted(stats["per_cat"].keys()):
        s = stats["per_cat"][cat]
        cp = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0
        cr = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0
        cf = 2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0
        per_category[cat] = {"precision": cp, "recall": cr, "f1": cf,
                             "tp": s["tp"], "fp": s["fp"], "fn": s["fn"]}
        if (s["tp"] + s["fn"]) > 0:
            cat_f1_values.append(cf)

    macro_f1 = sum(cat_f1_values) / len(cat_f1_values) if cat_f1_values else 0

    return {
        "precision": precision, "recall": recall,
        "f1_micro": f1, "f1_macro": macro_f1,
        "tp": tp, "fp": fp, "fn": fn,
        "size_accuracy": size_acc,
        "per_category": per_category,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--benchmark_jsonl", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--base_model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--iou_threshold", type=float, default=0.4)
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    # Load benchmark
    records = []
    with open(args.benchmark_jsonl) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    print(f"Loaded {len(records)} benchmark samples")

    model, processor = load_model(args.base_model, args.adapter_path, args.load_in_4bit)

    # Accumulators for step1, step2, and combined
    total_step1 = {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0,
                   "size_correct": 0, "size_total": 0,
                   "per_cat": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})}
    total_step2 = {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0,
                   "size_correct": 0, "size_total": 0,
                   "per_cat": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})}
    total_combined = {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0,
                      "size_correct": 0, "size_total": 0,
                      "per_cat": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})}
    sample_results = []

    for i, rec in enumerate(records, 1):
        img_path = rec["images"][0]
        gt_text = rec["messages"][-1]["content"]

        # Parse GT — need to combine both turns
        # In multi-step format, messages[2] = step1 GT, messages[4] = step2 GT
        gt_step1_text = rec["messages"][2]["content"]
        gt_step2_text = rec["messages"][4]["content"]

        try:
            gt_step1 = json.loads(gt_step1_text)
        except:
            gt_step1 = []
        try:
            gt_step2 = json.loads(gt_step2_text)
        except:
            gt_step2 = []

        gt_all = gt_step1 + gt_step2

        print(f"[{i}/{len(records)}] {Path(img_path).name}...", end=" ", flush=True)

        try:
            step1_features, step2_features, _, _ = run_multistep_inference(
                model, processor, img_path, args.max_new_tokens
            )
        except Exception as e:
            print(f"ERROR: {e}")
            step1_features, step2_features = [], []

        # Evaluate Step 1 (views)
        pred_view = extract_items(step1_features, VIEW_CATEGORIES_SET)
        gt_view = extract_items(gt_step1, VIEW_CATEGORIES_SET)
        stats1 = evaluate_items(pred_view, gt_view, args.iou_threshold)

        # Evaluate Step 2 (features)
        pred_feat = extract_items(step2_features, FEATURE_CATEGORIES_SET)
        gt_feat = extract_items(gt_step2, FEATURE_CATEGORIES_SET)
        stats2 = evaluate_items(pred_feat, gt_feat, args.iou_threshold)

        # Evaluate combined
        pred_all = extract_items(step1_features + step2_features)
        gt_all_items = extract_items(gt_all)
        stats_combined = evaluate_items(pred_all, gt_all_items, args.iou_threshold)

        # Accumulate
        for total, stats in [(total_step1, stats1), (total_step2, stats2), (total_combined, stats_combined)]:
            for k in ["tp", "fp", "fn", "iou_sum", "size_correct", "size_total"]:
                total[k] += stats[k]
            for cat, vals in stats["per_cat"].items():
                for k in ["tp", "fp", "fn"]:
                    total["per_cat"][cat][k] += vals[k]

        # Per-sample combined F1
        tp, fp, fn = stats_combined["tp"], stats_combined["fp"], stats_combined["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"views={len(step1_features)} feats={len(step2_features)} "
              f"P={p:.3f} R={r:.3f} F1={f1:.3f}")

        sample_results.append({
            "image": Path(img_path).name,
            "step1_count": len(step1_features),
            "step2_count": len(step2_features),
            "combined_tp": tp, "combined_fp": fp, "combined_fn": fn,
        })

    # Compute final metrics
    result = {
        "step1_views": compute_metrics(total_step1),
        "step2_features": compute_metrics(total_step2),
        "combined": compute_metrics(total_combined),
        "samples": sample_results,
    }

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Print summary
    for name, key in [("Step 1 (Views)", "step1_views"),
                      ("Step 2 (Features)", "step2_features"),
                      ("Combined", "combined")]:
        m = result[key]
        print(f"\n{'='*50}")
        print(f"{name}: P={m['precision']:.4f} R={m['recall']:.4f} "
              f"F1_micro={m['f1_micro']:.4f} F1_macro={m['f1_macro']:.4f}")
        print(f"  TP={m['tp']} FP={m['fp']} FN={m['fn']} SizeAcc={m['size_accuracy']:.4f}")
        for cat in sorted(m["per_category"].keys()):
            c = m["per_category"][cat]
            print(f"  {cat}: F1={c['f1']:.3f} (TP={c['tp']} FP={c['fp']} FN={c['fn']})")

    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
