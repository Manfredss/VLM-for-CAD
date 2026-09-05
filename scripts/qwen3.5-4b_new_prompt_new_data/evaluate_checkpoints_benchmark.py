#!/usr/bin/env python3
"""
Evaluate a checkpoint on the benchmark dataset.
Computes P/R/F1 with IoU matching, per-category stats, size accuracy.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

from inference_swift import load_model, run_inference_single, parse_output, _deduplicate


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


def evaluate_sample(pred_features, gt_features, iou_threshold=0.4):
    pred_items = []
    for f in pred_features:
        cat = f.get("category", "")
        bbox = normalize_box(f.get("bbox_2d", f.get("bbox", [])))
        size = f.get("size", "")
        if cat and bbox:
            pred_items.append({"category": cat, "bbox": bbox, "size": size})

    gt_items = []
    for f in gt_features:
        cat = f.get("category", "")
        bbox = normalize_box(f.get("bbox_2d", f.get("bbox", [])))
        size = f.get("size", "")
        if cat and bbox:
            gt_items.append({"category": cat, "bbox": bbox, "size": size})

    stats = {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0,
             "size_correct": 0, "size_total": 0,
             "per_cat": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})}

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--benchmark_jsonl", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--base_model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--iou_threshold", type=float, default=0.4)
    args = parser.parse_args()

    # Load benchmark data
    records = []
    with open(args.benchmark_jsonl) as f:
        for line in f:
            rec = json.loads(line.strip())
            records.append(rec)
    print(f"Loaded {len(records)} benchmark samples")

    # Load model
    model, processor = load_model(args.adapter_path, args.base_model)

    # Run evaluation
    total = {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0,
             "size_correct": 0, "size_total": 0,
             "per_cat": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})}
    sample_results = []

    for i, rec in enumerate(records, 1):
        msgs = rec["messages"]
        img_path = rec["images"][0]

        # Get ground truth from assistant message
        gt_text = msgs[-1]["content"]
        try:
            gt_features = json.loads(gt_text)
        except:
            gt_features = []

        print(f"[{i}/{len(records)}] {Path(img_path).name}...", end=" ", flush=True)
        try:
            raw = run_inference_single(model, processor, img_path, args.max_new_tokens)
            pred_features = parse_output(raw)
            pred_features = _deduplicate(pred_features)
        except Exception as e:
            print(f"ERROR: {e}")
            pred_features = []
            raw = ""

        stats = evaluate_sample(pred_features, gt_features, args.iou_threshold)

        for k in ["tp", "fp", "fn", "iou_sum", "size_correct", "size_total"]:
            total[k] += stats[k]
        for cat, vals in stats["per_cat"].items():
            for k in ["tp", "fp", "fn"]:
                total["per_cat"][cat][k] += vals[k]

        p = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0
        r = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
        print(f"P={p:.3f} R={r:.3f} F1={f1:.3f} (TP={stats['tp']} FP={stats['fp']} FN={stats['fn']})")

        sample_results.append({
            "image": Path(img_path).name,
            "tp": stats["tp"], "fp": stats["fp"], "fn": stats["fn"],
            "pred_count": len(pred_features), "gt_count": len(gt_features),
        })

    # Overall metrics
    tp, fp, fn = total["tp"], total["fp"], total["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    size_acc = total["size_correct"] / total["size_total"] if total["size_total"] > 0 else 0

    # Per-category metrics and macro-averaged F1
    per_category = {}
    cat_f1_values = []
    for cat in sorted(total["per_cat"].keys()):
        s = total["per_cat"][cat]
        cp = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0
        cr = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0
        cf = 2*cp*cr/(cp+cr) if (cp+cr) > 0 else 0
        per_category[cat] = {"precision": cp, "recall": cr, "f1": cf,
                             "tp": s["tp"], "fp": s["fp"], "fn": s["fn"]}
        # Only include categories present in GT for macro-average
        if (s["tp"] + s["fn"]) > 0:
            cat_f1_values.append(cf)

    macro_f1 = sum(cat_f1_values) / len(cat_f1_values) if cat_f1_values else 0.0

    result = {
        "overall": {
            "precision": precision, "recall": recall,
            "f1_micro": f1, "f1_macro": macro_f1,
            "tp": tp, "fp": fp, "fn": fn,
            "size_accuracy": size_acc,
        },
        "per_category": per_category,
        "samples": sample_results,
    }

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Overall: P={precision:.4f} R={recall:.4f} F1_micro={f1:.4f} F1_macro={macro_f1:.4f}")
    print(f"  TP={tp} FP={fp} FN={fn} SizeAcc={size_acc:.4f}")
    for cat in sorted(per_category.keys()):
        c = per_category[cat]
        print(f"  {cat}: F1={c['f1']:.3f} (TP={c['tp']} FP={c['fp']} FN={c['fn']})")
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
