"""Post-hoc scorer for two-turn CAD drawing extraction results.

Scores `result_views` and `result_features` against a test JSONL using the
shared CAD-aware metric:
  - class-aware matching, so group boxes do not steal child-hole matches
  - separate layout and feature metrics
  - normalized CAD size accuracy and partial dimension similarity
  - strict feature F1 for tighter localization visibility
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from cad_metrics import (  # noqa: E402
    DEFAULT_FEATURE_IOU,
    DEFAULT_LAYOUT_IOU,
    DEFAULT_STRICT_FEATURE_IOU,
    FEATURE_CATEGORIES,
    VIEW_CATEGORIES,
    add_stats,
    compute_split_metrics_from_items,
    empty_stats,
    extract_items,
    flatten_metrics,
    merge_stats,
    parse_json_arrays,
    prf,
)


def load_ground_truth(test_jsonl: Path) -> list:
    records = []
    with open(test_jsonl, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            images = data.get("images", [])
            if not images:
                continue
            assistants = [
                msg for msg in data.get("messages", [])
                if msg.get("role") == "assistant"
            ]
            if len(assistants) < 2:
                continue
            records.append({
                "name": Path(images[0]).name,
                "gt_views": extract_items(parse_json_arrays(assistants[0].get("content", ""))),
                "gt_features": extract_items(parse_json_arrays(assistants[1].get("content", ""))),
            })
    return records


def prediction_items(record: dict) -> tuple:
    pred_views = extract_items(record.get("result_views", []))
    pred_features = extract_items(record.get("result_features", []))

    if not pred_views and not pred_features and isinstance(record.get("result"), list):
        combined = extract_items(record["result"])
        pred_views = [item for item in combined if item["category"] in VIEW_CATEGORIES]
        pred_features = [item for item in combined if item["category"] in FEATURE_CATEGORIES]
    return pred_views, pred_features


def print_category_table(title: str, categories, stats: dict):
    print()
    print(title)
    for category in sorted(categories):
        cat_stats = stats["per_cat"].get(category, {"tp": 0, "fp": 0, "fn": 0})
        _, _, f1 = prf(cat_stats["tp"], cat_stats["fp"], cat_stats["fn"])
        print(
            f"  {category:46s} "
            f"tp={cat_stats['tp']:4d} fp={cat_stats['fp']:4d} "
            f"fn={cat_stats['fn']:4d} F1={f1:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--test_jsonl", required=True, type=Path)
    parser.add_argument("--layout_iou", type=float, default=DEFAULT_LAYOUT_IOU)
    parser.add_argument("--feature_iou", type=float, default=DEFAULT_FEATURE_IOU)
    parser.add_argument("--strict_feature_iou", type=float, default=DEFAULT_STRICT_FEATURE_IOU)
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as handle:
        results = json.load(handle)
    by_name = {record["dataitem_name"]: record for record in results}
    print(f"Loaded {len(results)} prediction records")

    ground_truth = load_ground_truth(args.test_jsonl)
    print(f"Loaded {len(ground_truth)} GT records")

    view_total = empty_stats()
    feature_total = empty_stats()
    strict_feature_total = empty_stats()

    matched, unmatched = 0, 0
    for gt in ground_truth:
        record = by_name.get(gt["name"])
        if record is None:
            unmatched += 1
            continue
        matched += 1
        pred_views, pred_features = prediction_items(record)
        _, view_stats, feature_stats, _, strict_feature_stats = compute_split_metrics_from_items(
            pred_views,
            pred_features,
            gt["gt_views"],
            gt["gt_features"],
            layout_iou=args.layout_iou,
            feature_iou=args.feature_iou,
            strict_feature_iou=args.strict_feature_iou,
        )
        merge_stats(view_total, view_stats)
        merge_stats(feature_total, feature_stats)
        merge_stats(strict_feature_total, strict_feature_stats)

    combined_total = add_stats(view_total, feature_total)
    metrics = flatten_metrics(view_total, feature_total, combined_total, strict_feature_total)

    print(f"Matched {matched} preds with GT, {unmatched} GT not in preds")
    print()
    print("=== Overall ===")
    print(
        "Layout   | "
        f"P={metrics['layout_precision']:.4f} R={metrics['layout_recall']:.4f} "
        f"F1={metrics['layout_f1']:.4f} mIoU={metrics['layout_bbox_iou_mean']:.4f}"
    )
    print(
        "Features | "
        f"P={metrics['feature_precision']:.4f} R={metrics['feature_recall']:.4f} "
        f"F1={metrics['feature_f1']:.4f} strictF1={metrics['feature_f1_strict']:.4f} "
        f"mIoU={metrics['feature_bbox_iou_mean']:.4f} "
        f"size_acc={metrics['size_accuracy']:.4f} size_sim={metrics['size_similarity']:.4f}"
    )
    print(
        "Combined | "
        f"P={metrics['detection_precision']:.4f} R={metrics['detection_recall']:.4f} "
        f"F1={metrics['detection_f1']:.4f} CADScore={metrics['cad_extraction_score']:.4f}"
    )

    print_category_table("=== Per-class layout ===", VIEW_CATEGORIES, view_total)
    print_category_table("=== Per-class features ===", FEATURE_CATEGORIES, feature_total)


if __name__ == "__main__":
    main()
