"""Step 2 of the DPO pipeline: score sampled predictions.

Uses the shared CAD-aware metric instead of the old class-agnostic IoU quality:
layout samples are ranked by class-aware layout F1 + IoU, while feature samples
also include normalized dimension similarity.
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.append(str(PIPELINE_DIR))

from cad_metrics import (  # noqa: E402
    DEFAULT_FEATURE_IOU,
    DEFAULT_LAYOUT_IOU,
    FEATURE_CATEGORIES,
    VIEW_CATEGORIES,
    extract_items,
    parse_json_arrays,
    quality_score,
)


def load_ground_truth(train_jsonl: Path) -> dict:
    gt_by_image = {}
    with open(train_jsonl, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_name = Path(record["images"][0]).name
            assistants = [
                msg for msg in record.get("messages", [])
                if msg.get("role") == "assistant"
            ]
            if len(assistants) < 2:
                continue
            gt_by_image[image_name] = (
                extract_items(parse_json_arrays(assistants[0].get("content", ""))),
                extract_items(parse_json_arrays(assistants[1].get("content", ""))),
            )
    return gt_by_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, help="from generate_predictions.py")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layout_iou", type=float, default=DEFAULT_LAYOUT_IOU)
    parser.add_argument("--feature_iou", type=float, default=DEFAULT_FEATURE_IOU)
    args = parser.parse_args()

    gt_by_image = load_ground_truth(Path(args.train_jsonl))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.samples, encoding="utf-8") as samples, open(out_path, "w", encoding="utf-8") as out:
        for line in samples:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_name = record["image"]
            if image_name not in gt_by_image:
                continue

            gt_views, gt_features = gt_by_image[image_name]
            pred_views = extract_items(parse_json_arrays(record.get("raw_views", "")))
            pred_features = extract_items(parse_json_arrays(record.get("raw_features", "")))

            record["quality_views"] = quality_score(
                pred_views,
                gt_views,
                VIEW_CATEGORIES,
                args.layout_iou,
                include_size=False,
            )
            record["quality_features"] = quality_score(
                pred_features,
                gt_features,
                FEATURE_CATEGORIES,
                args.feature_iou,
                include_size=True,
            )
            record["quality_total"] = (
                0.35 * record["quality_views"] + 0.65 * record["quality_features"]
            )
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"scored -> {out_path}")


if __name__ == "__main__":
    main()
