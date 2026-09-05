"""
IMMetric — eval metric for the multi-turn views + 7 features task.

Adapted from scripts/qwn3.5-27b_788_silver_plate_bend/metric.py:
  - ALL_CATEGORIES expanded to 13 view + 7 feature = 20.
  - View categories carry no `size` field; size accuracy is computed only
    over feature categories.
  - Otherwise: same IoU>=threshold + category-match logic.

Compatible with ms-swift 4.1.x via the swift.metrics.eval_metrics_map plugin point.
"""

import os
import re
import json
import logging
import numpy as np
import torch
from typing import Dict, List
from collections import defaultdict
from transformers import EvalPrediction, AutoTokenizer

try:
    from swift.trainers.mixin import eval_metrics_map
    _HAS_EVAL_METRICS_MAP = True
except ImportError:
    _HAS_EVAL_METRICS_MAP = False

try:
    from swift.plugin import metric_mapping
    _HAS_SWIFT_PLUGIN = True
except ImportError:
    _HAS_SWIFT_PLUGIN = False

logger = logging.getLogger(__name__)

_metric_model_path = os.environ.get('METRIC_MODEL', 'Qwen/Qwen3.5-27B')
_tokenizer = None

IOU_THRESHOLD = float(os.environ.get('METRIC_IOU_THRESHOLD', '0.4'))

VIEW_CATEGORIES = {
    "Title Block",
    "Notes",
    "Orthographic Projection - Front View",
    "Orthographic Projection - Top View",
    "Orthographic Projection - Left View",
    "Orthographic Projection - Right View",
    "Orthographic Projection - Bottom View",
    "Orthographic Projection - Rear View",
    "Isometric View",
    "Flat Pattern View",
    "Section View",
    "Detail View",
    "Auxiliary View",
}

FEATURE_CATEGORIES = {
    "Round Hole",
    "Rectangular Hole",
    "Threaded Hole",
    "Slotted Hole",
    "Round Hole Group",
    "Rectangular Hole Group",
    "Slotted Hole Group",
    "Fillet",
    "Bending",
    "Silver Plating",
}

ALL_CATEGORIES = sorted(VIEW_CATEGORIES) + sorted(FEATURE_CATEGORIES)


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(
                _metric_model_path, trust_remote_code=True
            )
            logger.info(f"Metric tokenizer loaded from {_metric_model_path}")
        except Exception as e:
            logger.warning(f'Failed to load metric tokenizer: {e}')
    return _tokenizer


def calculate_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def normalize_box(box) -> list:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return []
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return []
    max_coord = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if max_coord <= 1.5:
        x1, y1, x2, y2 = [v * 1000.0 for v in (x1, y1, x2, y2)]
    normalized = [int(round(v)) for v in (x1, y1, x2, y2)]
    normalized = [max(0, min(1000, v)) for v in normalized]
    if normalized[0] >= normalized[2] or normalized[1] >= normalized[3]:
        return []
    return normalized


def parse_features_from_text(text: str) -> list:
    """Parse a list of features from raw text. Supports multiple JSON arrays
    in the same string (multi-turn output) by concatenating them."""
    text = text.strip()

    items = []

    # Try to parse the whole thing first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Find all ```json ...``` blocks
    pattern = re.compile(r'```json\s*([\s\S]*?)```', re.MULTILINE)
    for m in pattern.finditer(text):
        try:
            parsed = json.loads(m.group(1).strip())
            if isinstance(parsed, list):
                items.extend(parsed)
        except json.JSONDecodeError:
            continue
    if items:
        return items

    # Find all top-level [ ... ] arrays (greedy multi-array support)
    bracket_pattern = re.compile(r'\[\s*(?:\{[\s\S]*?\}\s*,?\s*)*\]', re.MULTILINE)
    for m in bracket_pattern.finditer(text):
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                items.extend(parsed)
        except json.JSONDecodeError:
            continue

    return items


def extract_items(features: list) -> list:
    items = []
    for f in features:
        cat = f.get("category", f.get("label", ""))
        bbox = normalize_box(f.get("bbox_2d", f.get("bbox", [])))
        size = f.get("size", "")
        if cat and bbox:
            items.append({"category": cat, "bbox": bbox, "size": size})
    return items


def compute_detection_metrics(pred_text: str, label_text: str, iou_threshold: float = IOU_THRESHOLD) -> dict:
    pred_features = parse_features_from_text(pred_text)
    label_features = parse_features_from_text(label_text)

    pred_items = extract_items(pred_features)
    gt_items = extract_items(label_features)

    stats = {
        "tp": 0, "fp": 0, "fn": 0,
        "iou_sum": 0.0, "match_count": 0,
        "size_correct": 0, "size_total": 0,
        "quality_sum": 0.0,
        "per_cat": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
    }

    if not gt_items and not pred_items:
        return stats

    matched_gt = set()

    for pred in pred_items:
        best_iou = -1.0
        best_idx = -1
        best_gt = None

        for i, gt in enumerate(gt_items):
            if i in matched_gt:
                continue
            iou = calculate_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = i
                best_gt = gt

        if best_iou >= iou_threshold and best_idx >= 0:
            matched_gt.add(best_idx)
            cat_match = pred["category"] == best_gt["category"]

            stats["iou_sum"] += best_iou
            stats["match_count"] += 1

            iou_reward = 1.0 if best_iou >= 0.8 else best_iou
            label_score = 1.0 if cat_match else 0.0
            stats["quality_sum"] += 0.5 * iou_reward + 0.5 * label_score

            if cat_match:
                stats["tp"] += 1
                stats["per_cat"][pred["category"]]["tp"] += 1
                # Size scoring only applies to feature categories
                if pred["category"] in FEATURE_CATEGORIES:
                    stats["size_total"] += 1
                    if pred["size"] == best_gt["size"]:
                        stats["size_correct"] += 1
            else:
                stats["fp"] += 1
                stats["fn"] += 1
                stats["per_cat"][pred["category"]]["fp"] += 1
                stats["per_cat"][best_gt["category"]]["fn"] += 1
        else:
            stats["fp"] += 1
            stats["per_cat"][pred["category"]]["fp"] += 1

    for i, gt in enumerate(gt_items):
        if i not in matched_gt:
            stats["fn"] += 1
            stats["per_cat"][gt["category"]]["fn"] += 1

    return stats


def compute_task_acc(preds, labels, *, acc_strategy='token',
                     is_encoder_decoder=False, cu_seqlens=None):
    if isinstance(preds, torch.Tensor):
        if torch.is_floating_point(labels):
            return {}
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()

    if preds.ndim >= 2 and not is_encoder_decoder:
        labels = labels[..., 1:]
        preds = preds[..., :-1]

    if np.issubdtype(labels.dtype, np.floating) or preds.shape != labels.shape:
        return {}

    masks = labels != -100

    tokenizer = _get_tokenizer()
    if tokenizer is None:
        return 0

    label_text = tokenizer.decode(labels[masks], skip_special_tokens=True)
    pred_text = tokenizer.decode(preds[masks], skip_special_tokens=True)

    total = compute_detection_metrics(pred_text, label_text)

    tp, fp, fn = total["tp"], total["fp"], total["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_iou = total["iou_sum"] / total["match_count"] if total["match_count"] > 0 else 0.0
    size_acc = total["size_correct"] / total["size_total"] if total["size_total"] > 0 else 0.0
    avg_quality = total["quality_sum"] / total["match_count"] if total["match_count"] > 0 else 0.0

    cat_f1_values = []
    cat_metrics = {}
    for cat in ALL_CATEGORIES:
        cat_stats = total["per_cat"].get(cat, {"tp": 0, "fp": 0, "fn": 0})
        cat_tp, cat_fp, cat_fn = cat_stats["tp"], cat_stats["fp"], cat_stats["fn"]
        cat_p = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0.0
        cat_r = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0.0
        cat_f1 = 2 * cat_p * cat_r / (cat_p + cat_r) if (cat_p + cat_r) > 0 else 0.0
        safe_name = cat.replace(" ", "_").replace("-", "").lower()
        cat_metrics[f"f1_{safe_name}"] = cat_f1
        if (cat_tp + cat_fn) > 0:
            cat_f1_values.append(cat_f1)

    macro_f1 = sum(cat_f1_values) / len(cat_f1_values) if cat_f1_values else 0.0

    metrics = {
        "detection_f1": f1,
        "detection_f1_macro": macro_f1,
        "detection_precision": precision,
        "detection_recall": recall,
        "bbox_iou_mean": avg_iou,
        "size_accuracy": size_acc,
        "avg_quality": avg_quality,
        "IMMetric": avg_quality,
    }
    metrics.update(cat_metrics)

    return metrics


def preprocess_logits_for_task(logits: torch.Tensor, labels: torch.Tensor):
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_task_metrics(eval_prediction: EvalPrediction, *,
                         acc_strategy='token',
                         is_encoder_decoder=False) -> Dict[str, float]:
    result = compute_task_acc(
        eval_prediction.predictions,
        eval_prediction.label_ids,
        acc_strategy=acc_strategy,
        is_encoder_decoder=is_encoder_decoder,
    )
    if isinstance(result, dict):
        return result
    return {"IMMetric": result if isinstance(result, (int, float)) else 0.0}


if _HAS_EVAL_METRICS_MAP:
    from swift.metrics.acc import EvalMetrics

    class IMEvalMetrics(EvalMetrics):
        def compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
            result = compute_task_acc(
                eval_prediction.predictions,
                eval_prediction.label_ids,
                is_encoder_decoder=self.trainer.is_encoder_decoder,
            )
            if isinstance(result, dict):
                return result
            return {"IMMetric": result if isinstance(result, (int, float)) else 0.0}

        def preprocess_logits_for_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            return logits.argmax(dim=-1)

    eval_metrics_map['IMMetric'] = IMEvalMetrics

if _HAS_SWIFT_PLUGIN:
    metric_mapping['IMMetric'] = (compute_task_metrics, preprocess_logits_for_task)
