"""CAD-aware metrics for Siemens drawing layout and feature extraction.

The task is not generic object detection: output quality depends on finding
the right class, locating it in the drawing, and preserving CAD dimension
notation.  These helpers keep those concerns separate and reusable across SFT
eval, checkpoint eval, post-hoc scoring, DPO ranking, and RL rewards.
"""

import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DOCUMENT_CATEGORIES = {
    "Title Block",
    "Notes",
    "Revision Table",
    "Bill of Materials",
}

VIEW_REGION_CATEGORIES = {
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

# "View" in this pipeline means layout/document elements plus actual views.
VIEW_CATEGORIES = DOCUMENT_CATEGORIES | VIEW_REGION_CATEGORIES

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

GROUP_CATEGORIES = {
    "Round Hole Group",
    "Rectangular Hole Group",
    "Slotted Hole Group",
}

ALL_CATEGORIES = sorted(VIEW_CATEGORIES) + sorted(FEATURE_CATEGORIES)

DEFAULT_LAYOUT_IOU = 0.50
DEFAULT_FEATURE_IOU = 0.40
DEFAULT_STRICT_FEATURE_IOU = 0.60


def safe_metric_name(category: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")


def calculate_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def intersection_area(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def box_area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def containment_ratio(inner_box: Sequence[float], outer_box: Sequence[float]) -> float:
    area = box_area(inner_box)
    if area <= 0:
        return 0.0
    return intersection_area(inner_box, outer_box) / area


def center_in_box(inner_box: Sequence[float], outer_box: Sequence[float]) -> bool:
    cx = (inner_box[0] + inner_box[2]) / 2.0
    cy = (inner_box[1] + inner_box[3]) / 2.0
    return outer_box[0] <= cx <= outer_box[2] and outer_box[1] <= cy <= outer_box[3]


def normalize_box(box) -> list:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return []
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return []

    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1, y1, x2, y2 = [v * 1000.0 for v in (x1, y1, x2, y2)]

    normalized = [int(round(v)) for v in (x1, y1, x2, y2)]
    normalized = [max(0, min(1000, v)) for v in normalized]
    if normalized[0] >= normalized[2] or normalized[1] >= normalized[3]:
        return []
    return normalized


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text or "").strip()


def parse_json_array_blocks(text: str) -> List[list]:
    """Return JSON list blocks in order.

    Multi-turn outputs often decode as two adjacent JSON arrays.  Keeping the
    blocks lets callers score the view turn and feature turn separately.
    """
    if not isinstance(text, str):
        return []
    text = _strip_think_tags(text)
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [parsed]
        if isinstance(parsed, dict):
            for key in ("features", "views", "items", "result", "value"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [value]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    blocks = []
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE):
        try:
            parsed = json.loads(match.group(1).strip())
            if isinstance(parsed, list):
                blocks.append(parsed)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    if blocks:
        return blocks

    for match in re.finditer(r"\[\s*(?:\{[\s\S]*?\}\s*,?\s*)*\]", text):
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                blocks.append(parsed)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return blocks


def parse_json_arrays(text: str) -> list:
    items = []
    for block in parse_json_array_blocks(text):
        items.extend(block)
    return items


def extract_items(items: Iterable[dict]) -> list:
    extracted = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category", item.get("label", ""))).strip()
        bbox = normalize_box(item.get("bbox_2d", item.get("bbox", [])))
        if not category or not bbox:
            continue
        extracted.append({
            "category": category,
            "bbox": bbox,
            "size": str(item.get("size", "")).strip(),
        })
    return extracted


def normalize_size(size, category: Optional[str] = None) -> str:
    value = str(size or "").strip()
    if not value:
        return ""

    value = value.replace("，", ",").replace("。", ".")
    value = re.sub(r"(?<=\d),(?=\d)", ".", value)
    value = value.replace("×", "x").replace("＊", "x").replace("*", "x")
    value = value.replace("－", "-").replace("–", "-").replace("—", "-")
    value = value.replace("φ", "Ø").replace("Φ", "Ø").replace("ø", "Ø").replace("⌀", "Ø")
    value = re.sub(r"\bDIA\.?\b", "Ø", value, flags=re.IGNORECASE)
    value = re.sub(r"\bDEG(?:REE)?S?\b", "°", value, flags=re.IGNORECASE)
    value = re.sub(r"(?<=\d)\s*MM\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", "", value)
    value = value.upper()

    if category in {"Round Hole", "Round Hole Group"}:
        # The annotations often omit the diameter symbol for round holes.
        value = re.sub(r"(^|[X-])Ø(?=\d)", r"\1", value)
    if category in {"Slotted Hole", "Slotted Hole Group"}:
        # The source data uses LL as a slot marker in some group dimensions.
        value = value.replace("LL", "")

    value = re.sub(r"(?<=\d)\.0+(?=$|[^0-9])", "", value)
    return value


def _canonical_number(token: str) -> str:
    try:
        number = float(token)
    except ValueError:
        return token
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def dimension_tokens(size, category: Optional[str] = None) -> Counter:
    value = normalize_size(size, category)
    tokens = Counter()
    if not value:
        return tokens

    for number in re.findall(r"\d+(?:\.\d+)?", value):
        tokens[f"N:{_canonical_number(number)}"] += 1

    for symbol in ("M", "R", "Ø", "□", "DP", "LH", "@", "°"):
        if symbol in value:
            tokens[f"S:{symbol}"] += 1
    if "START_FROM_END" in value:
        tokens["S:START_FROM_END"] += 1
    if "+" in value or "/" in value:
        tokens["S:TOL"] += 1
    return tokens


def size_similarity(pred_size, gt_size, category: Optional[str] = None) -> float:
    pred_norm = normalize_size(pred_size, category)
    gt_norm = normalize_size(gt_size, category)
    if pred_norm == gt_norm:
        return 1.0
    if not gt_norm:
        return 1.0 if not pred_norm else 0.0
    if not pred_norm:
        return 0.0

    pred_tokens = dimension_tokens(pred_norm, category)
    gt_tokens = dimension_tokens(gt_norm, category)
    if not gt_tokens:
        return 0.0

    overlap = sum((pred_tokens & gt_tokens).values())
    precision = overlap / sum(pred_tokens.values()) if pred_tokens else 0.0
    recall = overlap / sum(gt_tokens.values()) if gt_tokens else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    if category == "Threaded Hole" and "M" in gt_norm and "M" not in pred_norm:
        score = min(score, 0.5)
    if category == "Fillet" and "R" in gt_norm and "R" not in pred_norm and "Ø" not in pred_norm:
        score = min(score, 0.7)
    return max(0.0, min(1.0, score))


def size_exact_match(pred_size, gt_size, category: Optional[str] = None) -> bool:
    return normalize_size(pred_size, category) == normalize_size(gt_size, category)


def empty_stats() -> dict:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "weighted_tp": 0.0,
        "weighted_fp": 0.0,
        "weighted_fn": 0.0,
        "iou_sum": 0.0,
        "match_count": 0,
        "size_exact_correct": 0,
        "size_score_sum": 0.0,
        "size_total": 0,
        "per_cat": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
    }


def _category_weight(category: str, category_weights: Optional[Dict[str, float]]) -> float:
    if not category_weights:
        return 1.0
    try:
        return float(category_weights.get(category, 1.0))
    except (TypeError, ValueError):
        return 1.0


def score_items(
    pred_items: Iterable[dict],
    gt_items: Iterable[dict],
    categories: Optional[Iterable[str]] = None,
    iou_threshold: float = DEFAULT_FEATURE_IOU,
    score_size: bool = False,
    category_weights: Optional[Dict[str, float]] = None,
    count_unexpected: bool = False,
) -> dict:
    """Class-aware greedy matching at one IoU threshold.

    Predictions only match GT boxes of the same category.  This is important for
    CAD drawings where group boxes can overlap all child holes and where views
    overlap many features.
    """
    stats = empty_stats()
    expected = set(categories) if categories is not None else None

    gt_scored = [
        item for item in (gt_items or [])
        if expected is None or item.get("category") in expected
    ]
    pred_scored = []
    unexpected_pred = []
    for item in pred_items or []:
        if expected is None or item.get("category") in expected:
            pred_scored.append(item)
        elif count_unexpected:
            unexpected_pred.append(item)

    candidates: List[Tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(pred_scored):
        pred_box = pred.get("bbox", [])
        pred_cat = pred.get("category", "")
        if len(pred_box) != 4:
            continue
        for gt_idx, gt in enumerate(gt_scored):
            if pred_cat != gt.get("category", ""):
                continue
            gt_box = gt.get("bbox", [])
            if len(gt_box) != 4:
                continue
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                candidates.append((iou, pred_idx, gt_idx))

    candidates.sort(key=lambda item: item[0], reverse=True)
    matched_pred, matched_gt = set(), set()

    for iou, pred_idx, gt_idx in candidates:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        pred = pred_scored[pred_idx]
        gt = gt_scored[gt_idx]
        category = gt.get("category", "")
        weight = _category_weight(category, category_weights)

        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        stats["tp"] += 1
        stats["weighted_tp"] += weight
        stats["iou_sum"] += iou
        stats["match_count"] += 1
        stats["per_cat"][category]["tp"] += 1

        gt_size = str(gt.get("size", "")).strip()
        if score_size and gt_size:
            stats["size_total"] += 1
            if size_exact_match(pred.get("size", ""), gt_size, category):
                stats["size_exact_correct"] += 1
            stats["size_score_sum"] += size_similarity(pred.get("size", ""), gt_size, category)

    for pred_idx, pred in enumerate(pred_scored):
        if pred_idx in matched_pred:
            continue
        category = pred.get("category", "")
        weight = _category_weight(category, category_weights)
        stats["fp"] += 1
        stats["weighted_fp"] += weight
        stats["per_cat"][category]["fp"] += 1

    for pred in unexpected_pred:
        category = pred.get("category", "")
        weight = _category_weight(category, category_weights)
        stats["fp"] += 1
        stats["weighted_fp"] += weight
        stats["per_cat"][category]["fp"] += 1

    for gt_idx, gt in enumerate(gt_scored):
        if gt_idx in matched_gt:
            continue
        category = gt.get("category", "")
        weight = _category_weight(category, category_weights)
        stats["fn"] += 1
        stats["weighted_fn"] += weight
        stats["per_cat"][category]["fn"] += 1

    return stats


def merge_stats(total: dict, stats: dict) -> dict:
    for key in (
        "tp", "fp", "fn", "weighted_tp", "weighted_fp", "weighted_fn",
        "iou_sum", "match_count", "size_exact_correct", "size_score_sum", "size_total",
    ):
        total[key] += stats.get(key, 0)
    for category, values in stats.get("per_cat", {}).items():
        for key in ("tp", "fp", "fn"):
            total["per_cat"][category][key] += values.get(key, 0)
    return total


def add_stats(*stats_list: dict) -> dict:
    total = empty_stats()
    for stats in stats_list:
        merge_stats(total, stats)
    return total


def prf(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def macro_f1(stats: dict, categories: Iterable[str]) -> float:
    values = []
    for category in categories:
        cat_stats = stats.get("per_cat", {}).get(category, {"tp": 0, "fp": 0, "fn": 0})
        if cat_stats["tp"] + cat_stats["fn"] <= 0:
            continue
        values.append(prf(cat_stats["tp"], cat_stats["fp"], cat_stats["fn"])[2])
    return sum(values) / len(values) if values else 0.0


def summarize_stats(stats: dict, categories: Optional[Iterable[str]] = None) -> dict:
    precision, recall, f1 = prf(stats["tp"], stats["fp"], stats["fn"])
    weighted_precision, weighted_recall, weighted_f1 = prf(
        stats["weighted_tp"], stats["weighted_fp"], stats["weighted_fn"]
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "mean_iou": stats["iou_sum"] / stats["match_count"] if stats["match_count"] else 0.0,
        "size_accuracy": (
            stats["size_exact_correct"] / stats["size_total"] if stats["size_total"] else 0.0
        ),
        "size_similarity": (
            stats["size_score_sum"] / stats["size_total"] if stats["size_total"] else 0.0
        ),
        "macro_f1": macro_f1(stats, categories or []),
    }


def flatten_metrics(
    layout_stats: dict,
    feature_stats: dict,
    combined_stats: Optional[dict] = None,
    strict_feature_stats: Optional[dict] = None,
) -> dict:
    if combined_stats is None:
        combined_stats = add_stats(layout_stats, feature_stats)
    if strict_feature_stats is None:
        strict_feature_stats = feature_stats

    layout = summarize_stats(layout_stats, sorted(VIEW_CATEGORIES))
    feature = summarize_stats(feature_stats, sorted(FEATURE_CATEGORIES))
    combined = summarize_stats(combined_stats, ALL_CATEGORIES)
    strict_feature = summarize_stats(strict_feature_stats, sorted(FEATURE_CATEGORIES))

    feature_score = (
        0.55 * feature["f1"]
        + 0.20 * feature["size_similarity"]
        + 0.15 * feature["mean_iou"]
        + 0.10 * strict_feature["f1"]
    )
    layout_score = 0.85 * layout["f1"] + 0.15 * layout["mean_iou"]
    layout_present = (
        layout_stats["tp"] + layout_stats["fp"] + layout_stats["fn"]
    ) > 0
    feature_present = (
        feature_stats["tp"] + feature_stats["fp"] + feature_stats["fn"]
    ) > 0
    if layout_present and feature_present:
        cad_score = 0.75 * feature_score + 0.25 * layout_score
    elif feature_present:
        cad_score = feature_score
    elif layout_present:
        cad_score = layout_score
    else:
        cad_score = 1.0

    metrics = {
        "layout_precision": layout["precision"],
        "layout_recall": layout["recall"],
        "layout_f1": layout["f1"],
        "layout_f1_macro": layout["macro_f1"],
        "layout_bbox_iou_mean": layout["mean_iou"],
        "feature_precision": feature["precision"],
        "feature_recall": feature["recall"],
        "feature_f1": feature["f1"],
        "feature_f1_macro": feature["macro_f1"],
        "feature_f1_strict": strict_feature["f1"],
        "feature_bbox_iou_mean": feature["mean_iou"],
        "size_accuracy": feature["size_accuracy"],
        "size_similarity": feature["size_similarity"],
        "detection_precision": combined["precision"],
        "detection_recall": combined["recall"],
        "detection_f1": combined["f1"],
        "detection_f1_macro": combined["macro_f1"],
        "bbox_iou_mean": combined["mean_iou"],
        "cad_layout_score": layout_score,
        "cad_feature_score": feature_score,
        "cad_extraction_score": cad_score,
        "IMMetric": cad_score,
    }

    for category in ALL_CATEGORIES:
        source = layout_stats if category in VIEW_CATEGORIES else feature_stats
        cat_stats = source.get("per_cat", {}).get(category, {"tp": 0, "fp": 0, "fn": 0})
        metrics[f"f1_{safe_metric_name(category)}"] = prf(
            cat_stats["tp"], cat_stats["fp"], cat_stats["fn"]
        )[2]
    return metrics


def compute_combined_metrics_from_items(
    pred_items: Iterable[dict],
    gt_items: Iterable[dict],
    layout_iou: float = DEFAULT_LAYOUT_IOU,
    feature_iou: float = DEFAULT_FEATURE_IOU,
    strict_feature_iou: float = DEFAULT_STRICT_FEATURE_IOU,
    category_weights: Optional[Dict[str, float]] = None,
) -> Tuple[dict, dict, dict, dict, dict]:
    pred_items = list(pred_items or [])
    gt_items = list(gt_items or [])
    layout_stats = score_items(
        pred_items, gt_items, VIEW_CATEGORIES, layout_iou,
        category_weights=category_weights, count_unexpected=False,
    )
    feature_stats = score_items(
        pred_items, gt_items, FEATURE_CATEGORIES, feature_iou,
        score_size=True, category_weights=category_weights, count_unexpected=False,
    )
    strict_feature_stats = score_items(
        pred_items, gt_items, FEATURE_CATEGORIES, strict_feature_iou,
        score_size=True, category_weights=category_weights, count_unexpected=False,
    )
    combined_stats = add_stats(layout_stats, feature_stats)
    unknown_stats = score_items(
        [item for item in pred_items if item.get("category") not in set(ALL_CATEGORIES)],
        [],
        ALL_CATEGORIES,
        min(layout_iou, feature_iou),
        category_weights=category_weights,
        count_unexpected=True,
    )
    merge_stats(combined_stats, unknown_stats)
    metrics = flatten_metrics(layout_stats, feature_stats, combined_stats, strict_feature_stats)
    return metrics, layout_stats, feature_stats, combined_stats, strict_feature_stats


def compute_split_metrics_from_items(
    pred_views: Iterable[dict],
    pred_features: Iterable[dict],
    gt_views: Iterable[dict],
    gt_features: Iterable[dict],
    layout_iou: float = DEFAULT_LAYOUT_IOU,
    feature_iou: float = DEFAULT_FEATURE_IOU,
    strict_feature_iou: float = DEFAULT_STRICT_FEATURE_IOU,
    category_weights: Optional[Dict[str, float]] = None,
) -> Tuple[dict, dict, dict, dict, dict]:
    pred_views = list(pred_views or [])
    pred_features = list(pred_features or [])
    gt_views = list(gt_views or [])
    gt_features = list(gt_features or [])

    layout_stats = score_items(
        pred_views, gt_views, VIEW_CATEGORIES, layout_iou,
        category_weights=category_weights, count_unexpected=True,
    )
    feature_stats = score_items(
        pred_features, gt_features, FEATURE_CATEGORIES, feature_iou,
        score_size=True, category_weights=category_weights, count_unexpected=True,
    )
    strict_feature_stats = score_items(
        pred_features, gt_features, FEATURE_CATEGORIES, strict_feature_iou,
        score_size=True, category_weights=category_weights, count_unexpected=True,
    )
    combined_stats = add_stats(layout_stats, feature_stats)
    metrics = flatten_metrics(layout_stats, feature_stats, combined_stats, strict_feature_stats)
    return metrics, layout_stats, feature_stats, combined_stats, strict_feature_stats


def compute_metrics_from_text(
    pred_text: str,
    label_text: str,
    layout_iou: float = DEFAULT_LAYOUT_IOU,
    feature_iou: float = DEFAULT_FEATURE_IOU,
    strict_feature_iou: float = DEFAULT_STRICT_FEATURE_IOU,
    category_weights: Optional[Dict[str, float]] = None,
) -> dict:
    pred_blocks = parse_json_array_blocks(pred_text)
    label_blocks = parse_json_array_blocks(label_text)

    if len(pred_blocks) >= 2 and len(label_blocks) >= 2:
        pred_views = extract_items(pred_blocks[0])
        pred_features = extract_items(pred_blocks[1])
        gt_views = extract_items(label_blocks[0])
        gt_features = extract_items(label_blocks[1])
        metrics, _, _, _, _ = compute_split_metrics_from_items(
            pred_views, pred_features, gt_views, gt_features,
            layout_iou=layout_iou, feature_iou=feature_iou,
            strict_feature_iou=strict_feature_iou,
            category_weights=category_weights,
        )
        return metrics

    pred_items = extract_items(item for block in pred_blocks for item in block)
    gt_items = extract_items(item for block in label_blocks for item in block)
    metrics, _, _, _, _ = compute_combined_metrics_from_items(
        pred_items, gt_items,
        layout_iou=layout_iou, feature_iou=feature_iou,
        strict_feature_iou=strict_feature_iou,
        category_weights=category_weights,
    )
    return metrics


def quality_score(
    pred_items: Iterable[dict],
    gt_items: Iterable[dict],
    categories: Iterable[str],
    iou_threshold: float,
    include_size: bool = False,
    category_weights: Optional[Dict[str, float]] = None,
) -> float:
    pred_items = list(pred_items or [])
    gt_items = list(gt_items or [])
    expected = set(categories)
    if not [g for g in gt_items if g.get("category") in expected] and not pred_items:
        return 1.0
    stats = score_items(
        pred_items, gt_items, expected, iou_threshold,
        score_size=include_size, category_weights=category_weights,
        count_unexpected=True,
    )
    summary = summarize_stats(stats, expected)
    if include_size:
        return max(0.0, min(1.0, (
            0.60 * summary["weighted_f1"]
            + 0.25 * summary["size_similarity"]
            + 0.15 * summary["mean_iou"]
        )))
    return max(0.0, min(1.0, (
        0.85 * summary["weighted_f1"]
        + 0.15 * summary["mean_iou"]
    )))


def serializable_stats(stats: dict) -> dict:
    result = deepcopy(stats)
    result["per_cat"] = {k: dict(v) for k, v in stats.get("per_cat", {}).items()}
    return result
