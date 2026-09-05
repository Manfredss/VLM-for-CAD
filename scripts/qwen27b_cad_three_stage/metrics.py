#!/usr/bin/env python3
"""Dependency-free metrics for the canonical three-stage CAD task.

Matching is class-aware and global within an image: every same-class pair above
the IoU threshold is ranked by IoU, then greedily selected one-to-one.  This
avoids the common order-dependent bug where a mediocre early match blocks the
best later match, especially for overlapping Group/child annotations.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Iterable, Mapping, Optional, Sequence

try:  # Works both as a package import and as ``python metrics.py``.
    from .cad_schema import (
        DOCUMENT_CATEGORIES,
        FEATURE_CATEGORIES,
        GENERIC_VIEW_REGION,
        STAGE1_CATEGORIES,
        VIEW_CATEGORIES,
        bbox_iou,
        normalize_bbox,
        normalize_category,
        normalize_projection_method,
        normalize_size,
        parse_json_object,
        validate_cross_stage_outputs,
        validate_stage_output,
    )
except ImportError:
    from cad_schema import (  # type: ignore
        DOCUMENT_CATEGORIES,
        FEATURE_CATEGORIES,
        GENERIC_VIEW_REGION,
        STAGE1_CATEGORIES,
        VIEW_CATEGORIES,
        bbox_iou,
        normalize_bbox,
        normalize_category,
        normalize_projection_method,
        normalize_size,
        parse_json_object,
        validate_cross_stage_outputs,
        validate_stage_output,
    )


DEFAULT_LAYOUT_IOU = 0.50
DEFAULT_VIEW_IOU = 0.50
DEFAULT_FEATURE_IOU = 0.40
DEFAULT_STRICT_FEATURE_IOU = 0.60

ROUND_FOCUS_CATEGORIES = ("Round Hole", "Round Hole Group")
SLOTTED_FOCUS_CATEGORIES = ("Slotted Hole", "Slotted Hole Group")


def _canonical_number(token: str) -> str:
    try:
        number = float(token)
    except (TypeError, ValueError):
        return str(token)
    return (
        str(int(round(number)))
        if abs(number - round(number)) < 1e-9
        else f"{number:.6f}".rstrip("0").rstrip(".")
    )


def dimension_tokens(size: Any, category: Optional[str] = None) -> Counter[str]:
    value = normalize_size(size, category)
    tokens: Counter[str] = Counter()
    for number in re.findall(r"\d+(?:\.\d+)?", value):
        tokens[f"N:{_canonical_number(number)}"] += 1
    for symbol in ("M", "R", "Ø", "H7", "DP", "°", "@", "+", "/"):
        if symbol in value:
            tokens[f"S:{symbol}"] += 1
    return tokens


def size_similarity(
    predicted: Any, target: Any, category: Optional[str] = None
) -> float:
    pred_norm, target_norm = (
        normalize_size(predicted, category),
        normalize_size(target, category),
    )
    if pred_norm == target_norm:
        return 1.0
    if not pred_norm or not target_norm:
        return 0.0
    pred_tokens, target_tokens = (
        dimension_tokens(pred_norm, category),
        dimension_tokens(target_norm, category),
    )
    overlap = sum((pred_tokens & target_tokens).values())
    precision = overlap / sum(pred_tokens.values()) if pred_tokens else 0.0
    recall = overlap / sum(target_tokens.values()) if target_tokens else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def size_normalized_equal(
    predicted: Any, target: Any, category: Optional[str] = None
) -> bool:
    return normalize_size(predicted, category) == normalize_size(target, category)


def _empty_category_stats() -> dict[str, Any]:
    return {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0, "matches": 0}


def empty_stats() -> dict[str, Any]:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "iou_sum": 0.0,
        "match_count": 0,
        "size_exact_correct": 0,
        "size_similarity_sum": 0.0,
        "size_gt_total": 0,
        "size_matched_total": 0,
        "size_empty_gt_matched_total": 0,
        "size_empty_gt_hallucinated": 0,
        "matches": [],
        "per_category": defaultdict(_empty_category_stats),
    }


def _normalized_item(
    item: Any, kind: str, region_boxes: Optional[Mapping[str, Sequence[float]]] = None
) -> Optional[dict[str, Any]]:
    if not isinstance(item, Mapping):
        return None
    raw_category = item.get("category", item.get("label", ""))
    if kind == "layout":
        category = normalize_category(raw_category, kind="stage1")
        if category is None and normalize_category(raw_category, kind="view"):
            category = GENERIC_VIEW_REGION
    else:
        category = normalize_category(raw_category, kind=kind)
    # Preserve an invalid prediction label so it becomes an FP instead of being
    # silently dropped.  Validation rate separately exposes schema failures.
    if category is None:
        category = str(raw_category or "").strip() or "<missing-category>"
    region_id = str(item.get("region_id", "")).strip()
    box_value = item.get("bbox", item.get("bbox_2d"))
    if box_value is None and region_id and region_boxes:
        box_value = region_boxes.get(region_id)
    bbox = normalize_bbox(box_value)
    return {
        "category": category,
        "bbox": bbox,
        "size": str(item.get("size", "") or "").strip(),
        "region_id": region_id,
    }


def normalize_items(
    items: Iterable[Any],
    kind: str,
    region_boxes: Optional[Mapping[str, Sequence[float]]] = None,
) -> list[dict[str, Any]]:
    result = []
    for item in items or []:
        normalized = _normalized_item(item, kind, region_boxes)
        if normalized is not None:
            result.append(normalized)
    return result


def greedy_match(
    predicted: Iterable[Mapping[str, Any]],
    target: Iterable[Mapping[str, Any]],
    *,
    iou_threshold: float,
    categories: Optional[Iterable[str]] = None,
    score_size: bool = False,
    ignore_outside_categories: bool = False,
) -> dict[str, Any]:
    """Perform class-aware global greedy matching for one image."""
    pred_items, gt_items = list(predicted or []), list(target or [])
    expected = set(categories) if categories is not None else None
    if expected is not None and ignore_outside_categories:
        pred_items = [p for p in pred_items if p.get("category") in expected]
        gt_items = [g for g in gt_items if g.get("category") in expected]

    stats = empty_stats()
    stats["size_gt_total"] = (
        sum(
            1
            for gt in gt_items
            if str(gt.get("size", "")).strip()
            and (expected is None or gt.get("category") in expected)
        )
        if score_size
        else 0
    )

    candidates: list[tuple[float, int, int]] = []
    for pred_index, pred in enumerate(pred_items):
        category = pred.get("category")
        if expected is not None and category not in expected:
            continue
        if len(pred.get("bbox", [])) != 4:
            continue
        for gt_index, gt in enumerate(gt_items):
            if category != gt.get("category") or (
                expected is not None and gt.get("category") not in expected
            ):
                continue
            if len(gt.get("bbox", [])) != 4:
                continue
            iou = bbox_iou(pred["bbox"], gt["bbox"])
            if iou >= iou_threshold:
                candidates.append((iou, pred_index, gt_index))
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    for iou, pred_index, gt_index in candidates:
        if pred_index in matched_pred or gt_index in matched_gt:
            continue
        matched_pred.add(pred_index)
        matched_gt.add(gt_index)
        pred, gt = pred_items[pred_index], gt_items[gt_index]
        category = str(gt.get("category", ""))
        stats["tp"] += 1
        stats["iou_sum"] += iou
        stats["match_count"] += 1
        stats["per_category"][category]["tp"] += 1
        stats["per_category"][category]["iou_sum"] += iou
        stats["per_category"][category]["matches"] += 1
        match = {
            "pred_index": pred_index,
            "target_index": gt_index,
            "category": category,
            "iou": iou,
        }
        if score_size:
            if str(gt.get("size", "")).strip():
                stats["size_matched_total"] += 1
                exact = size_normalized_equal(
                    pred.get("size"), gt.get("size"), category
                )
                similarity = size_similarity(pred.get("size"), gt.get("size"), category)
                stats["size_exact_correct"] += int(exact)
                stats["size_similarity_sum"] += similarity
                match.update({"size_exact": exact, "size_similarity": similarity})
            else:
                hallucinated = bool(str(pred.get("size", "")).strip())
                stats["size_empty_gt_matched_total"] += 1
                stats["size_empty_gt_hallucinated"] += int(hallucinated)
                match["size_hallucinated_on_empty_gt"] = hallucinated
        stats["matches"].append(match)

    for pred_index, pred in enumerate(pred_items):
        category = str(pred.get("category", ""))
        if pred_index in matched_pred:
            continue
        if (
            expected is not None
            and category not in expected
            and ignore_outside_categories
        ):
            continue
        stats["fp"] += 1
        stats["per_category"][category]["fp"] += 1
    for gt_index, gt in enumerate(gt_items):
        category = str(gt.get("category", ""))
        if gt_index in matched_gt:
            continue
        if (
            expected is not None
            and category not in expected
            and ignore_outside_categories
        ):
            continue
        stats["fn"] += 1
        stats["per_category"][category]["fn"] += 1
    return stats


def merge_stats(
    total: dict[str, Any], addition: Mapping[str, Any], *, keep_matches: bool = False
) -> dict[str, Any]:
    for key in (
        "tp",
        "fp",
        "fn",
        "iou_sum",
        "match_count",
        "size_exact_correct",
        "size_similarity_sum",
        "size_gt_total",
        "size_matched_total",
        "size_empty_gt_matched_total",
        "size_empty_gt_hallucinated",
    ):
        total[key] += addition.get(key, 0)
    if keep_matches:
        total["matches"].extend(addition.get("matches", []))
    for category, values in addition.get("per_category", {}).items():
        for key in ("tp", "fp", "fn", "iou_sum", "matches"):
            total["per_category"][category][key] += values.get(key, 0)
    return total


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def exact_pass(stats: Mapping[str, Any], *, require_size: bool = False) -> bool:
    if stats.get("fp", 0) or stats.get("fn", 0):
        return False
    return not require_size or (
        stats.get("size_exact_correct", 0) == stats.get("size_gt_total", 0)
        and not stats.get("size_empty_gt_hallucinated", 0)
    )


def summarize_stats(
    stats: Mapping[str, Any], categories: Optional[Iterable[str]] = None
) -> dict[str, Any]:
    precision, recall, f1 = _prf(
        int(stats.get("tp", 0)), int(stats.get("fp", 0)), int(stats.get("fn", 0))
    )
    requested = list(categories or stats.get("per_category", {}).keys())
    per_category: dict[str, Any] = {}
    macro_values = []
    for category in requested:
        values = stats.get("per_category", {}).get(category, _empty_category_stats())
        cat_precision, cat_recall, cat_f1 = _prf(
            values["tp"], values["fp"], values["fn"]
        )
        if values["tp"] + values["fp"] + values["fn"]:
            macro_values.append(cat_f1)
        per_category[category] = {
            "tp": values["tp"],
            "fp": values["fp"],
            "fn": values["fn"],
            "precision": cat_precision,
            "recall": cat_recall,
            "f1": cat_f1,
            "mean_iou": values["iou_sum"] / values["matches"]
            if values["matches"]
            else 0.0,
        }
    size_gt_total = int(stats.get("size_gt_total", 0))
    size_matched = int(stats.get("size_matched_total", 0))
    empty_gt_matched = int(stats.get("size_empty_gt_matched_total", 0))
    empty_gt_hallucinated = int(stats.get("size_empty_gt_hallucinated", 0))
    return {
        "tp": int(stats.get("tp", 0)),
        "fp": int(stats.get("fp", 0)),
        "fn": int(stats.get("fn", 0)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": sum(macro_values) / len(macro_values) if macro_values else 0.0,
        "mean_iou": stats.get("iou_sum", 0.0) / stats.get("match_count", 1)
        if stats.get("match_count", 0)
        else 0.0,
        "size_normalized_accuracy": stats.get("size_exact_correct", 0) / size_gt_total
        if size_gt_total
        else 0.0,
        "size_normalized_accuracy_matched": stats.get("size_exact_correct", 0)
        / size_matched
        if size_matched
        else 0.0,
        "size_similarity": stats.get("size_similarity_sum", 0.0) / size_gt_total
        if size_gt_total
        else 0.0,
        "size_gt_total": size_gt_total,
        "size_empty_gt_matched_total": empty_gt_matched,
        "size_empty_gt_hallucinated": empty_gt_hallucinated,
        "empty_gt_size_hallucination_rate": (
            empty_gt_hallucinated / empty_gt_matched if empty_gt_matched else 0.0
        ),
        "per_category": per_category,
    }


def _jsonish(value: Any) -> Any:
    if isinstance(value, (Mapping, list)):
        return deepcopy(value)
    if not isinstance(value, str):
        return None
    obj = parse_json_object(value)
    if obj is not None:
        return obj
    text = re.sub(r"<think>[\s\S]*?</think>", "", value).strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _assistant_outputs(messages: Any) -> list[dict[str, Any]]:
    outputs = []
    if not isinstance(messages, list):
        return outputs
    for message in messages:
        if (
            not isinstance(message, Mapping)
            or str(message.get("role", "")).casefold() != "assistant"
        ):
            continue
        parsed = _jsonish(message.get("content", ""))
        if isinstance(parsed, Mapping):
            outputs.append(dict(parsed))
    return outputs


def _flat_items_from_raw_tasks(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for task in (
        record.get("tasks", []) if isinstance(record.get("tasks"), list) else []
    ):
        if not isinstance(task, Mapping):
            continue
        for task_value in (
            task.get("task_values", [])
            if isinstance(task.get("task_values"), list)
            else []
        ):
            if not isinstance(task_value, Mapping):
                continue
            value = task_value.get("value", {})
            if not isinstance(value, Mapping):
                value = {}
            result.append(
                {
                    "category": value.get("label", value.get("category", "")),
                    "size": value.get("size", ""),
                    "bbox": task_value.get("bbox", value.get("bbox", [])),
                }
            )
    return result


def stages_from_flat_items(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    regions, views, features = [], [], []
    view_index = document_index = 0
    for item in items or []:
        raw_category = item.get("category", item.get("label", ""))
        feature_category = normalize_category(raw_category, kind="feature")
        view_category = normalize_category(raw_category, kind="view")
        document_category = normalize_category(raw_category, kind="document")
        bbox = normalize_bbox(item.get("bbox", item.get("bbox_2d")))
        if feature_category:
            features.append(
                {
                    "category": feature_category,
                    "size": str(item.get("size", "") or ""),
                    "bbox": bbox,
                }
            )
        elif view_category:
            view_index += 1
            region_id = str(item.get("region_id", "") or f"r{view_index:03d}")
            regions.append(
                {"region_id": region_id, "category": GENERIC_VIEW_REGION, "bbox": bbox}
            )
            views.append(
                {"region_id": region_id, "category": view_category, "bbox": bbox}
            )
        elif document_category:
            document_index += 1
            region_id = str(item.get("region_id", "") or f"d{document_index:03d}")
            regions.append(
                {"region_id": region_id, "category": document_category, "bbox": bbox}
            )
    scope = [
        category
        for category in FEATURE_CATEGORIES
        if any(f["category"] == category for f in features)
    ]
    return {
        "stage1": {"regions": regions},
        "stage2": {"projection_method": "unknown", "views": views},
        "stage3": {"feature_scope": scope, "features": features},
    }


def coerce_three_stage(record: Any) -> dict[str, Any]:
    """Coerce canonical, inference, ms-swift, and legacy-flat rows to 3 stages."""
    parsed = _jsonish(record)
    if not isinstance(parsed, Mapping):
        return {"stage1": None, "stage2": None, "stage3": None}
    obj: Mapping[str, Any] = parsed
    for wrapper in ("ground_truth", "target", "targets", "prediction", "predictions"):
        if isinstance(obj.get(wrapper), Mapping):
            obj = obj[wrapper]
            break

    stages: dict[str, Any] = {"stage1": None, "stage2": None, "stage3": None}
    for number in (1, 2, 3):
        for key in (
            f"stage{number}",
            f"step{number}",
            f"stage_{number}",
            f"step_{number}",
            f"stage{number}_output",
            f"step{number}_output",
        ):
            value = _jsonish(obj.get(key))
            if isinstance(value, Mapping):
                stages[f"stage{number}"] = dict(value)
                break

    if stages["stage1"] is None and isinstance(obj.get("regions"), list):
        stages["stage1"] = {"regions": deepcopy(obj["regions"])}
    if stages["stage2"] is None and isinstance(obj.get("views"), list):
        stages["stage2"] = {
            "projection_method": obj.get("projection_method", "unknown"),
            "views": deepcopy(obj["views"]),
        }
    if stages["stage3"] is None and isinstance(obj.get("features"), list):
        stages["stage3"] = {
            "feature_scope": deepcopy(
                obj.get("feature_scope", list(FEATURE_CATEGORIES))
            ),
            "features": deepcopy(obj["features"]),
        }

    outputs = _assistant_outputs(obj.get("messages"))
    for output in outputs:
        if "regions" in output and stages["stage1"] is None:
            stages["stage1"] = output
        elif "views" in output and stages["stage2"] is None:
            stages["stage2"] = output
        elif "features" in output and stages["stage3"] is None:
            stages["stage3"] = output

    flat = obj.get("result")
    if not isinstance(flat, list):
        flat = _flat_items_from_raw_tasks(obj)
    if (
        isinstance(flat, list)
        and flat
        and all(value is None for value in stages.values())
    ):
        stages = stages_from_flat_items(flat)
    return stages


def _record_scope(record: Any, key: str, kind: str) -> Optional[list[str]]:
    """Read an explicit annotation scope; ``None`` means legacy/full scope."""
    obj = _jsonish(record)
    if not isinstance(obj, Mapping):
        return None
    candidates: list[Mapping[str, Any]] = [obj]
    for wrapper in ("ground_truth", "target", "targets"):
        if isinstance(obj.get(wrapper), Mapping):
            candidates.append(obj[wrapper])
    for candidate in tuple(candidates):
        if isinstance(candidate.get("metadata"), Mapping):
            candidates.append(candidate["metadata"])
        if isinstance(candidate.get("annotation_scope"), Mapping):
            candidates.append(candidate["annotation_scope"])
    found = False
    values: Any = None
    for candidate in candidates:
        if key in candidate:
            found, values = True, candidate.get(key)
            break
    if not found:
        return None
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        canonical = normalize_category(value, kind=kind)
        if canonical and canonical not in result:
            result.append(canonical)
    return result


def extract_stage_items(stages: Mapping[str, Any]) -> dict[str, Any]:
    stage1 = stages.get("stage1") if isinstance(stages.get("stage1"), Mapping) else None
    stage2 = stages.get("stage2") if isinstance(stages.get("stage2"), Mapping) else None
    stage3 = stages.get("stage3") if isinstance(stages.get("stage3"), Mapping) else None

    raw_regions = stage1.get("regions", []) if stage1 else []
    layout = normalize_items(raw_regions, "layout")
    region_boxes = {
        str(item.get("region_id", "")): normalize_bbox(item.get("bbox"))
        for item in raw_regions
        if isinstance(item, Mapping) and str(item.get("region_id", ""))
    }
    views = normalize_items(
        stage2.get("views", []) if stage2 else [], "view", region_boxes
    )
    features = normalize_items(stage3.get("features", []) if stage3 else [], "feature")
    scope = []
    if stage3 and isinstance(stage3.get("feature_scope"), list):
        for category in stage3["feature_scope"]:
            canonical = normalize_category(category, kind="feature")
            if canonical and canonical not in scope:
                scope.append(canonical)
    if stage3 and not scope:
        scope = list(FEATURE_CATEGORIES)
    return {
        "layout": layout,
        "views": views,
        "features": features,
        "feature_scope": scope,
        "projection_method": normalize_projection_method(
            stage2.get("projection_method")
        )
        if stage2
        else None,
        "present": {
            "layout": stage1 is not None,
            "views": stage2 is not None,
            "features": stage3 is not None,
        },
    }


def _validity(stages: Mapping[str, Any]) -> tuple[bool, dict[str, list[str]]]:
    errors: dict[str, list[str]] = {}
    for number in (1, 2, 3):
        payload = stages.get(f"stage{number}")
        if payload is None:
            continue
        stage_errors = validate_stage_output(number, payload, strict=False)
        if stage_errors:
            errors[f"stage{number}"] = stage_errors
    return not errors, errors


def _subset(
    items: Iterable[Mapping[str, Any]], categories: Iterable[str]
) -> list[Mapping[str, Any]]:
    expected = set(categories)
    return [item for item in items if item.get("category") in expected]


def _filter_scoped_view_regions(
    layout_items: Iterable[Mapping[str, Any]],
    view_items: Iterable[Mapping[str, Any]],
    allowed_views: Iterable[str],
) -> list[Mapping[str, Any]]:
    """Ignore generic regions explicitly classified outside a partial scope.

    An unclassified generic region is retained because it may be an in-scope
    Stage-1 detection whose Stage-2 semantic label was missed.
    """
    allowed = set(allowed_views)
    classified = {
        str(item.get("region_id", "")): item.get("category")
        for item in view_items
        if str(item.get("region_id", ""))
    }
    result = []
    for item in layout_items:
        region_id = str(item.get("region_id", ""))
        if (
            item.get("category") == GENERIC_VIEW_REGION
            and region_id in classified
            and classified[region_id] not in allowed
        ):
            continue
        result.append(item)
    return result


def _category_exact_rates(
    trackers: Mapping[str, Mapping[str, int]],
) -> dict[str, float]:
    return {
        category: values["pass"] / values["total"] if values["total"] else 0.0
        for category, values in trackers.items()
    }


def evaluate_pairs(
    pairs: Iterable[Any],
    *,
    layout_iou: float = DEFAULT_LAYOUT_IOU,
    view_iou: float = DEFAULT_VIEW_IOU,
    feature_iou: float = DEFAULT_FEATURE_IOU,
    strict_feature_iou: float = DEFAULT_STRICT_FEATURE_IOU,
) -> dict[str, Any]:
    """Evaluate an iterable of ``(prediction, ground_truth)`` pairs.

    A mapping pair may instead contain ``prediction``, ``ground_truth`` and an
    optional boolean ``json_valid``.  Missing GT stages are not scored, which is
    required for Stage-3-only crop training/evaluation rows.
    """
    totals = {
        "layout": empty_stats(),
        "views": empty_stats(),
        "features": empty_stats(),
        "features_strict": empty_stats(),
    }
    stage_pass = {
        name: {"pass": 0, "total": 0}
        for name in ("layout", "views", "features", "overall")
    }
    category_exact: dict[str, dict[str, int]] = defaultdict(
        lambda: {"pass": 0, "total": 0}
    )
    round_total, slotted_total, view_focus_total = (
        empty_stats(),
        empty_stats(),
        empty_stats(),
    )
    focus_pass = {
        "round": {"pass": 0, "total": 0},
        "slotted": {"pass": 0, "total": 0},
        "view": {"pass": 0, "total": 0},
    }
    projection = {"correct": 0, "total": 0, "known_correct": 0, "known_total": 0}
    valid_count = constraint_valid_count = sample_count = 0
    per_image = []

    for index, pair in enumerate(pairs):
        explicit_valid = None
        if isinstance(pair, Mapping):
            pred_record = pair.get("prediction")
            gt_record = pair.get("ground_truth", pair.get("target"))
            explicit_valid = pair.get("json_valid")
            sample_id = str(pair.get("sample_id", index))
        else:
            pred_record, gt_record = pair
            sample_id = str(index)
        pred_stages, gt_stages = (
            coerce_three_stage(pred_record),
            coerce_three_stage(gt_record),
        )
        pred_data, gt_data = (
            extract_stage_items(pred_stages),
            extract_stage_items(gt_stages),
        )
        view_scope = _record_scope(gt_record, "view_scope", "view")
        document_scope = _record_scope(gt_record, "document_scope", "document")
        scoped_views = list(VIEW_CATEGORIES) if view_scope is None else view_scope
        scoped_documents = (
            list(DOCUMENT_CATEGORIES) if document_scope is None else document_scope
        )
        scoped_layout = (
            [GENERIC_VIEW_REGION] if scoped_views else []
        ) + scoped_documents
        schema_valid, schema_errors = _validity(pred_stages)
        constraint_errors = validate_cross_stage_outputs(
            pred_stages.get("stage1"),
            pred_stages.get("stage2"),
            pred_stages.get("stage3"),
        )
        constraint_valid = not constraint_errors
        is_valid = bool(explicit_valid) if explicit_valid is not None else schema_valid
        valid_count += int(is_valid)
        constraint_valid_count += int(constraint_valid)
        sample_count += 1
        image_result: dict[str, Any] = {
            "sample_id": sample_id,
            "json_valid": is_valid,
            "validation_errors": schema_errors,
            "constraint_valid": constraint_valid,
            "constraint_errors": constraint_errors,
        }
        applicable_results = []

        if gt_data["present"]["layout"]:
            pred_layout, gt_layout = pred_data["layout"], gt_data["layout"]
            if view_scope is not None:
                pred_layout = _filter_scoped_view_regions(
                    pred_layout, pred_data["views"], scoped_views
                )
                gt_layout = _filter_scoped_view_regions(
                    gt_layout, gt_data["views"], scoped_views
                )
            stats = greedy_match(
                pred_layout,
                gt_layout,
                iou_threshold=layout_iou,
                categories=scoped_layout,
                ignore_outside_categories=set(scoped_layout) != set(STAGE1_CATEGORIES),
            )
            merge_stats(totals["layout"], stats)
            passed = exact_pass(stats)
            stage_pass["layout"]["total"] += 1
            stage_pass["layout"]["pass"] += int(passed)
            applicable_results.append(passed)
            image_result["layout_exact_pass"] = passed

        if gt_data["present"]["views"]:
            stats = greedy_match(
                pred_data["views"],
                gt_data["views"],
                iou_threshold=view_iou,
                categories=scoped_views,
                ignore_outside_categories=set(scoped_views) != set(VIEW_CATEGORIES),
            )
            merge_stats(totals["views"], stats)
            merge_stats(view_focus_total, stats)
            passed = exact_pass(stats)
            stage_pass["views"]["total"] += 1
            stage_pass["views"]["pass"] += int(passed)
            focus_pass["view"]["total"] += 1
            focus_pass["view"]["pass"] += int(passed)
            applicable_results.append(passed)
            image_result["view_exact_pass"] = passed
            gt_projection, pred_projection = (
                gt_data["projection_method"],
                pred_data["projection_method"],
            )
            if gt_projection is not None:
                projection["total"] += 1
                projection["correct"] += int(pred_projection == gt_projection)
                if gt_projection != "unknown":
                    projection["known_total"] += 1
                    projection["known_correct"] += int(pred_projection == gt_projection)

        if gt_data["present"]["features"]:
            scope = gt_data["feature_scope"] or list(FEATURE_CATEGORIES)
            ignore_unscoped = set(scope) != set(FEATURE_CATEGORIES)
            stats = greedy_match(
                pred_data["features"],
                gt_data["features"],
                iou_threshold=feature_iou,
                categories=scope,
                score_size=True,
                ignore_outside_categories=ignore_unscoped,
            )
            strict_stats = greedy_match(
                pred_data["features"],
                gt_data["features"],
                iou_threshold=strict_feature_iou,
                categories=scope,
                score_size=True,
                ignore_outside_categories=ignore_unscoped,
            )
            merge_stats(totals["features"], stats)
            merge_stats(totals["features_strict"], strict_stats)
            passed = exact_pass(stats, require_size=True)
            stage_pass["features"]["total"] += 1
            stage_pass["features"]["pass"] += int(passed)
            applicable_results.append(passed)
            image_result["feature_exact_pass"] = passed

            for label, categories, aggregate in (
                ("round", ROUND_FOCUS_CATEGORIES, round_total),
                ("slotted", SLOTTED_FOCUS_CATEGORIES, slotted_total),
            ):
                if not set(categories) & set(scope):
                    continue
                pred_subset, gt_subset = (
                    _subset(pred_data["features"], categories),
                    _subset(gt_data["features"], categories),
                )
                focus_stats = greedy_match(
                    pred_subset,
                    gt_subset,
                    iou_threshold=feature_iou,
                    categories=categories,
                    score_size=True,
                )
                merge_stats(aggregate, focus_stats)
                if (
                    pred_subset or gt_subset
                ):  # do not inflate exact rate with irrelevant empty images
                    focus_pass[label]["total"] += 1
                    focus_pass[label]["pass"] += int(
                        exact_pass(focus_stats, require_size=True)
                    )

        for category, threshold, pred_items, gt_items in (
            *(
                (category, view_iou, pred_data["views"], gt_data["views"])
                for category in scoped_views
                if gt_data["present"]["views"]
            ),
            *(
                (category, feature_iou, pred_data["features"], gt_data["features"])
                for category in FEATURE_CATEGORIES
                if gt_data["present"]["features"]
                and category in (gt_data["feature_scope"] or FEATURE_CATEGORIES)
            ),
        ):
            pred_subset, gt_subset = (
                _subset(pred_items, (category,)),
                _subset(gt_items, (category,)),
            )
            if not pred_subset and not gt_subset:
                continue
            cat_stats = greedy_match(
                pred_subset,
                gt_subset,
                iou_threshold=threshold,
                categories=(category,),
                score_size=category in FEATURE_CATEGORIES,
            )
            category_exact[category]["total"] += 1
            category_exact[category]["pass"] += int(
                exact_pass(cat_stats, require_size=category in FEATURE_CATEGORIES)
            )

        overall_passed = (
            bool(applicable_results)
            and all(applicable_results)
            and is_valid
            and constraint_valid
        )
        stage_pass["overall"]["total"] += 1
        stage_pass["overall"]["pass"] += int(overall_passed)
        image_result["overall_exact_pass"] = overall_passed
        per_image.append(image_result)

    layout_summary = summarize_stats(totals["layout"], STAGE1_CATEGORIES)
    view_summary = summarize_stats(totals["views"], VIEW_CATEGORIES)
    feature_summary = summarize_stats(totals["features"], FEATURE_CATEGORIES)
    strict_summary = summarize_stats(totals["features_strict"], FEATURE_CATEGORIES)
    for name, summary in (
        ("layout", layout_summary),
        ("views", view_summary),
        ("features", feature_summary),
    ):
        values = stage_pass[name]
        summary["per_image_exact_pass"] = (
            values["pass"] / values["total"] if values["total"] else 0.0
        )
        summary["evaluated_images"] = values["total"]

    round_summary = summarize_stats(round_total, ROUND_FOCUS_CATEGORIES)
    slotted_summary = summarize_stats(slotted_total, SLOTTED_FOCUS_CATEGORIES)
    focus_f1_values = []
    if focus_pass["round"]["total"]:
        focus_f1_values.append(round_summary["f1"])
    if focus_pass["slotted"]["total"]:
        focus_f1_values.append(slotted_summary["f1"])
    if focus_pass["view"]["total"]:
        focus_f1_values.append(view_summary["f1"])
    focus_exact_values = [
        values["pass"] / values["total"]
        for values in focus_pass.values()
        if values["total"]
    ]
    focus_macro_f1 = (
        sum(focus_f1_values) / len(focus_f1_values) if focus_f1_values else 0.0
    )
    focus_exact_rate = (
        sum(focus_exact_values) / len(focus_exact_values) if focus_exact_values else 0.0
    )
    valid_rate = valid_count / sample_count if sample_count else 0.0
    constraint_valid_rate = (
        constraint_valid_count / sample_count if sample_count else 0.0
    )
    focus_score = (
        0.40 * focus_exact_rate
        + 0.40 * focus_macro_f1
        + 0.10 * strict_summary["f1"]
        + 0.05 * valid_rate
        + 0.05 * constraint_valid_rate
    )
    overall_values = stage_pass["overall"]

    return {
        "samples": sample_count,
        "thresholds": {
            "layout_iou": layout_iou,
            "view_iou": view_iou,
            "feature_iou": feature_iou,
            "strict_feature_iou": strict_feature_iou,
        },
        "layout": layout_summary,
        "view": view_summary,
        "feature": feature_summary,
        "feature_strict": strict_summary,
        "projection_method": {
            **projection,
            "accuracy": projection["correct"] / projection["total"]
            if projection["total"]
            else 0.0,
            "known_accuracy": projection["known_correct"] / projection["known_total"]
            if projection["known_total"]
            else 0.0,
        },
        "json_valid_rate": valid_rate,
        "constraint_valid_rate": constraint_valid_rate,
        "constraint_violation_rate": 1.0 - constraint_valid_rate
        if sample_count
        else 0.0,
        "per_image_exact_pass": overall_values["pass"] / overall_values["total"]
        if overall_values["total"]
        else 0.0,
        "per_category_exact_pass": _category_exact_rates(category_exact),
        "focus": {
            "round_hole_f1": round_summary["f1"],
            "slotted_hole_f1": slotted_summary["f1"],
            "view_f1": view_summary["f1"],
            "focus_macro_f1": focus_macro_f1,
            "focus_exact_pass": focus_exact_rate,
            "strict_feature_f1": strict_summary["f1"],
            "json_valid_rate": valid_rate,
            "constraint_valid_rate": constraint_valid_rate,
            "weights": {
                "exact": 0.40,
                "macro_f1": 0.40,
                "strict_iou_f1": 0.10,
                "json_valid": 0.05,
                "constraint_valid": 0.05,
            },
            "FocusScore": focus_score,
        },
        "FocusScore": focus_score,
        "per_image": per_image,
    }


def serializable_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(stats))
    result["per_category"] = {
        key: dict(value) for key, value in stats.get("per_category", {}).items()
    }
    return result


__all__ = [
    "DEFAULT_LAYOUT_IOU",
    "DEFAULT_VIEW_IOU",
    "DEFAULT_FEATURE_IOU",
    "DEFAULT_STRICT_FEATURE_IOU",
    "ROUND_FOCUS_CATEGORIES",
    "SLOTTED_FOCUS_CATEGORIES",
    "dimension_tokens",
    "size_similarity",
    "size_normalized_equal",
    "empty_stats",
    "normalize_items",
    "greedy_match",
    "merge_stats",
    "exact_pass",
    "summarize_stats",
    "stages_from_flat_items",
    "coerce_three_stage",
    "extract_stage_items",
    "evaluate_pairs",
    "serializable_stats",
]
