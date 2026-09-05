#!/usr/bin/env python3
"""Evaluate three-stage inference JSONL against a held-out three-stage JSONL.

Expected ground-truth rows contain canonical ``stage1``/``stage2``/``stage3``
objects.  Training-style rows containing assistant JSON in ``messages`` are
also supported.  Inference rows may use the canonical fields or the format
written by ``infer_three_stage.py``::

    {"sample_id": "...", "image": "...", "status": "ok",
     "step1": {...}, "step2": {...}, "step3": {...}}

The ground-truth file must carry labels (direct stages or assistant messages);
an image-only manifest is not sufficient by itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

try:
    from .cad_schema import validate_stage_output
    from .metrics import (
        DEFAULT_FEATURE_IOU,
        DEFAULT_LAYOUT_IOU,
        DEFAULT_STRICT_FEATURE_IOU,
        DEFAULT_VIEW_IOU,
        coerce_three_stage,
        evaluate_pairs,
    )
except ImportError:
    from cad_schema import validate_stage_output  # type: ignore
    from metrics import (  # type: ignore
        DEFAULT_FEATURE_IOU,
        DEFAULT_LAYOUT_IOU,
        DEFAULT_STRICT_FEATURE_IOU,
        DEFAULT_VIEW_IOU,
        coerce_three_stage,
        evaluate_pairs,
    )


def _read_records(path: Path) -> tuple[list[Any], list[str]]:
    errors: list[str] = []
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return [], [f"{path}: empty file"]
    if path.suffix.casefold() == ".jsonl":
        records = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_number}: {exc.msg}")
        return records, errors
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return [], [f"{path}:{exc.lineno}: {exc.msg}"]
    if isinstance(payload, list):
        return payload, errors
    if isinstance(payload, Mapping):
        for key in ("results", "predictions", "data", "items", "samples"):
            if isinstance(payload.get(key), list):
                return list(payload[key]), errors
        return [dict(payload)], errors
    return [], [f"{path}: top-level JSON must be an object or array"]


def _first_image(record: Mapping[str, Any]) -> str:
    images = record.get("images")
    if isinstance(images, list) and images:
        return str(images[0])
    for key in ("image", "image_path", "url"):
        if record.get(key):
            return str(record[key])
    return ""


def record_keys(record: Any) -> list[str]:
    """Return ordered join keys, preferring globally unique sample IDs."""
    if not isinstance(record, Mapping):
        return []
    keys = []
    metadata = (
        record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    )
    for value in (
        record.get("sample_id"),
        record.get("dataitem_name"),
        record.get("id"),
        metadata.get("sample_id"),
        metadata.get("dataitem_name"),
    ):
        text = str(value or "").strip()
        if text and text not in keys:
            keys.append(text)
    image = _first_image(record)
    if image:
        for value in (image, Path(image).name):
            if value and value not in keys:
                keys.append(value)
    return keys


def _primary_id(record: Any, fallback: str) -> str:
    keys = record_keys(record)
    return keys[0] if keys else fallback


def _index_predictions(records: Iterable[Any]) -> tuple[dict[str, int], list[str]]:
    index: dict[str, int] = {}
    duplicates = []
    for row_index, record in enumerate(records):
        for key in record_keys(record):
            if key in index and index[key] != row_index:
                duplicates.append(key)
            else:
                index[key] = row_index
    return index, sorted(set(duplicates))


def _prediction_valid(
    prediction: Any, ground_truth: Any
) -> tuple[bool, dict[str, list[str]]]:
    pred_stages, gt_stages = (
        coerce_three_stage(prediction),
        coerce_three_stage(ground_truth),
    )
    errors: dict[str, list[str]] = {}
    if isinstance(prediction, Mapping):
        status = str(prediction.get("status", "ok")).strip().casefold()
        if status not in {"", "ok", "success", "completed", "complete"}:
            errors["status"] = [f"inference status={prediction.get('status')!r}"]
        diagnostics = prediction.get("diagnostics")
        raw_valid = (
            diagnostics.get("raw_json_valid")
            if isinstance(diagnostics, Mapping)
            else None
        )
        if isinstance(raw_valid, Mapping) and raw_valid.get("overall") is False:
            errors["raw_json_valid"] = [
                "one or more original model responses were invalid before repair/normalization"
            ]
    for number in (1, 2, 3):
        gt_payload, pred_payload = (
            gt_stages.get(f"stage{number}"),
            pred_stages.get(f"stage{number}"),
        )
        if gt_payload is None:
            continue
        if pred_payload is None:
            errors[f"stage{number}"] = ["missing or unparsable output"]
            continue
        stage_errors = validate_stage_output(number, pred_payload, strict=True)
        if stage_errors:
            errors[f"stage{number}"] = stage_errors
    return not errors, errors


def align_records(
    ground_truth: list[Any],
    predictions: list[Any],
    *,
    allow_order_fallback: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prediction_index, duplicate_keys = _index_predictions(predictions)
    used: set[int] = set()
    pairs = []
    missing = []
    order_fallbacks = 0

    for gt_index, gt_record in enumerate(ground_truth):
        pred_index: Optional[int] = None
        for key in record_keys(gt_record):
            candidate = prediction_index.get(key)
            if candidate is not None and candidate not in used:
                pred_index = candidate
                break
        if (
            pred_index is None
            and allow_order_fallback
            and gt_index < len(predictions)
            and gt_index not in used
        ):
            pred_index = gt_index
            order_fallbacks += 1
        sample_id = _primary_id(gt_record, str(gt_index))
        if pred_index is None:
            prediction = {}
            missing.append(sample_id)
            valid, validity_errors = False, {"alignment": ["missing prediction"]}
        else:
            prediction = predictions[pred_index]
            used.add(pred_index)
            valid, validity_errors = _prediction_valid(prediction, gt_record)
        pairs.append(
            {
                "sample_id": sample_id,
                "prediction": prediction,
                "ground_truth": gt_record,
                "json_valid": valid,
                "alignment_validation_errors": validity_errors,
            }
        )

    extras = [
        _primary_id(record, str(index))
        for index, record in enumerate(predictions)
        if index not in used
    ]
    return pairs, {
        "ground_truth_rows": len(ground_truth),
        "prediction_rows": len(predictions),
        "matched_rows": len(ground_truth) - len(missing),
        "missing_predictions": missing,
        "extra_predictions": extras,
        "duplicate_prediction_keys": duplicate_keys,
        "order_fallbacks": order_fallbacks,
    }


def _print_summary(report: Mapping[str, Any]) -> None:
    alignment = report["alignment"]
    print(
        f"Samples: {report['samples']}  matched={alignment['matched_rows']}  missing={len(alignment['missing_predictions'])}  extra={len(alignment['extra_predictions'])}"
    )
    for label, key in (
        ("Layout", "layout"),
        ("View", "view"),
        ("Feature", "feature"),
        ("Feature@strict", "feature_strict"),
    ):
        values = report[key]
        exact = values.get("per_image_exact_pass")
        exact_text = f" exact={exact:.4f}" if exact is not None else ""
        print(
            f"{label:14s} P={values['precision']:.4f} R={values['recall']:.4f} F1={values['f1']:.4f} macro={values['macro_f1']:.4f}{exact_text}"
        )
    print(
        f"Size(normalized): recall-aware={report['feature']['size_normalized_accuracy']:.4f} matched={report['feature']['size_normalized_accuracy_matched']:.4f}"
    )
    print(
        "Size hallucination on empty GT: "
        f"{report['feature']['empty_gt_size_hallucination_rate']:.4f} "
        f"({report['feature']['size_empty_gt_hallucinated']}/"
        f"{report['feature']['size_empty_gt_matched_total']})"
    )
    print(f"JSON valid rate: {report['json_valid_rate']:.4f}")
    print(f"Cross-stage constraint valid rate: {report['constraint_valid_rate']:.4f}")
    print(f"Per-image exact pass: {report['per_image_exact_pass']:.4f}")
    print(f"FocusScore: {report['FocusScore']:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ground-truth",
        "--test",
        dest="ground_truth",
        type=Path,
        required=True,
        help="Labeled three-stage JSONL/JSON. Image-only rows cannot be evaluated.",
    )
    parser.add_argument(
        "--predictions",
        "--infer",
        dest="predictions",
        type=Path,
        required=True,
        help="Inference JSONL/JSON with step1/step2/step3 (or stage1/stage2/stage3).",
    )
    parser.add_argument(
        "--output", type=Path, help="Write the full metrics JSON report."
    )
    parser.add_argument("--layout-iou", type=float, default=DEFAULT_LAYOUT_IOU)
    parser.add_argument("--view-iou", type=float, default=DEFAULT_VIEW_IOU)
    parser.add_argument("--feature-iou", type=float, default=DEFAULT_FEATURE_IOU)
    parser.add_argument(
        "--strict-feature-iou", type=float, default=DEFAULT_STRICT_FEATURE_IOU
    )
    parser.add_argument(
        "--allow-order-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicitly allow row-order matching when no sample/image key matches (default: false).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on parse/alignment errors or invalid prediction JSON.",
    )
    parser.add_argument(
        "--no-per-image",
        action="store_true",
        help="Omit per-image details from --output.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ground_truth, gt_parse_errors = _read_records(args.ground_truth)
        predictions, pred_parse_errors = _read_records(args.predictions)
    except (OSError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not ground_truth:
        print("ERROR: no readable ground-truth rows", file=sys.stderr)
        return 2

    pairs, alignment = align_records(
        ground_truth, predictions, allow_order_fallback=args.allow_order_fallback
    )
    report = evaluate_pairs(
        pairs,
        layout_iou=args.layout_iou,
        view_iou=args.view_iou,
        feature_iou=args.feature_iou,
        strict_feature_iou=args.strict_feature_iou,
    )
    report["alignment"] = alignment
    report["parse_errors"] = {
        "ground_truth": gt_parse_errors,
        "predictions": pred_parse_errors,
    }
    # Keep the stricter, alignment-aware validation diagnostics produced here.
    for metrics_row, pair in zip(report.get("per_image", []), pairs):
        metrics_row["alignment_validation_errors"] = pair["alignment_validation_errors"]
    if args.no_per_image:
        report.pop("per_image", None)

    _print_summary(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Report: {args.output}")

    failed = bool(
        gt_parse_errors
        or pred_parse_errors
        or alignment["missing_predictions"]
        or alignment["extra_predictions"]
        or alignment["duplicate_prediction_keys"]
        or report["json_valid_rate"] < 1.0
        or report["constraint_valid_rate"] < 1.0
    )
    return 1 if args.strict and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
