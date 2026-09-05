#!/usr/bin/env python3
"""Cached three-stage inference with separate layout/view and feature LoRAs.

The layout/view adapter generates Stage 1 and Stage 2 once per image.  Each
feature adapter then receives the same image plus the normalized predicted
Stage 1/2 objects in the single-turn format used by the P3-R3 feature SFT.
This validates real upstream-error propagation without re-running layout/view
generation for every checkpoint pair.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import infer_adapters as multi
import infer_three_stage as core
from cad_schema import (
    DOCUMENT_CATEGORIES,
    SYSTEM_PROMPT,
    validate_cross_stage_outputs,
    validate_stage_output,
)


LOGGER = logging.getLogger("cad_dual_adapter_inference")


def adapter_spec(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("adapter must use LABEL=PATH")
    label, path = value.split("=", 1)
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label.strip()).strip("-.")
    if not label or not path.strip():
        raise argparse.ArgumentTypeError("adapter label and path must be non-empty")
    return label, path.strip()


def combination_spec(value: str) -> tuple[str, str]:
    """Parse an explicit layout:feature checkpoint pair."""
    if ":" not in value:
        raise argparse.ArgumentTypeError("combination must use LAYOUT:FEATURE")
    layout, feature = (part.strip() for part in value.split(":", 1))
    if not layout or not feature:
        raise argparse.ArgumentTypeError(
            "combination layout and feature labels must be non-empty"
        )
    return layout, feature


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--layout-adapter", action="append", type=adapter_spec, required=True
    )
    parser.add_argument(
        "--feature-adapter", action="append", type=adapter_spec, required=True
    )
    parser.add_argument(
        "--combination",
        action="append",
        type=combination_spec,
        help=(
            "Evaluate only this LAYOUT:FEATURE pair (repeatable). "
            "The default is the full Cartesian product."
        ),
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-max-tokens", type=int, default=2048)
    parser.add_argument("--step1-max-new-tokens", type=int, default=1536)
    parser.add_argument("--step2-max-new-tokens", type=int, default=1536)
    parser.add_argument("--step3-max-new-tokens", type=int, default=3072)
    parser.add_argument("--feature-scope", nargs="*")
    parser.add_argument("--projection-method", choices=("auto", "first_angle", "third_angle", "unknown"), default="auto")
    parser.add_argument("--device-map", choices=("auto", "balanced", "single"), default="single")
    parser.add_argument("--attn-impl", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def collect_images(args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    proxy = argparse.Namespace(
        input_jsonl=args.input_jsonl,
        image=[],
        image_dir=None,
        recursive=False,
        limit=args.limit,
    )
    return core.collect_images(proxy)


def load_records(path: Path) -> OrderedDict[str, dict[str, Any]]:
    return OrderedDict() if not path.exists() else core.load_existing(path)


def activate(model: Any, adapter_name: str) -> None:
    model.set_adapter(adapter_name)
    model.eval()
    LOGGER.info("Activated adapter=%s", adapter_name)


def infer_layout_view(
    model: Any,
    processor: Any,
    image_path: Path,
    *,
    step1_tokens: int,
    step2_tokens: int,
    projection_override: str | None,
    save_raw: bool,
) -> dict[str, Any]:
    image_uri = image_path.resolve().as_uri()
    warnings: list[str] = []
    timings: dict[str, float] = {}
    raw_valid: dict[str, bool] = {}
    raw_outputs: dict[str, str] = {}

    messages1 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        core._image_message(image_uri, core.step1_prompt()),
    ]
    started = time.monotonic()
    raw1 = core.generate(model, processor, messages1, max_new_tokens=step1_tokens)
    timings["step1"] = round(time.monotonic() - started, 3)
    raw_valid["step1"] = core.raw_stage_json_valid(raw1, 1)
    parsed1, parse_errors = core.parse_stage_json(raw1, 1)
    stage1, normal_errors = core.normalize_step1(parsed1)
    warnings.extend(parse_errors + normal_errors)
    raw_outputs["step1"] = raw1

    messages2 = messages1 + [
        {"role": "assistant", "content": core._json_text(stage1)},
        {"role": "user", "content": core.step2_prompt()},
    ]
    started = time.monotonic()
    raw2 = core.generate(model, processor, messages2, max_new_tokens=step2_tokens)
    timings["step2"] = round(time.monotonic() - started, 3)
    raw_valid["step2"] = core.raw_stage_json_valid(raw2, 2)
    parsed2, parse_errors = core.parse_stage_json(raw2, 2)
    stage2, normal_errors = core.normalize_step2(parsed2, stage1)
    warnings.extend(parse_errors + normal_errors)
    raw_outputs["step2"] = raw2
    if projection_override is not None:
        stage2["projection_method"] = projection_override

    result: dict[str, Any] = {
        "step1": stage1,
        "step2": stage2,
        "warnings": warnings,
        "timings_seconds": timings,
        "raw_json_valid": raw_valid,
    }
    if save_raw:
        result["raw_outputs"] = raw_outputs
    return result


def feature_prompt(
    feature_scope: Sequence[str], stage1: Mapping[str, Any], stage2: Mapping[str, Any]
) -> str:
    # Keep this text byte-for-byte aligned with feature_row() in the P3-R3
    # dataset builder.  The only difference is that context is predicted here.
    context = (
        "\n\n上游阶段参考结果（只用于关联视图区；feature_scope 和特征标签仍以当前任务为准）："
        f"\nstage1={core._json_text(stage1)}"
        f"\nstage2={core._json_text(stage2)}"
    )
    return core.step3_prompt(feature_scope, crop_context=False) + context


def legacy_result(
    stage1: Mapping[str, Any],
    stage2: Mapping[str, Any],
    stage3: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for region in stage1.get("regions", []):
        if region.get("category") in DOCUMENT_CATEGORIES:
            output.append(
                {"category": region["category"], "size": "", "bbox": region["bbox"]}
            )
    for view in stage2.get("views", []):
        output.append(
            {"category": view["category"], "size": "", "bbox": view["bbox"]}
        )
    for feature in stage3.get("features", []):
        output.append(
            {
                "category": feature["category"],
                "size": feature.get("size", ""),
                "bbox": feature["bbox"],
            }
        )
    return output


def infer_feature(
    model: Any,
    processor: Any,
    image_path: Path,
    layout: Mapping[str, Any],
    *,
    feature_scope: Sequence[str],
    step3_tokens: int,
    save_raw: bool,
) -> dict[str, Any]:
    stage1 = copy.deepcopy(layout["step1"])
    stage2 = copy.deepcopy(layout["step2"])
    image_uri = image_path.resolve().as_uri()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        core._image_message(image_uri, feature_prompt(feature_scope, stage1, stage2)),
    ]
    started = time.monotonic()
    raw3 = core.generate(model, processor, messages, max_new_tokens=step3_tokens)
    step3_seconds = round(time.monotonic() - started, 3)
    parsed3, parse_errors = core.parse_stage_json(raw3, 3)
    stage3, normal_errors = core.normalize_step3(parsed3, feature_scope)
    warnings = list(layout.get("warnings", [])) + parse_errors + normal_errors

    raw_valid = dict(layout.get("raw_json_valid", {}))
    raw_valid["step3_full"] = core.raw_stage_json_valid(raw3, 3)
    raw_valid["overall"] = bool(raw_valid) and all(raw_valid.values())
    timings = dict(layout.get("timings_seconds", {}))
    timings["step3_full"] = step3_seconds
    constraint_errors = validate_cross_stage_outputs(stage1, stage2, stage3)
    warnings.extend(f"cross-stage constraint: {error}" for error in constraint_errors)
    warnings.extend(
        f"final step3 schema: {error}"
        for error in validate_stage_output(3, stage3, strict=True)
    )
    diagnostics = {
        "protocol_mode": "three_stage_dual_adapter",
        "warnings": warnings,
        "raw_json_valid": raw_valid,
        "timings_seconds": timings,
        "feature_candidates": len(stage3.get("features", [])),
        "feature_count": len(stage3.get("features", [])),
        "size_conflicts": [],
        "crop_count": 0,
        "constraint_errors": constraint_errors,
        "constraint_valid": not constraint_errors,
    }
    result: dict[str, Any] = {
        "step1": stage1,
        "step2": stage2,
        "step3": stage3,
        "result": legacy_result(stage1, stage2, stage3),
        "diagnostics": diagnostics,
    }
    if save_raw:
        raw_outputs = dict(layout.get("raw_outputs", {}))
        raw_outputs["step3_full"] = raw3
        result["raw_outputs"] = raw_outputs
    return result


def generate_layout_cache(
    *,
    model: Any,
    processor: Any,
    adapter_label: str,
    adapter_name: str,
    images: Sequence[tuple[str, str, Path]],
    args: argparse.Namespace,
) -> Path:
    activate(model, adapter_name)
    path = args.output_dir / "layout_cache" / f"{adapter_label}_predictions.jsonl"
    records = OrderedDict() if args.overwrite else load_records(path)
    complete = {
        str(row.get("image"))
        for row in records.values()
        if row.get("status") == "ok" and row.get("image")
    }
    dirty = 0
    pending = [item for item in images if str(item[2]) not in complete]
    for index, (sample_id, name, image_path) in enumerate(pending, 1):
        started = time.time()
        try:
            inferred = infer_layout_view(
                model,
                processor,
                image_path,
                step1_tokens=args.step1_max_new_tokens,
                step2_tokens=args.step2_max_new_tokens,
                projection_override=(
                    None if args.projection_method == "auto" else args.projection_method
                ),
                save_raw=args.save_raw,
            )
            row = {
                "sample_id": sample_id,
                "dataitem_name": name,
                "image": str(image_path),
                "status": "ok",
                "layout_adapter": adapter_label,
                **inferred,
                "elapsed_seconds": round(time.time() - started, 3),
            }
        except Exception as exc:
            LOGGER.exception("layout adapter=%s failed: %s", adapter_label, image_path)
            row = {
                "sample_id": sample_id,
                "dataitem_name": name,
                "image": str(image_path),
                "status": "error",
                "layout_adapter": adapter_label,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.time() - started, 3),
            }
            if args.fail_fast:
                records[str(image_path)] = row
                core.atomic_save(path, records.values())
                raise
        records[str(image_path)] = row
        dirty += 1
        if dirty >= args.save_every:
            core.atomic_save(path, records.values())
            dirty = 0
        LOGGER.info(
            "layout=%s [%d/%d] status=%s elapsed=%.1fs",
            adapter_label,
            index,
            len(pending),
            row["status"],
            row["elapsed_seconds"],
        )
    if dirty or not path.exists():
        core.atomic_save(path, records.values())
    return path


def generate_combination(
    *,
    model: Any,
    processor: Any,
    layout_label: str,
    layout_path: Path,
    feature_label: str,
    feature_name: str,
    images: Sequence[tuple[str, str, Path]],
    feature_scope: Sequence[str],
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any]]:
    activate(model, feature_name)
    combo = f"{layout_label}__{feature_label}"
    output = args.output_dir / f"{combo}_predictions.jsonl"
    records = OrderedDict() if args.overwrite else load_records(output)
    complete = {
        str(row.get("image"))
        for row in records.values()
        if row.get("status") == "ok" and row.get("image")
    }
    layouts = {
        str(row.get("image")): row
        for row in core._read_records(layout_path)
        if row.get("status") == "ok" and row.get("image")
    }
    dirty = 0
    pending = [item for item in images if str(item[2]) not in complete]
    for index, (sample_id, name, image_path) in enumerate(pending, 1):
        started = time.time()
        layout = layouts.get(str(image_path))
        if layout is None:
            raise RuntimeError(f"missing successful layout cache for {image_path}")
        try:
            inferred = infer_feature(
                model,
                processor,
                image_path,
                layout,
                feature_scope=feature_scope,
                step3_tokens=args.step3_max_new_tokens,
                save_raw=args.save_raw,
            )
            row = {
                "sample_id": sample_id,
                "dataitem_name": name,
                "image": str(image_path),
                "status": "ok",
                "layout_adapter": layout_label,
                "feature_adapter": feature_label,
                "protocol_mode": "three_stage_dual_adapter",
                "feature_mode": "full",
                "image_max_tokens": args.image_max_tokens,
                **inferred,
                "elapsed_seconds": round(
                    float(layout.get("elapsed_seconds", 0.0))
                    + time.time()
                    - started,
                    3,
                ),
            }
        except Exception as exc:
            LOGGER.exception("combo=%s failed: %s", combo, image_path)
            row = {
                "sample_id": sample_id,
                "dataitem_name": name,
                "image": str(image_path),
                "status": "error",
                "layout_adapter": layout_label,
                "feature_adapter": feature_label,
                "error": f"{type(exc).__name__}: {exc}",
            }
            if args.fail_fast:
                records[str(image_path)] = row
                core.atomic_save(output, records.values())
                raise
        records[str(image_path)] = row
        dirty += 1
        if dirty >= args.save_every:
            core.atomic_save(output, records.values())
            dirty = 0
        LOGGER.info(
            "combo=%s [%d/%d] status=%s elapsed=%.1fs",
            combo,
            index,
            len(pending),
            row["status"],
            row.get("elapsed_seconds", 0.0),
        )
    if dirty or not output.exists():
        core.atomic_save(output, records.values())
    metrics_path = args.output_dir / f"{combo}_metrics.json"
    metrics = multi.evaluate(args.ground_truth, output, metrics_path)
    elapsed = [
        float(row.get("elapsed_seconds", 0.0))
        for row in records.values()
        if row.get("status") == "ok"
    ]
    metrics["mean_elapsed_seconds"] = sum(elapsed) / len(elapsed) if elapsed else 0.0
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output, metrics


def compact_metrics(label: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "samples": metrics.get("samples"),
        "FocusScore": metrics["FocusScore"],
        "layout_macro_f1": metrics["layout"]["macro_f1"],
        "view_f1": metrics["view"]["f1"],
        "view_macro_f1": metrics["view"]["macro_f1"],
        "feature_f1": metrics["feature"]["f1"],
        "feature_macro_f1": metrics["feature"]["macro_f1"],
        "strict_feature_f1": metrics["feature_strict"]["f1"],
        "round_hole_f1": metrics["focus"]["round_hole_f1"],
        "slotted_hole_f1": metrics["focus"]["slotted_hole_f1"],
        "json_valid_rate": metrics["json_valid_rate"],
        "constraint_valid_rate": metrics["constraint_valid_rate"],
        "mean_elapsed_seconds": float(metrics.get("mean_elapsed_seconds", 0.0)),
    }


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = collect_images(args)
    if not images:
        raise ValueError("no accessible images found")
    feature_scope = core.parse_scope(args.feature_scope)

    first_layout_label, first_layout_path = args.layout_adapter[0]
    model, processor = core.load_model(
        args.model_path,
        first_layout_path,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
        attn_impl=args.attn_impl,
    )
    multi.configure_image_budget(processor, args.image_max_tokens)
    adapter_names = {first_layout_label: "default"}
    for label, path in args.layout_adapter[1:] + args.feature_adapter:
        if label in adapter_names:
            raise ValueError(f"duplicate adapter label: {label}")
        LOGGER.info("Loading adapter=%s path=%s", label, path)
        model.load_adapter(path, adapter_name=label)
        adapter_names[label] = label

    layout_labels = {label for label, _ in args.layout_adapter}
    feature_labels = {label for label, _ in args.feature_adapter}
    if args.combination:
        combinations = list(dict.fromkeys(args.combination))
        for layout_label, feature_label in combinations:
            if layout_label not in layout_labels:
                raise ValueError(
                    f"unknown layout label in --combination: {layout_label}"
                )
            if feature_label not in feature_labels:
                raise ValueError(
                    f"unknown feature label in --combination: {feature_label}"
                )
    else:
        combinations = [
            (layout_label, feature_label)
            for layout_label, _ in args.layout_adapter
            for feature_label, _ in args.feature_adapter
        ]

    layout_paths: dict[str, Path] = {}
    for label, _ in args.layout_adapter:
        layout_paths[label] = generate_layout_cache(
            model=model,
            processor=processor,
            adapter_label=label,
            adapter_name=adapter_names[label],
            images=images,
            args=args,
        )

    ranking = []
    for layout_label, feature_label in combinations:
        _, metrics = generate_combination(
            model=model,
            processor=processor,
            layout_label=layout_label,
            layout_path=layout_paths[layout_label],
            feature_label=feature_label,
            feature_name=adapter_names[feature_label],
            images=images,
            feature_scope=feature_scope,
            args=args,
        )
        row = compact_metrics(f"{layout_label}__{feature_label}", metrics)
        ranking.append(row)
        LOGGER.info(
            "combo=%s FocusScore=%.6f view_macro=%.6f feature_macro=%.6f",
            row["label"],
            row["FocusScore"],
            row["view_macro_f1"],
            row["feature_macro_f1"],
        )
    ranking.sort(
        key=lambda row: (row["FocusScore"], row["feature_macro_f1"]), reverse=True
    )
    report = {
        "selection_metric": "generation_FocusScore",
        "input": str(Path(args.input_jsonl).resolve()),
        "ground_truth": str(args.ground_truth.resolve()),
        "image_max_tokens": args.image_max_tokens,
        "combinations": [f"{layout}:{feature}" for layout, feature in combinations],
        "ranking": ranking,
        "winner": ranking[0],
    }
    path = args.output_dir / "dual_adapter_ranking.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
