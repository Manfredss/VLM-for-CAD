#!/usr/bin/env python3
"""Evaluate multiple LoRA adapters while keeping one 27B base model resident.

Keeping the base model on the same GPU avoids a second long model-load interval
and makes checkpoint comparisons both faster and safer on reclaim-sensitive
clusters.  Every adapter writes an independent resumable prediction JSONL and,
when ground truth is supplied, a metrics JSON report.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import infer_three_stage as core
from evaluate_predictions import _read_records as read_eval_records
from evaluate_predictions import align_records
from metrics import evaluate_pairs


LOGGER = logging.getLogger("cad_multi_adapter_inference")


def adapter_spec(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("adapter must use LABEL=PATH")
    label, path = value.split("=", 1)
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label.strip()).strip("-.")
    if not safe_label or not path.strip():
        raise argparse.ArgumentTypeError("adapter label and path must be non-empty")
    return safe_label, path.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter", action="append", type=adapter_spec, required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-mode", choices=("full", "crops", "hybrid"), default="full")
    parser.add_argument(
        "--protocol-mode",
        choices=("three_stage", "two_stage"),
        default="three_stage",
    )
    parser.add_argument("--projection-method", choices=("auto", "first_angle", "third_angle", "unknown"), default="auto")
    parser.add_argument("--feature-scope", nargs="*")
    parser.add_argument("--crop-padding", type=float, default=0.12)
    parser.add_argument("--dedup-iou", type=float, default=0.55)
    parser.add_argument("--max-crops", type=int, default=0)
    parser.add_argument("--step1-max-new-tokens", type=int, default=1536)
    parser.add_argument("--step2-max-new-tokens", type=int, default=1536)
    parser.add_argument("--layout-view-max-new-tokens", type=int, default=2048)
    parser.add_argument("--step3-max-new-tokens", type=int, default=3072)
    parser.add_argument(
        "--image-max-tokens",
        type=int,
        default=2048,
        help="Maximum merged vision tokens per image; 2048 matches 27B training",
    )
    parser.add_argument("--device-map", choices=("auto", "balanced", "single"), default="single")
    parser.add_argument("--attn-impl", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def configure_image_budget(processor: Any, image_max_tokens: int) -> int:
    """Match the Qwen VL pixel budget to the training-time vision token cap."""

    if image_max_tokens <= 0:
        raise ValueError("--image-max-tokens must be positive")
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError("processor has no image_processor")
    patch_size = int(getattr(image_processor, "patch_size", 16) or 16)
    merge_size = int(getattr(image_processor, "merge_size", 2) or 2)
    max_pixels = image_max_tokens * (patch_size * merge_size) ** 2
    size = getattr(image_processor, "size", None)
    configured = False
    if hasattr(size, "longest_edge"):
        size.longest_edge = max_pixels
        configured = True
    if isinstance(size, dict):
        size["longest_edge"] = max_pixels
        configured = True
    if not configured:
        raise ValueError(f"unsupported image processor size object: {size!r}")
    # Keep compatibility with Qwen processor variants that consult the legacy
    # attribute instead of SizeDict.longest_edge.
    image_processor.max_pixels = max_pixels
    LOGGER.info(
        "Vision budget: max_tokens=%d max_pixels=%d patch=%d merge=%d size=%s",
        image_max_tokens,
        max_pixels,
        patch_size,
        merge_size,
        image_processor.size,
    )
    return max_pixels


def collect_images(args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    proxy = argparse.Namespace(
        input_jsonl=args.input_jsonl,
        image=[],
        image_dir=None,
        recursive=False,
        limit=args.limit,
    )
    images = core.collect_images(proxy)
    return [
        item
        for index, item in enumerate(images)
        if index % args.num_shards == args.shard_index
    ]


def artifact_stem(label: str, args: argparse.Namespace) -> str:
    parts = [label]
    if args.protocol_mode != "three_stage":
        parts.append(args.protocol_mode)
    parts.append(args.feature_mode)
    if args.num_shards > 1:
        parts.append(f"shard{args.shard_index:02d}-of-{args.num_shards:02d}")
    return "_".join(parts)


def evaluate(
    ground_truth_path: Path,
    predictions_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    ground_truth, gt_errors = read_eval_records(ground_truth_path)
    predictions, pred_errors = read_eval_records(predictions_path)
    pairs, alignment = align_records(
        ground_truth, predictions, allow_order_fallback=False
    )
    report = evaluate_pairs(pairs)
    report["alignment"] = alignment
    report["parse_errors"] = {
        "ground_truth": gt_errors,
        "predictions": pred_errors,
    }
    for metrics_row, pair in zip(report.get("per_image", []), pairs):
        metrics_row["alignment_validation_errors"] = pair[
            "alignment_validation_errors"
        ]
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def infer_adapter(
    *,
    model: Any,
    processor: Any,
    label: str,
    images: Sequence[tuple[str, str, Path]],
    args: argparse.Namespace,
    feature_scope: Sequence[str],
) -> tuple[Path, int]:
    output_path = (
        args.output_dir / f"{artifact_stem(label, args)}_predictions.jsonl"
    ).resolve()
    existing: OrderedDict[str, dict[str, Any]] = (
        OrderedDict() if args.overwrite else core.load_existing(output_path)
    )
    complete_paths = {
        str(record.get("image"))
        for record in existing.values()
        if record.get("status") == "ok" and record.get("image")
    }
    pending = [item for item in images if str(item[2]) not in complete_paths]
    LOGGER.info(
        "adapter=%s images=%d complete=%d pending=%d protocol=%s mode=%s shard=%d/%d",
        label,
        len(images),
        len(images) - len(pending),
        len(pending),
        args.protocol_mode,
        args.feature_mode,
        args.shard_index,
        args.num_shards,
    )
    started = time.time()
    dirty = 0
    failures = 0
    for index, (sample_id, name, image_path) in enumerate(pending, 1):
        item_started = time.time()
        try:
            inference = core.infer_one(
                model,
                processor,
                image_path,
                protocol_mode=args.protocol_mode,
                feature_mode=args.feature_mode,
                feature_scope=feature_scope,
                crop_padding=args.crop_padding,
                dedup_iou=args.dedup_iou,
                max_crops=args.max_crops,
                layout_view_tokens=args.layout_view_max_new_tokens,
                step1_tokens=args.step1_max_new_tokens,
                step2_tokens=args.step2_max_new_tokens,
                step3_tokens=args.step3_max_new_tokens,
                save_raw=args.save_raw,
                projection_override=(
                    None if args.projection_method == "auto" else args.projection_method
                ),
            )
            record = {
                "sample_id": sample_id,
                "dataitem_name": name,
                "image": str(image_path),
                "status": "ok",
                "adapter_label": label,
                "protocol_mode": args.protocol_mode,
                "feature_mode": args.feature_mode,
                "image_max_tokens": args.image_max_tokens,
                **inference,
                "elapsed_seconds": round(time.time() - item_started, 3),
            }
        except Exception as exc:
            failures += 1
            LOGGER.exception("adapter=%s inference failed: %s", label, image_path)
            record = {
                "sample_id": sample_id,
                "dataitem_name": name,
                "image": str(image_path),
                "status": "error",
                "adapter_label": label,
                "protocol_mode": args.protocol_mode,
                "feature_mode": args.feature_mode,
                "image_max_tokens": args.image_max_tokens,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.time() - item_started, 3),
            }
            if args.fail_fast:
                existing[str(image_path)] = record
                core.atomic_save(output_path, existing.values())
                raise
        existing[str(image_path)] = record
        dirty += 1
        if dirty >= args.save_every:
            core.atomic_save(output_path, existing.values())
            dirty = 0
        average = (time.time() - started) / index
        eta = average * (len(pending) - index)
        LOGGER.info(
            "adapter=%s [%d/%d] status=%s elapsed=%.1fs ETA=%.1fmin",
            label,
            index,
            len(pending),
            record["status"],
            record["elapsed_seconds"],
            eta / 60,
        )
    if dirty or not output_path.exists():
        core.atomic_save(output_path, existing.values())
    return output_path, failures


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = collect_images(args)
    if not images:
        raise ValueError("no accessible images found")
    feature_scope = core.parse_scope(args.feature_scope)

    first_label, first_path = args.adapter[0]
    model, processor = core.load_model(
        args.model_path,
        first_path,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
        attn_impl=args.attn_impl,
    )
    configure_image_budget(processor, args.image_max_tokens)
    total_failures = 0
    for adapter_index, (label, path) in enumerate(args.adapter):
        if adapter_index:
            LOGGER.info("Loading adapter=%s path=%s into resident base model", label, path)
            model.load_adapter(path, adapter_name=label)
            model.set_adapter(label)
            model.eval()
        output_path, failures = infer_adapter(
            model=model,
            processor=processor,
            label=label,
            images=images,
            args=args,
            feature_scope=feature_scope,
        )
        total_failures += failures
        if args.ground_truth:
            metrics_path = (
                args.output_dir / f"{artifact_stem(label, args)}_metrics.json"
            )
            report = evaluate(args.ground_truth, output_path, metrics_path)
            LOGGER.info(
                "adapter=%s FocusScore=%.6f feature_macro_f1=%.6f json_valid=%.6f constraints=%.6f report=%s",
                label,
                report["FocusScore"],
                report["feature"]["macro_f1"],
                report["json_valid_rate"],
                report["constraint_valid_rate"],
                metrics_path,
            )
    return 1 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
