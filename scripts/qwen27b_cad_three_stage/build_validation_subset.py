#!/usr/bin/env python3
"""Build a deterministic, focus-aware subset for generation validation.

This utility keeps inference rows and ground-truth rows aligned by stable IDs or
image paths.  It greedily covers rare view labels and the configured feature
labels, then adds hard negatives and diverse filler rows.  It is intended for
pipeline smoke tests only; production checkpoint selection must still use the
complete validation split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .cad_schema import VIEW_CATEGORIES, normalize_category
    from .metrics import coerce_three_stage
except ImportError:
    from cad_schema import VIEW_CATEGORIES, normalize_category  # type: ignore
    from metrics import coerce_three_stage  # type: ignore


DEFAULT_FOCUS_FEATURES = (
    "Round Hole",
    "Round Hole Group",
    "Slotted Hole",
    "Slotted Hole Group",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(dict(value))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def row_keys(row: Mapping[str, Any]) -> list[str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    keys: list[str] = []
    for value in (
        row.get("sample_id"),
        row.get("dataitem_name"),
        row.get("id"),
        metadata.get("sample_id"),
        metadata.get("dataitem_name"),
    ):
        text = str(value or "").strip()
        if text and text not in keys:
            keys.append(text)
    images = row.get("images") if isinstance(row.get("images"), list) else []
    for value in (*images, row.get("image"), row.get("image_path")):
        text = str(value or "").strip()
        if not text:
            continue
        for key in (text, Path(text).name):
            if key and key not in keys:
                keys.append(key)
    return keys


def align_rows(
    inference_rows: Sequence[dict[str, Any]],
    ground_truth_rows: Sequence[dict[str, Any]],
) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    index: dict[str, int] = {}
    duplicates: set[str] = set()
    for position, row in enumerate(inference_rows):
        for key in row_keys(row):
            if key in index and index[key] != position:
                duplicates.add(key)
            else:
                index[key] = position
    if duplicates:
        raise ValueError(f"duplicate inference join keys: {sorted(duplicates)[:10]}")

    used: set[int] = set()
    aligned: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    missing: list[str] = []
    for gt_position, gt_row in enumerate(ground_truth_rows):
        match = next(
            (
                index[key]
                for key in row_keys(gt_row)
                if key in index and index[key] not in used
            ),
            None,
        )
        if match is None:
            missing.append(row_keys(gt_row)[0] if row_keys(gt_row) else str(gt_position))
            continue
        used.add(match)
        aligned.append((gt_position, inference_rows[match], gt_row))
    if missing or len(used) != len(inference_rows):
        raise ValueError(
            f"split alignment failed: missing={len(missing)} "
            f"unused_inference={len(inference_rows) - len(used)}"
        )
    return aligned


def label_tokens(row: Mapping[str, Any]) -> set[str]:
    stages = coerce_three_stage(row)
    tokens: set[str] = set()
    stage2 = stages.get("stage2")
    if isinstance(stage2, Mapping):
        for item in stage2.get("views", []):
            if not isinstance(item, Mapping):
                continue
            category = normalize_category(item.get("category"), kind="view")
            if category:
                tokens.add(f"view:{category}")
    stage3 = stages.get("stage3")
    if isinstance(stage3, Mapping):
        for item in stage3.get("features", []):
            if not isinstance(item, Mapping):
                continue
            category = normalize_category(item.get("category"), kind="feature")
            if category:
                tokens.add(f"feature:{category}")
    return tokens


def stable_tiebreak(seed: int, sample_key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def build_subset(
    aligned: Sequence[tuple[int, dict[str, Any], dict[str, Any]]],
    *,
    max_samples: int,
    min_positive_per_target: int,
    negative_samples: int,
    focus_features: Sequence[str],
    seed: int,
) -> tuple[list[tuple[int, dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if min_positive_per_target <= 0:
        raise ValueError("min_positive_per_target must be positive")
    labels = [label_tokens(gt_row) for _, _, gt_row in aligned]
    focus_tokens = {f"feature:{category}" for category in focus_features}
    target_tokens = focus_tokens | {f"view:{category}" for category in VIEW_CATEGORIES}
    availability = Counter(token for row_labels in labels for token in row_labels)
    quotas = {
        token: min(min_positive_per_target, availability[token])
        for token in sorted(target_tokens)
        if availability[token]
    }

    selected: list[int] = []
    selected_set: set[int] = set()
    covered = Counter()
    while len(selected) < max_samples:
        deficits = {token for token, quota in quotas.items() if covered[token] < quota}
        if not deficits:
            break
        candidates: list[tuple[float, int, int]] = []
        for index, row_labels in enumerate(labels):
            if index in selected_set:
                continue
            gain_tokens = row_labels & deficits
            if not gain_tokens:
                continue
            rarity_gain = sum(1.0 / availability[token] for token in gain_tokens)
            score = 1000.0 * len(gain_tokens) + rarity_gain
            key = row_keys(aligned[index][2])
            candidates.append(
                (score, -stable_tiebreak(seed, key[0] if key else str(index)), index)
            )
        if not candidates:
            break
        index = max(candidates)[2]
        selected.append(index)
        selected_set.add(index)
        covered.update(labels[index] & quotas.keys())

    negative_candidates = []
    for index, row_labels in enumerate(labels):
        if index in selected_set or row_labels & focus_tokens:
            continue
        view_diversity = len({token for token in row_labels if token.startswith("view:")})
        key = row_keys(aligned[index][2])
        negative_candidates.append(
            (
                view_diversity,
                -stable_tiebreak(seed + 1, key[0] if key else str(index)),
                index,
            )
        )
    for _, _, index in sorted(negative_candidates, reverse=True)[:negative_samples]:
        if len(selected) >= max_samples:
            break
        selected.append(index)
        selected_set.add(index)

    filler = []
    for index, row_labels in enumerate(labels):
        if index in selected_set:
            continue
        target_diversity = len(row_labels & target_tokens)
        all_diversity = len(row_labels)
        key = row_keys(aligned[index][2])
        filler.append(
            (
                target_diversity,
                all_diversity,
                -stable_tiebreak(seed + 2, key[0] if key else str(index)),
                index,
            )
        )
    for *_, index in sorted(filler, reverse=True):
        if len(selected) >= min(max_samples, len(aligned)):
            break
        selected.append(index)
        selected_set.add(index)

    # Preserve source order in the written files to keep inspection intuitive.
    selected_aligned = sorted((aligned[index] for index in selected), key=lambda item: item[0])
    selected_counts = Counter(
        token for index in selected for token in labels[index] if token in target_tokens
    )
    summary = {
        "seed": seed,
        "source_samples": len(aligned),
        "selected_samples": len(selected_aligned),
        "requested_max_samples": max_samples,
        "min_positive_per_target": min_positive_per_target,
        "requested_negative_samples": negative_samples,
        "focus_features": list(focus_features),
        "target_availability": dict(sorted(availability.items())),
        "target_selected_image_counts": dict(sorted(selected_counts.items())),
        "unmet_quotas": {
            token: quota - selected_counts[token]
            for token, quota in quotas.items()
            if selected_counts[token] < quota
        },
        "sample_ids": [
            (row_keys(gt_row)[0] if row_keys(gt_row) else str(position))
            for position, _, gt_row in selected_aligned
        ],
    }
    return selected_aligned, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-inference", type=Path, required=True)
    parser.add_argument("--output-ground-truth", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--min-positive-per-target", type=int, default=3)
    parser.add_argument("--negative-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--focus-feature",
        action="append",
        dest="focus_features",
        help="May be repeated; defaults to Round/Slotted Hole and Group labels",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inference_rows = read_jsonl(args.inference)
    ground_truth_rows = read_jsonl(args.ground_truth)
    aligned = align_rows(inference_rows, ground_truth_rows)
    focus_features = tuple(args.focus_features or DEFAULT_FOCUS_FEATURES)
    selected, summary = build_subset(
        aligned,
        max_samples=args.max_samples,
        min_positive_per_target=args.min_positive_per_target,
        negative_samples=args.negative_samples,
        focus_features=focus_features,
        seed=args.seed,
    )
    write_jsonl(args.output_inference, (row for _, row, _ in selected))
    write_jsonl(args.output_ground_truth, (row for _, _, row in selected))
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
