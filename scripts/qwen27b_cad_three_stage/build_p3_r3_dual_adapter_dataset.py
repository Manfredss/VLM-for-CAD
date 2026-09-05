#!/usr/bin/env python3
"""Build leakage-safe P3-R3 datasets for two specialised LoRA adapters.

The layout/view adapter learns only the first two P3 turns.  The feature
adapter learns a fresh-image Stage-3 turn conditioned on Stage-1/2 context.
All targets come from the original P3 training split; the optional noisy
context changes only upstream reference boxes, never the supervised target.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from build_p2_protocol_dataset import (
    atomic_json,
    atomic_jsonl,
    json_compact,
    parse_assistant,
    read_rows,
    row_kind,
)
from cad_schema import (
    FEATURE_CATEGORIES,
    VIEW_CATEGORIES,
    infer_family_id,
    normalize_category,
    validate_cross_stage_outputs,
    validate_stage_output,
)


DEFAULT_FOCUS_VIEWS = tuple(VIEW_CATEGORIES)
DEFAULT_FOCUS_FEATURES = (
    "Round Hole",
    "Round Hole Group",
    "Slotted Hole",
    "Slotted Hole Group",
    "Fillet",
    "Fillet Group",
    "Chamfer",
    "Chamfer Group",
    "Counterbore Hole",
    "Counterbore Hole Group",
)


def stable_rank(seed: int, namespace: str, value: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def row_image(row: Mapping[str, Any]) -> str:
    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1 or not images[0]:
        raise ValueError("every training row must reference exactly one image")
    return str(images[0])


def parse_three_stage(row: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    if row_kind(row) != "three_stage":
        raise ValueError("expected a seven-message three-stage row")
    messages = row["messages"]
    stages = (
        parse_assistant(messages[2], "stage1"),
        parse_assistant(messages[4], "stage2"),
        parse_assistant(messages[6], "stage3"),
    )
    errors = []
    for number, stage in enumerate(stages, 1):
        errors.extend(validate_stage_output(number, stage, strict=True))
    errors.extend(validate_cross_stage_outputs(*stages))
    if errors:
        raise ValueError("invalid source P3 row: " + " | ".join(errors))
    return stages


def feature_stage(row: Mapping[str, Any]) -> dict[str, Any]:
    kind = row_kind(row)
    if kind == "three_stage":
        return parse_three_stage(row)[2]
    if kind == "feature_only":
        stage3 = parse_assistant(row["messages"][2], "feature_only")
        errors = validate_stage_output(3, stage3, strict=True)
        if errors:
            raise ValueError("invalid feature-only row: " + " | ".join(errors))
        return stage3
    raise ValueError("unsupported source row")


def categories_for_row(row: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    views: set[str] = set()
    features: set[str] = set()
    if row_kind(row) == "three_stage":
        _, stage2, stage3 = parse_three_stage(row)
        for item in stage2.get("views", []):
            category = normalize_category(item.get("category"), kind="view")
            if category:
                views.add(category)
    else:
        stage3 = feature_stage(row)
    for item in stage3.get("features", []):
        category = normalize_category(item.get("category"), kind="feature")
        if category:
            features.add(category)
    return views, features


def balanced_select(
    pool: Sequence[int],
    labels: Mapping[int, set[str]],
    targets: Sequence[str],
    count: int,
    *,
    seed: int,
    namespace: str,
) -> list[int]:
    if count < 0 or count > len(pool):
        raise ValueError(f"{namespace}: cannot select {count} unique rows from {len(pool)}")
    target_set = set(targets)
    available = {
        target: sum(target in labels.get(index, set()) for index in pool)
        for target in targets
    }
    selected: list[int] = []
    selected_set: set[int] = set()
    covered: Counter[str] = Counter()
    while len(selected) < count:
        viable = [
            target
            for target in targets
            if any(
                index not in selected_set and target in labels.get(index, set())
                for index in pool
            )
        ]
        if not viable:
            break
        chosen = min(
            viable,
            key=lambda target: (
                covered[target],
                available[target],
                stable_rank(seed, f"{namespace}:target", target),
            ),
        )
        candidates = [
            index
            for index in pool
            if index not in selected_set and chosen in labels.get(index, set())
        ]
        index = max(
            candidates,
            key=lambda candidate: (
                sum(
                    1.0 / (1.0 + covered[token])
                    for token in labels.get(candidate, set()) & target_set
                ),
                -stable_rank(seed, namespace, str(candidate)),
            ),
        )
        selected.append(index)
        selected_set.add(index)
        covered.update(labels.get(index, set()) & target_set)
    if len(selected) < count:
        filler = sorted(
            (index for index in pool if index not in selected_set),
            key=lambda index: stable_rank(seed + 1, f"{namespace}:filler", str(index)),
        )
        selected.extend(filler[: count - len(selected)])
    if len(selected) != count:
        raise ValueError(f"{namespace}: selected {len(selected)} of requested {count}")
    return selected


def select_hard_and_general(
    pool: Sequence[int],
    labels: Mapping[int, set[str]],
    targets: Sequence[str],
    total: int,
    hard_fraction: float,
    *,
    seed: int,
    namespace: str,
) -> tuple[list[int], list[int]]:
    hard_count = round(total * hard_fraction)
    general_count = total - hard_count
    hard = balanced_select(
        pool, labels, targets, hard_count, seed=seed, namespace=f"{namespace}:hard"
    )
    hard_set = set(hard)
    remaining = sorted(
        (index for index in pool if index not in hard_set),
        key=lambda index: stable_rank(seed + 7, f"{namespace}:general", str(index)),
    )
    if len(remaining) < general_count:
        raise ValueError(f"{namespace}: insufficient general replay rows")
    return hard, remaining[:general_count]


def layout_view_row(row: Mapping[str, Any]) -> dict[str, Any]:
    parse_three_stage(row)
    return {
        "messages": copy.deepcopy(row["messages"][:5]),
        "images": [row_image(row)],
    }


def jitter_bbox(bbox: Sequence[Any], rng: random.Random) -> list[int]:
    values = [int(round(float(value))) for value in bbox]
    jittered = [
        max(0, min(1000, value + rng.randint(-8, 8))) for value in values
    ]
    x1, y1, x2, y2 = jittered
    if x2 <= x1:
        x2 = min(1000, x1 + 1)
        x1 = min(x1, x2 - 1)
    if y2 <= y1:
        y2 = min(1000, y1 + 1)
        y1 = min(y1, y2 - 1)
    return [x1, y1, x2, y2]


def noisy_context(
    stage1: Mapping[str, Any], stage2: Mapping[str, Any], *, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(seed)
    first = copy.deepcopy(stage1)
    second = copy.deepcopy(stage2)
    for region in first.get("regions", []):
        if isinstance(region, Mapping) and isinstance(region.get("bbox"), list):
            region["bbox"] = jitter_bbox(region["bbox"], rng)
    stage1_boxes = {
        str(item.get("region_id")): item.get("bbox")
        for item in first.get("regions", [])
        if isinstance(item, Mapping)
    }
    for view in second.get("views", []):
        if not isinstance(view, Mapping):
            continue
        region_id = str(view.get("region_id", ""))
        if "bbox" in view and region_id in stage1_boxes:
            view["bbox"] = stage1_boxes[region_id]
    if rng.random() < 0.10:
        second["projection_method"] = "unknown"
    return first, second


def feature_row(
    row: Mapping[str, Any], *, context_mode: str, seed: int
) -> dict[str, Any]:
    kind = row_kind(row)
    if kind == "feature_only":
        feature_stage(row)
        return {
            "messages": copy.deepcopy(row["messages"]),
            "images": [row_image(row)],
        }
    stage1, stage2, stage3 = parse_three_stage(row)
    if context_mode == "noisy_proxy":
        context1, context2 = noisy_context(stage1, stage2, seed=seed)
    elif context_mode == "ground_truth":
        context1, context2 = stage1, stage2
    else:
        raise ValueError(f"unknown context mode: {context_mode}")
    feature_prompt = str(row["messages"][5].get("content") or "")
    context = (
        "\n\n上游阶段参考结果（只用于关联视图区；feature_scope 和特征标签仍以当前任务为准）："
        f"\nstage1={json_compact(context1)}"
        f"\nstage2={json_compact(context2)}"
    )
    return {
        "messages": [
            copy.deepcopy(row["messages"][0]),
            {"role": "user", "content": f"<image>{feature_prompt}{context}"},
            {"role": "assistant", "content": json_compact(stage3)},
        ],
        "images": [row_image(row)],
    }


def context_mode(seed: int, index: int, noisy_ratio: float) -> str:
    digest = stable_rank(seed, "context_mode", str(index)) / float(1 << 64)
    return "noisy_proxy" if digest < noisy_ratio else "ground_truth"


def label_counts(indices: Sequence[int], labels: Mapping[int, set[str]]) -> dict[str, int]:
    counts = Counter(token for index in indices for token in labels.get(index, set()))
    return dict(sorted(counts.items()))


def exact_duplicate_count(rows: Sequence[Mapping[str, Any]]) -> int:
    hashes = {
        hashlib.sha256(
            json.dumps(row, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        for row in rows
    }
    return len(rows) - len(hashes)


def build_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    total: int,
    hard_fraction: float,
    noisy_context_ratio: float,
    seed: int,
    split: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    three_indices: list[int] = []
    feature_indices: list[int] = []
    view_labels: dict[int, set[str]] = {}
    feature_labels: dict[int, set[str]] = {}
    families: dict[int, str] = {}
    source_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        kind = row_kind(row)
        source_counts[kind] += 1
        if kind not in {"three_stage", "feature_only"}:
            raise ValueError(f"{split} row {index}: unsupported message structure")
        views, features = categories_for_row(row)
        view_labels[index] = {f"view:{category}" for category in views}
        feature_labels[index] = {f"feature:{category}" for category in features}
        families[index] = infer_family_id(row_image(row))
        feature_indices.append(index)
        if kind == "three_stage":
            three_indices.append(index)

    view_targets = [f"view:{category}" for category in DEFAULT_FOCUS_VIEWS]
    feature_targets = [f"feature:{category}" for category in DEFAULT_FOCUS_FEATURES]
    lv_hard, lv_general = select_hard_and_general(
        three_indices,
        view_labels,
        view_targets,
        total,
        hard_fraction,
        seed=seed,
        namespace=f"{split}:layout_view",
    )
    feat_hard, feat_general = select_hard_and_general(
        feature_indices,
        feature_labels,
        feature_targets,
        total,
        hard_fraction,
        seed=seed + 101,
        namespace=f"{split}:feature",
    )

    layout_components = [("hard", index) for index in lv_hard] + [
        ("general", index) for index in lv_general
    ]
    feature_components = [("hard", index) for index in feat_hard] + [
        ("general", index) for index in feat_general
    ]
    random.Random(seed).shuffle(layout_components)
    random.Random(seed + 1).shuffle(feature_components)
    layout_output = [layout_view_row(rows[index]) for _, index in layout_components]
    feature_output = []
    context_counts: Counter[str] = Counter()
    for _, index in feature_components:
        if row_kind(rows[index]) == "feature_only":
            mode = "feature_only_no_context"
        else:
            mode = context_mode(seed + 211, index, noisy_context_ratio)
        context_counts[mode] += 1
        feature_output.append(
            feature_row(
                rows[index],
                context_mode=("ground_truth" if mode == "feature_only_no_context" else mode),
                seed=stable_rank(seed + 307, "jitter", str(index)),
            )
        )

    return {
        "layout_view": layout_output,
        "feature": feature_output,
    }, {
        "source_counts": dict(sorted(source_counts.items())),
        "composition": {
            "layout_view": {"hard": len(lv_hard), "general": len(lv_general)},
            "feature": {"hard": len(feat_hard), "general": len(feat_general)},
        },
        "context_modes": dict(sorted(context_counts.items())),
        "category_instances": {
            "layout_view_hard": label_counts(lv_hard, view_labels),
            "layout_view_general": label_counts(lv_general, view_labels),
            "feature_hard": label_counts(feat_hard, feature_labels),
            "feature_general": label_counts(feat_general, feature_labels),
        },
        "selected_families": {
            "layout_view": sorted({families[index] for index in lv_hard + lv_general}),
            "feature": sorted({families[index] for index in feat_hard + feat_general}),
        },
        "exact_duplicate_rows": {
            "layout_view": exact_duplicate_count(layout_output),
            "feature": exact_duplicate_count(feature_output),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--val-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=640)
    parser.add_argument("--val-rows", type=int, default=64)
    parser.add_argument("--hard-fraction", type=float, default=0.40)
    parser.add_argument("--noisy-context-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.hard_fraction <= 1:
        parser.error("hard-fraction must be in [0,1]")
    if not 0 <= args.noisy_context_ratio <= 1:
        parser.error("noisy-context-ratio must be in [0,1]")

    output_dir = args.output_dir.resolve()
    paths = {
        "train_layout_view": output_dir / "train_layout_view.jsonl",
        "val_layout_view": output_dir / "val_layout_view.jsonl",
        "train_feature": output_dir / "train_feature.jsonl",
        "val_feature": output_dir / "val_feature.jsonl",
        "manifest": output_dir / "p3_r3_dual_adapter_manifest.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "outputs already exist; pass --overwrite deliberately: "
            + ", ".join(map(str, existing))
        )

    train_rows = read_rows(args.train_input.resolve())
    val_rows = read_rows(args.val_input.resolve())
    train_output, train_audit = build_split(
        train_rows,
        total=args.train_rows,
        hard_fraction=args.hard_fraction,
        noisy_context_ratio=args.noisy_context_ratio,
        seed=args.seed,
        split="train",
    )
    val_output, val_audit = build_split(
        val_rows,
        total=args.val_rows,
        hard_fraction=args.hard_fraction,
        noisy_context_ratio=args.noisy_context_ratio,
        seed=args.seed + 1000,
        split="val",
    )

    train_families = {
        family
        for task in train_audit["selected_families"].values()
        for family in task
    }
    val_families = {
        family
        for task in val_audit["selected_families"].values()
        for family in task
    }
    overlap = sorted(train_families & val_families)
    if overlap:
        raise ValueError(f"train/val family leakage detected: {overlap[:20]}")

    output_meta = {}
    for split in ("train", "val"):
        for task in ("layout_view", "feature"):
            path = paths[f"{split}_{task}"]
            count, sha = atomic_jsonl(path, (train_output if split == "train" else val_output)[task])
            output_meta[f"{split}_{task}"] = {
                "path": str(path),
                "rows": count,
                "sha256": sha,
            }
    manifest = {
        "schema": "cad_p3_r3_dual_adapter_v1",
        "seed": args.seed,
        "inputs": {
            "train": str(args.train_input.resolve()),
            "val": str(args.val_input.resolve()),
        },
        "strategy": {
            "hard_fraction": args.hard_fraction,
            "general_replay_fraction": 1.0 - args.hard_fraction,
            "noisy_context_ratio": args.noisy_context_ratio,
            "noisy_context": "deterministic +/-8 bbox jitter and 10% projection_method=unknown; targets unchanged",
            "focus_views": list(DEFAULT_FOCUS_VIEWS),
            "focus_features": list(DEFAULT_FOCUS_FEATURES),
        },
        "family_leakage": overlap,
        "audit": {"train": train_audit, "val": val_audit},
        "outputs": output_meta,
    }
    atomic_json(paths["manifest"], manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
