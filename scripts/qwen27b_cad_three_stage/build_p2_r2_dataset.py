#!/usr/bin/env python3
"""Build the small, replay-balanced P2-R2 protocol adaptation dataset.

P2-R1 converted almost every full-image P3 conversation to P2 and replayed
only a small feature-only subset.  That made layout formatting easier but
caused measurable forgetting in holes and rare views.  P2-R2 deliberately:

* keeps a compact P2 curriculum;
* balances all view labels plus weak CAD feature labels;
* replays original P3 conversations, including hard/rare examples;
* keeps a small feature-only replay slice.

The output is deterministic and contains no invented labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from build_p2_protocol_dataset import (
    atomic_json,
    atomic_jsonl,
    convert_three_stage,
    parse_assistant,
    read_rows,
    row_kind,
)
from cad_schema import VIEW_CATEGORIES, normalize_category, validate_stage_output


DEFAULT_FOCUS_FEATURES = (
    "Round Hole",
    "Round Hole Group",
    "Slotted Hole",
    "Slotted Hole Group",
    "Fillet",
    "Fillet Group",
    "Chamfer",
    "Chamfer Group",
)


def stable_rank(seed: int, namespace: str, index: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{namespace}\0{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def target_tokens(
    *,
    views: Sequence[str],
    features: Sequence[str],
) -> tuple[str, ...]:
    return tuple(
        [f"view:{category}" for category in views]
        + [f"feature:{category}" for category in features]
    )


def audit_tokens(audit: Mapping[str, Any]) -> set[str]:
    return {
        *(f"view:{category}" for category in audit["view_categories"]),
        *(f"feature:{category}" for category in audit["feature_categories"]),
    }


def feature_only_tokens(row: Mapping[str, Any]) -> set[str]:
    messages = row["messages"]
    stage3 = parse_assistant(messages[2], "feature_only")
    errors = validate_stage_output(3, stage3, strict=True)
    if errors:
        raise ValueError("invalid feature-only target: " + " | ".join(errors))
    result: set[str] = set()
    for item in stage3.get("features", []):
        if not isinstance(item, Mapping):
            continue
        category = normalize_category(item.get("category"), kind="feature")
        if category:
            result.add(f"feature:{category}")
    return result


def balanced_select(
    pool: Sequence[int],
    labels: Mapping[int, set[str]],
    targets: Sequence[str],
    count: int,
    *,
    seed: int,
    namespace: str,
) -> list[int]:
    """Select unique rows while keeping target-instance counts approximately even."""

    if count < 0:
        raise ValueError("selection count must be non-negative")
    target_set = set(targets)
    available = {
        target: sum(target in labels.get(index, set()) for index in pool)
        for target in targets
    }
    selected: list[int] = []
    selected_set: set[int] = set()
    covered = Counter()

    while len(selected) < count:
        viable_targets = [
            target
            for target in targets
            if any(
                index not in selected_set and target in labels.get(index, set())
                for index in pool
            )
        ]
        if not viable_targets:
            break
        # Lowest covered count wins; rare labels break ties so they cannot be
        # starved by common front-view and round-hole rows.
        chosen_target = min(
            viable_targets,
            key=lambda target: (
                covered[target],
                available[target],
                stable_rank(seed, f"{namespace}:target", targets.index(target)),
            ),
        )
        candidates = [
            index
            for index in pool
            if index not in selected_set
            and chosen_target in labels.get(index, set())
        ]
        index = max(
            candidates,
            key=lambda value: (
                sum(
                    1.0 / (1.0 + covered[token])
                    for token in labels.get(value, set()) & target_set
                ),
                -stable_rank(seed, namespace, value),
            ),
        )
        selected.append(index)
        selected_set.add(index)
        covered.update(labels.get(index, set()) & target_set)

    if len(selected) < count:
        filler = sorted(
            (index for index in pool if index not in selected_set),
            key=lambda index: stable_rank(seed + 1, f"{namespace}:filler", index),
        )
        selected.extend(filler[: count - len(selected)])
    if len(selected) != count:
        raise ValueError(
            f"{namespace}: requested {count} unique rows, selected {len(selected)}"
        )
    return selected


def random_select(
    pool: Sequence[int],
    count: int,
    *,
    seed: int,
    namespace: str,
) -> list[int]:
    values = sorted(
        pool,
        key=lambda index: stable_rank(seed, namespace, index),
    )
    if len(values) < count:
        raise ValueError(
            f"{namespace}: requested {count} rows from a pool of {len(values)}"
        )
    return values[:count]


def clean_source_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "messages": row["messages"],
        "images": row["images"],
    }


def category_instances(
    indices: Sequence[int],
    labels: Mapping[int, set[str]],
) -> dict[str, int]:
    counts = Counter(
        token
        for index in indices
        for token in labels.get(index, set())
    )
    return dict(sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--p2-hard", type=int, default=200)
    parser.add_argument("--p2-general", type=int, default=240)
    parser.add_argument("--p3-hard", type=int, default=100)
    parser.add_argument("--p3-general", type=int, default=60)
    parser.add_argument("--feature-replay", type=int, default=40)
    parser.add_argument("--focus-feature", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_path = output_dir / "train_p2_r2.jsonl"
    manifest_path = output_dir / "p2_r2_manifest.json"
    if not args.overwrite:
        existing = [path for path in (output_path, manifest_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "outputs already exist; pass --overwrite deliberately: "
                + ", ".join(map(str, existing))
            )

    focus_features = tuple(args.focus_feature or DEFAULT_FOCUS_FEATURES)
    targets = target_tokens(views=VIEW_CATEGORIES, features=focus_features)
    rows = read_rows(args.train_input.resolve())

    three_indices: list[int] = []
    feature_indices: list[int] = []
    p2_rows: dict[int, dict[str, Any]] = {}
    labels: dict[int, set[str]] = {}
    source_counts = Counter()
    for index, row in enumerate(rows):
        kind = row_kind(row)
        source_counts[kind] += 1
        if kind == "three_stage":
            converted, audit = convert_three_stage(row)
            three_indices.append(index)
            p2_rows[index] = converted
            labels[index] = audit_tokens(audit)
        elif kind == "feature_only":
            feature_indices.append(index)
            labels[index] = feature_only_tokens(row)
        else:
            raise ValueError(f"row {index}: unsupported message structure")

    p2_hard = balanced_select(
        three_indices,
        labels,
        targets,
        args.p2_hard,
        seed=args.seed,
        namespace="p2_hard",
    )
    p2_general = random_select(
        [index for index in three_indices if index not in set(p2_hard)],
        args.p2_general,
        seed=args.seed,
        namespace="p2_general",
    )
    p3_hard = balanced_select(
        three_indices,
        labels,
        targets,
        args.p3_hard,
        seed=args.seed + 11,
        namespace="p3_hard",
    )
    p3_general = random_select(
        [index for index in three_indices if index not in set(p3_hard)],
        args.p3_general,
        seed=args.seed + 11,
        namespace="p3_general",
    )
    feature_replay = balanced_select(
        feature_indices,
        labels,
        [f"feature:{category}" for category in focus_features],
        args.feature_replay,
        seed=args.seed + 23,
        namespace="feature_replay",
    )

    components: list[tuple[str, int, dict[str, Any]]] = []
    components.extend(("p2_hard", index, p2_rows[index]) for index in p2_hard)
    components.extend(("p2_general", index, p2_rows[index]) for index in p2_general)
    components.extend(
        ("p3_hard", index, clean_source_row(rows[index])) for index in p3_hard
    )
    components.extend(
        ("p3_general", index, clean_source_row(rows[index])) for index in p3_general
    )
    components.extend(
        ("feature_replay", index, clean_source_row(rows[index]))
        for index in feature_replay
    )
    random.Random(args.seed).shuffle(components)
    output_rows = [row for _, _, row in components]
    output_count, output_sha = atomic_jsonl(output_path, output_rows)

    selections = {
        "p2_hard": p2_hard,
        "p2_general": p2_general,
        "p3_hard": p3_hard,
        "p3_general": p3_general,
        "feature_replay": feature_replay,
    }
    exact_duplicates = output_count - len(
        {
            hashlib.sha256(
                json.dumps(row, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()
            for row in output_rows
        }
    )
    p2_source = set(p2_hard) | set(p2_general)
    p3_source = set(p3_hard) | set(p3_general)
    manifest = {
        "schema": "cad_p2_protocol_r2",
        "seed": args.seed,
        "input": str(args.train_input.resolve()),
        "source_counts": dict(sorted(source_counts.items())),
        "targets": list(targets),
        "composition": {name: len(indices) for name, indices in selections.items()},
        "intentional_cross_protocol_pairs": len(p2_source & p3_source),
        "exact_duplicate_rows": exact_duplicates,
        "category_instances": {
            name: category_instances(indices, labels)
            for name, indices in selections.items()
        },
        "output": {
            "path": str(output_path),
            "rows": output_count,
            "sha256": output_sha,
        },
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
