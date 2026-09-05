#!/usr/bin/env python3
"""Convert existing three-stage ms-swift rows into P2 protocol-adaptation data.

P2 uses two model generations:

1. One compact object containing Stage-1 ``regions`` plus Stage-2
   ``projection_method`` and ``views``.
2. The existing exhaustive Stage-3 feature object.

No annotations are invented.  The converter only restructures assistant
targets already present in the leakage-safe train/validation JSONL files.
Feature-only rows can be retained as a deterministic replay subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cad_schema import (
    DOCUMENT_CATEGORIES,
    SYSTEM_PROMPT,
    VIEW_CATEGORIES,
    layout_view_prompt,
    validate_cross_stage_outputs,
    validate_stage_output,
)


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError(f"{path}: top-level JSON must be a list")
            values: Iterable[Any] = payload
        else:
            values = (
                json.loads(line)
                for line in handle
                if line.strip()
            )
        for index, value in enumerate(values):
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}: row {index} is not an object")
            rows.append(dict(value))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def parse_assistant(message: Mapping[str, Any], label: str) -> dict[str, Any]:
    if message.get("role") != "assistant":
        raise ValueError(f"{label}: expected assistant role")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError(f"{label}: assistant content is not text")
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}: assistant content is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}: assistant JSON is not an object")
    return dict(value)


def parse_layout_scope(content: str) -> tuple[list[str], list[str]]:
    marker = "layout_scope="
    position = content.rfind(marker)
    if position < 0:
        return list(VIEW_CATEGORIES), list(DOCUMENT_CATEGORIES)
    text = content[position + len(marker) :].lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as exc:
        raise ValueError("cannot parse layout_scope from Stage-1 prompt") from exc
    if not isinstance(value, Mapping):
        raise ValueError("layout_scope is not an object")
    view_scope = [
        str(category)
        for category in value.get("view_scope", [])
        if str(category) in VIEW_CATEGORIES
    ]
    document_scope = [
        str(category)
        for category in value.get("document_scope", [])
        if str(category) in DOCUMENT_CATEGORIES
    ]
    return view_scope, document_scope


def combined_scope_suffix(
    view_scope: Sequence[str], document_scope: Sequence[str]
) -> str:
    scope = {
        "view_scope": list(view_scope),
        "document_scope": list(document_scope),
    }
    return (
        "\n本训练样本仅对下列 layout_scope 中的视图和文档类别做了穷尽标注。"
        "未列入 scope 的类别不是负样本，不得补造或强行分类。"
        f"\nlayout_scope={json_compact(scope)}"
    )


def compact_layout_view(
    stage1: Mapping[str, Any], stage2: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "projection_method": stage2.get("projection_method", "unknown"),
        "regions": [
            {
                "region_id": region["region_id"],
                "category": region["category"],
                "bbox": region["bbox"],
            }
            for region in stage1.get("regions", [])
        ],
        "views": [
            {"region_id": view["region_id"], "category": view["category"]}
            for view in stage2.get("views", [])
        ],
    }


def row_kind(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return "invalid"
    roles = [message.get("role") for message in messages if isinstance(message, Mapping)]
    if len(messages) == 7 and roles == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]:
        return "three_stage"
    if len(messages) == 3 and roles == ["system", "user", "assistant"]:
        return "feature_only"
    return "invalid"


def convert_three_stage(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = row["messages"]
    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1 or not images[0]:
        raise ValueError("P2 row must reference exactly one image")

    stage1 = parse_assistant(messages[2], "stage1")
    stage2 = parse_assistant(messages[4], "stage2")
    stage3 = parse_assistant(messages[6], "stage3")
    stage_errors = (
        validate_stage_output(1, stage1, strict=True)
        + validate_stage_output(2, stage2, strict=True)
        + validate_stage_output(3, stage3, strict=True)
    )
    if stage_errors:
        raise ValueError("invalid source targets: " + " | ".join(stage_errors))
    cross_errors = validate_cross_stage_outputs(stage1, stage2, stage3)
    if cross_errors:
        raise ValueError("invalid source constraints: " + " | ".join(cross_errors))

    first_user = messages[1]
    if not isinstance(first_user, Mapping) or first_user.get("role") != "user":
        raise ValueError("stage1 user message is malformed")
    first_content = str(first_user.get("content") or "")
    view_scope, document_scope = parse_layout_scope(first_content)
    p2_prompt = layout_view_prompt(
        document_scope=document_scope,
        view_scope=view_scope,
    )
    p2_prompt += combined_scope_suffix(view_scope, document_scope)

    feature_user = messages[5]
    if not isinstance(feature_user, Mapping) or feature_user.get("role") != "user":
        raise ValueError("stage3 user message is malformed")
    output = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>{p2_prompt}"},
            {
                "role": "assistant",
                "content": json_compact(compact_layout_view(stage1, stage2)),
            },
            {"role": "user", "content": str(feature_user.get("content") or "")},
            {"role": "assistant", "content": json_compact(stage3)},
        ],
        "images": [str(images[0])],
    }
    audit = {
        "view_scope": view_scope,
        "document_scope": document_scope,
        "regions": len(stage1.get("regions", [])),
        "views": len(stage2.get("views", [])),
        "features": len(stage3.get("features", [])),
        "region_categories": [
            str(item.get("category")) for item in stage1.get("regions", [])
        ],
        "view_categories": [
            str(item.get("category")) for item in stage2.get("views", [])
        ],
        "feature_categories": [
            str(item.get("category")) for item in stage3.get("features", [])
        ],
    }
    return output, audit


def deterministic_keep(row: Mapping[str, Any], index: int, ratio: float, seed: int) -> bool:
    if ratio <= 0:
        return False
    if ratio >= 1:
        return True
    images = row.get("images")
    image = str(images[0]) if isinstance(images, list) and images else ""
    digest = hashlib.sha256(f"{seed}\0{index}\0{image}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return value < ratio


def convert_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    replay_ratio: float,
    seed: int,
    shuffle: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    counts = Counter()
    categories = {
        "region": Counter(),
        "view": Counter(),
        "feature": Counter(),
    }
    for index, row in enumerate(rows):
        kind = row_kind(row)
        counts[f"source_{kind}"] += 1
        if kind == "three_stage":
            converted, audit = convert_three_stage(row)
            output.append(converted)
            counts["output_p2"] += 1
            categories["region"].update(audit["region_categories"])
            categories["view"].update(audit["view_categories"])
            categories["feature"].update(audit["feature_categories"])
        elif kind == "feature_only":
            if deterministic_keep(row, index, replay_ratio, seed):
                output.append(
                    {
                        "messages": row["messages"],
                        "images": row["images"],
                    }
                )
                counts["output_feature_replay"] += 1
        else:
            raise ValueError(f"row {index}: unsupported messages/roles structure")

    if not output or not counts["output_p2"]:
        raise ValueError("conversion produced no P2 rows")
    if shuffle:
        random.Random(seed).shuffle(output)
    return output, {
        "counts": dict(sorted(counts.items())),
        "category_instances": {
            kind: dict(sorted(values.items()))
            for kind, values in categories.items()
        },
    }


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        for row in rows:
            data = (json_compact(row) + "\n").encode("utf-8")
            handle.write(data)
            digest.update(data)
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)
    return count, digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--val-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-feature-replay-ratio", type=float, default=0.20)
    parser.add_argument("--val-feature-replay-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for name, ratio in (
        ("train", args.train_feature_replay_ratio),
        ("val", args.val_feature_replay_ratio),
    ):
        if not 0 <= ratio <= 1:
            parser.error(f"{name} feature replay ratio must be in [0,1]")

    output_dir = args.output_dir.resolve()
    outputs = {
        "train": output_dir / "train_p2_protocol.jsonl",
        "val": output_dir / "val_p2_protocol.jsonl",
        "manifest": output_dir / "p2_protocol_manifest.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "outputs already exist; pass --overwrite deliberately: " + ", ".join(existing)
        )

    train_rows = read_rows(args.train_input.resolve())
    val_rows = read_rows(args.val_input.resolve())
    converted_train, train_audit = convert_split(
        train_rows,
        replay_ratio=args.train_feature_replay_ratio,
        seed=args.seed,
        shuffle=True,
    )
    converted_val, val_audit = convert_split(
        val_rows,
        replay_ratio=args.val_feature_replay_ratio,
        seed=args.seed,
        shuffle=False,
    )
    train_count, train_sha = atomic_jsonl(outputs["train"], converted_train)
    val_count, val_sha = atomic_jsonl(outputs["val"], converted_val)
    manifest = {
        "schema": "cad_p2_protocol_v1",
        "seed": args.seed,
        "inputs": {
            "train": str(args.train_input.resolve()),
            "val": str(args.val_input.resolve()),
        },
        "feature_replay_ratio": {
            "train": args.train_feature_replay_ratio,
            "val": args.val_feature_replay_ratio,
        },
        "outputs": {
            "train": {
                "path": str(outputs["train"]),
                "rows": train_count,
                "sha256": train_sha,
            },
            "val": {
                "path": str(outputs["val"]),
                "rows": val_count,
                "sha256": val_sha,
            },
        },
        "audit": {
            "train": train_audit,
            "val": val_audit,
        },
    }
    atomic_json(outputs["manifest"], manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
