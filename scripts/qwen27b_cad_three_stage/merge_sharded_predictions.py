#!/usr/bin/env python3
"""Merge resumable inference shards into one deterministic JSONL file."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: record is not an object")
            yield dict(value)


def record_key(record: Mapping[str, Any]) -> str:
    for field in ("sample_id", "image", "dataitem_name"):
        value = str(record.get(field) or "").strip()
        if value:
            return f"{field}:{value}"
    raise ValueError("record has no sample_id, image, or dataitem_name")


def reference_order(path: Path) -> dict[str, int]:
    order: dict[str, int] = {}
    for index, record in enumerate(read_jsonl(path)):
        candidates = {
            f"{field}:{str(record.get(field) or '').strip()}"
            for field in ("sample_id", "image", "dataitem_name")
            if str(record.get(field) or "").strip()
        }
        for candidate in candidates:
            order.setdefault(candidate, index)
    return order


def atomic_write(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-jsonl", type=Path)
    parser.add_argument("--expected-count", type=int, default=0)
    args = parser.parse_args()

    inputs = sorted({path.resolve() for path in args.input})
    if not inputs:
        raise ValueError("no input shards")
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing input shards: {missing}")

    merged: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for path in inputs:
        for record in read_jsonl(path):
            key = record_key(record)
            if key in merged:
                if merged[key] != record:
                    raise ValueError(f"conflicting duplicate record: {key}")
                continue
            merged[key] = record

    if args.expected_count and len(merged) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} records, found {len(merged)}"
        )

    records = list(merged.values())
    if args.reference_jsonl:
        order = reference_order(args.reference_jsonl.resolve())
        records.sort(key=lambda record: order.get(record_key(record), len(order)))
    atomic_write(args.output.resolve(), records)
    print(
        json.dumps(
            {
                "inputs": [str(path) for path in inputs],
                "records": len(records),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
