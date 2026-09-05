#!/usr/bin/env python3
"""Compare validation-generation reports and select an eligible checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def load_report(spec: str) -> tuple[str, Path, dict[str, Any]]:
    if "=" not in spec:
        raise ValueError("--report must use LABEL=PATH")
    label, raw_path = spec.split("=", 1)
    path = Path(raw_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: report is not an object")
    return label.strip(), path, dict(value)


def metric(report: Mapping[str, Any], *path: str) -> float:
    value: Any = report
    for key in path:
        value = value.get(key, {}) if isinstance(value, Mapping) else {}
    return float(value or 0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, help="LABEL=PATH")
    parser.add_argument("--min-json-valid", type=float, default=0.98)
    parser.add_argument("--min-constraint-valid", type=float, default=0.98)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    candidates = []
    for spec in args.report:
        label, path, report = load_report(spec)
        json_valid = metric(report, "json_valid_rate")
        constraint_valid = metric(report, "constraint_valid_rate")
        row = {
            "label": label,
            "path": str(path),
            "samples": int(report.get("samples", 0)),
            "eligible": (
                json_valid >= args.min_json_valid
                and constraint_valid >= args.min_constraint_valid
            ),
            "FocusScore": metric(report, "FocusScore"),
            "round_hole_f1": metric(report, "focus", "round_hole_f1"),
            "slotted_hole_f1": metric(report, "focus", "slotted_hole_f1"),
            "view_f1": metric(report, "focus", "view_f1"),
            "feature_macro_f1": metric(report, "feature", "macro_f1"),
            "strict_feature_f1": metric(report, "feature_strict", "f1"),
            "json_valid_rate": json_valid,
            "constraint_valid_rate": constraint_valid,
            "per_image_exact_pass": metric(report, "per_image_exact_pass"),
        }
        row["rank_key"] = [
            row["eligible"],
            row["FocusScore"],
            row["feature_macro_f1"],
            row["strict_feature_f1"],
            row["per_image_exact_pass"],
        ]
        candidates.append(row)

    candidates.sort(key=lambda row: tuple(row["rank_key"]), reverse=True)
    result = {
        "selection_split": "validation",
        "gates": {
            "min_json_valid": args.min_json_valid,
            "min_constraint_valid": args.min_constraint_valid,
        },
        "winner": candidates[0]["label"] if candidates and candidates[0]["eligible"] else None,
        "candidates": candidates,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if result["winner"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
