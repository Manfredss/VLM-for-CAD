#!/usr/bin/env python3
"""Audit three-stage CAD datasets before any expensive 27B training run.

Checks include JSON/schema integrity, canonical categories, annotation scopes,
normalized bboxes, duplicate IDs, family/augmentation leakage across splits and
per-split label distributions.  The basic audit is fully offline and depends
only on the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

try:
    from .cad_schema import (
        DOCUMENT_CATEGORIES,
        GENERIC_VIEW_REGION,
        infer_family_id,
        is_augmented_name,
        normalize_category,
        validate_cross_stage_outputs,
        validate_stage_output,
    )
    from .metrics import coerce_three_stage
except ImportError:
    from cad_schema import (  # type: ignore
        DOCUMENT_CATEGORIES,
        GENERIC_VIEW_REGION,
        infer_family_id,
        is_augmented_name,
        normalize_category,
        validate_cross_stage_outputs,
        validate_stage_output,
    )
    from metrics import coerce_three_stage  # type: ignore


class Issues:
    def __init__(self, max_examples: int = 20) -> None:
        self.max_examples = max_examples
        self.counts: dict[str, Counter[str]] = {
            "error": Counter(),
            "warning": Counter(),
        }
        self.examples: dict[str, dict[str, list[str]]] = {
            "error": defaultdict(list),
            "warning": defaultdict(list),
        }

    def add(self, severity: str, code: str, detail: str) -> None:
        self.counts[severity][code] += 1
        bucket = self.examples[severity][code]
        if len(bucket) < self.max_examples:
            bucket.append(detail)

    def total(self, severity: str) -> int:
        return sum(self.counts[severity].values())

    def as_dict(self) -> dict[str, Any]:
        return {
            severity: {
                code: {
                    "count": count,
                    "examples": self.examples[severity].get(code, []),
                }
                for code, count in sorted(self.counts[severity].items())
            }
            for severity in ("error", "warning")
        }


def _flatten_paths(values: Optional[list[list[Path]]]) -> list[Path]:
    return [path for group in values or [] for path in group]


def _read_file(path: Path, split: str, issues: Issues) -> list[Any]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        issues.add("error", "file_read", f"{split}:{path}: {exc}")
        return []
    if path.suffix.casefold() == ".jsonl":
        records = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                issues.add(
                    "error", "json_parse", f"{split}:{path}:{line_number}: {exc.msg}"
                )
        return records
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        issues.add("error", "json_parse", f"{split}:{path}:{exc.lineno}: {exc.msg}")
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        for key in ("data", "items", "samples", "records"):
            if isinstance(payload.get(key), list):
                return list(payload[key])
        return [dict(payload)]
    issues.add(
        "error",
        "json_structure",
        f"{split}:{path}: top-level JSON must be object or array",
    )
    return []


def _metadata(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return (
        record.get("metadata", {})
        if isinstance(record.get("metadata"), Mapping)
        else {}
    )


def _image_name(record: Mapping[str, Any]) -> str:
    images = record.get("images")
    if isinstance(images, list) and images:
        return str(images[0])
    for source in (record, _metadata(record)):
        for key in ("image", "image_path", "dataitem_name", "url", "source_image"):
            if source.get(key):
                return str(source[key])
    return ""


def _sample_id(record: Any, fallback: str) -> str:
    if not isinstance(record, Mapping):
        return fallback
    metadata = _metadata(record)
    for value in (
        record.get("sample_id"),
        record.get("dataitem_name"),
        metadata.get("sample_id"),
        metadata.get("dataitem_name"),
    ):
        if str(value or "").strip():
            return str(value).strip()
    image = _image_name(record)
    return Path(image).name if image else fallback


def _family_id(record: Mapping[str, Any], sample_id: str) -> str:
    metadata = _metadata(record)
    for source in (record, metadata):
        for key in ("family_id", "drawing_family", "parent_image_id"):
            if str(source.get(key, "")).strip():
                return str(source[key]).strip().casefold()
    return infer_family_id(_image_name(record) or sample_id)


def _is_augmented(record: Mapping[str, Any]) -> bool:
    metadata = _metadata(record)
    for source in (record, metadata):
        if "is_augmented" in source:
            return bool(source["is_augmented"])
        if str(source.get("parent_image_id", "")).strip():
            return True
    return is_augmented_name(_image_name(record))


def _scope(record: Mapping[str, Any], key: str, kind: str) -> Optional[list[str]]:
    containers: list[Mapping[str, Any]] = [record, _metadata(record)]
    annotation_scope = record.get("annotation_scope")
    if isinstance(annotation_scope, Mapping):
        containers.append(annotation_scope)
    found, raw = False, None
    for container in containers:
        if key in container:
            found, raw = True, container[key]
            break
    if not found:
        return None
    if not isinstance(raw, list):
        return []
    result = []
    for category in raw:
        canonical = normalize_category(category, kind=kind)
        if canonical and canonical not in result:
            result.append(canonical)
    return result


def _iter_raw_task_items(record: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
    for task in (
        record.get("tasks", []) if isinstance(record.get("tasks"), list) else []
    ):
        if not isinstance(task, Mapping):
            continue
        for item in (
            task.get("task_values", [])
            if isinstance(task.get("task_values"), list)
            else []
        ):
            if not isinstance(item, Mapping):
                continue
            value = (
                item.get("value", {}) if isinstance(item.get("value"), Mapping) else {}
            )
            yield (
                str(value.get("label", value.get("category", ""))),
                item.get("bbox", value.get("bbox")),
            )


def _audit_stage_relations(
    record: Mapping[str, Any],
    stages: Mapping[str, Any],
    split: str,
    sample_id: str,
    issues: Issues,
    distribution: dict[str, Counter[str]],
) -> None:
    context = f"{split}:{sample_id}"
    if all(stages.get(f"stage{number}") is None for number in (1, 2, 3)):
        issues.add(
            "error",
            "missing_targets",
            f"{context}: no stage targets or parseable assistant outputs",
        )
        return

    for number in (1, 2, 3):
        payload = stages.get(f"stage{number}")
        if payload is None:
            continue
        for error in validate_stage_output(number, payload, strict=False):
            code = "bbox" if ".bbox" in error else "schema"
            issues.add("error", code, f"{context}: stage{number}: {error}")

    stage1 = stages.get("stage1") if isinstance(stages.get("stage1"), Mapping) else None
    stage2 = stages.get("stage2") if isinstance(stages.get("stage2"), Mapping) else None
    stage3 = stages.get("stage3") if isinstance(stages.get("stage3"), Mapping) else None
    for error in validate_cross_stage_outputs(stage1, stage2, stage3):
        issues.add("error", "cross_stage_constraint", f"{context}: {error}")
    document_scope = _scope(record, "document_scope", "document")
    view_scope = _scope(record, "view_scope", "view")

    region_ids: set[str] = set()
    if stage1:
        for item in (
            stage1.get("regions", []) if isinstance(stage1.get("regions"), list) else []
        ):
            if not isinstance(item, Mapping):
                continue
            region_id = str(item.get("region_id", "")).strip()
            if region_id:
                region_ids.add(region_id)
            category = normalize_category(item.get("category"), kind="stage1")
            if category:
                distribution["layout"][category] += 1
                if (
                    category in DOCUMENT_CATEGORIES
                    and document_scope is not None
                    and category not in document_scope
                ):
                    issues.add(
                        "error",
                        "scope_violation",
                        f"{context}: Stage1 {category!r} outside document_scope",
                    )
                if category == GENERIC_VIEW_REGION and view_scope == []:
                    issues.add(
                        "error",
                        "scope_violation",
                        f"{context}: View Region present with empty view_scope",
                    )

    if stage2:
        seen_view_regions: set[str] = set()
        for item in (
            stage2.get("views", []) if isinstance(stage2.get("views"), list) else []
        ):
            if not isinstance(item, Mapping):
                continue
            region_id = str(item.get("region_id", "")).strip()
            if region_id:
                seen_view_regions.add(region_id)
                if stage1 and region_id not in region_ids:
                    issues.add(
                        "error",
                        "region_reference",
                        f"{context}: Stage2 region_id {region_id!r} absent from Stage1",
                    )
            category = normalize_category(item.get("category"), kind="view")
            if category:
                distribution["view"][category] += 1
                if view_scope is not None and category not in view_scope:
                    issues.add(
                        "error",
                        "scope_violation",
                        f"{context}: Stage2 {category!r} outside view_scope",
                    )
        if stage1:
            expected_view_regions = {
                str(item.get("region_id", "")).strip()
                for item in stage1.get("regions", [])
                if isinstance(item, Mapping)
                and normalize_category(item.get("category"), kind="stage1")
                == GENERIC_VIEW_REGION
            }
            missing = expected_view_regions - seen_view_regions
            if missing:
                issues.add(
                    "warning",
                    "unclassified_region",
                    f"{context}: Stage2 misses {sorted(missing)}",
                )

    if stage3:
        raw_scope = stage3.get("feature_scope")
        if not isinstance(raw_scope, list):
            issues.add(
                "error", "missing_scope", f"{context}: Stage3 feature_scope is required"
            )
            feature_scope: set[str] = set()
        else:
            feature_scope = {
                canonical
                for category in raw_scope
                if (canonical := normalize_category(category, kind="feature"))
                is not None
            }
            for category in feature_scope:
                distribution["feature_scope"][category] += 1
        metadata_scope = _scope(record, "feature_scope", "feature")
        if metadata_scope is not None and set(metadata_scope) != feature_scope:
            issues.add(
                "error",
                "scope_mismatch",
                f"{context}: metadata feature_scope differs from Stage3 feature_scope",
            )
        for item in (
            stage3.get("features", [])
            if isinstance(stage3.get("features"), list)
            else []
        ):
            if not isinstance(item, Mapping):
                continue
            category = normalize_category(item.get("category"), kind="feature")
            if category:
                distribution["feature"][category] += 1
                if feature_scope and category not in feature_scope:
                    issues.add(
                        "error",
                        "scope_violation",
                        f"{context}: Stage3 {category!r} outside feature_scope",
                    )


def audit(
    sources: Mapping[str, list[Path]],
    max_examples: int = 20,
    *,
    require_images: bool = False,
) -> dict[str, Any]:
    issues = Issues(max_examples)
    family_splits: dict[str, set[str]] = defaultdict(set)
    family_examples: dict[str, list[str]] = defaultdict(list)
    sample_splits: dict[str, set[str]] = defaultdict(set)
    augmented_by_family: dict[str, set[str]] = defaultdict(set)
    split_reports: dict[str, Any] = {}

    for split, paths in sources.items():
        distribution: dict[str, Counter[str]] = defaultdict(Counter)
        records_count = augmented_count = 0
        for path in paths:
            records = _read_file(path, split, issues)
            for row_index, record in enumerate(records):
                records_count += 1
                fallback = f"{path.name}:{row_index + 1}"
                sample_id = _sample_id(record, fallback)
                if not isinstance(record, Mapping):
                    issues.add(
                        "error",
                        "record_type",
                        f"{split}:{sample_id}: row must be a JSON object",
                    )
                    continue
                image_reference = _image_name(record)
                if not image_reference:
                    issues.add(
                        "error" if require_images else "warning",
                        "image_reference",
                        f"{split}:{sample_id}: no images/image path",
                    )
                elif "://" in image_reference and not image_reference.startswith(
                    "file://"
                ):
                    issues.add(
                        "error" if require_images else "warning",
                        "image_access",
                        f"{split}:{sample_id}: remote URI cannot be checked offline: {image_reference}",
                    )
                else:
                    local_image = Path(
                        image_reference.removeprefix("file://")
                    ).expanduser()
                    if not local_image.is_file():
                        issues.add(
                            "error" if require_images else "warning",
                            "image_access",
                            f"{split}:{sample_id}: image not readable: {local_image}",
                        )
                family = _family_id(record, sample_id)
                augmented = _is_augmented(record)
                augmented_count += int(augmented)
                family_splits[family].add(split)
                sample_splits[sample_id].add(split)
                if len(family_examples[family]) < max_examples:
                    family_examples[family].append(f"{split}:{sample_id}")
                if augmented:
                    augmented_by_family[family].add(split)
                    if split != "train":
                        issues.add(
                            "error",
                            "augmented_nontrain",
                            f"{split}:{sample_id} family={family}",
                        )

                stages = coerce_three_stage(record)
                _audit_stage_relations(
                    record, stages, split, sample_id, issues, distribution
                )

                # Raw Labelbox-style sources use pixel coordinates and cannot be
                # range-checked without image dimensions, but malformed xyxy is
                # still detectable.
                for raw_category, raw_bbox in _iter_raw_task_items(record):
                    if normalize_category(raw_category, kind="all") is None:
                        issues.add(
                            "error",
                            "unknown_category",
                            f"{split}:{sample_id}: {raw_category!r}",
                        )
                    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
                        issues.add(
                            "error",
                            "bbox",
                            f"{split}:{sample_id}: malformed raw bbox {raw_bbox!r}",
                        )
                    else:
                        try:
                            x1, y1, x2, y2 = [float(value) for value in raw_bbox]
                            if x1 >= x2 or y1 >= y2:
                                issues.add(
                                    "error",
                                    "bbox",
                                    f"{split}:{sample_id}: invalid raw xyxy {raw_bbox!r}",
                                )
                        except (TypeError, ValueError):
                            issues.add(
                                "error",
                                "bbox",
                                f"{split}:{sample_id}: nonnumeric raw bbox {raw_bbox!r}",
                            )

        split_reports[split] = {
            "files": [str(path) for path in paths],
            "records": records_count,
            "augmented_records": augmented_count,
            "distribution": {
                name: dict(counter.most_common())
                for name, counter in sorted(distribution.items())
            },
        }

    leaking_families = {
        family: sorted(splits)
        for family, splits in family_splits.items()
        if len(splits) > 1
    }
    duplicate_samples = {
        sample: sorted(splits)
        for sample, splits in sample_splits.items()
        if len(splits) > 1
    }
    augmentation_leaks = {
        family: sorted(family_splits[family])
        for family in augmented_by_family
        if len(family_splits[family]) > 1
    }
    for family, splits in leaking_families.items():
        issues.add(
            "error",
            "family_leakage",
            f"{family}: splits={splits}, examples={family_examples[family]}",
        )
    for sample, splits in duplicate_samples.items():
        issues.add("error", "sample_leakage", f"{sample}: splits={splits}")
    for family, splits in augmentation_leaks.items():
        issues.add(
            "error",
            "augmentation_leakage",
            f"{family}: augmented family crosses {splits}",
        )

    error_count, warning_count = issues.total("error"), issues.total("warning")
    return {
        "status": "fail" if error_count else ("warning" if warning_count else "pass"),
        "summary": {
            "records": sum(values["records"] for values in split_reports.values()),
            "families": len(family_splits),
            "errors": error_count,
            "warnings": warning_count,
        },
        "splits": split_reports,
        "leakage": {
            "family_count": len(leaking_families),
            "sample_id_count": len(duplicate_samples),
            "augmentation_family_count": len(augmentation_leaks),
            "families": leaking_families,
            "sample_ids": duplicate_samples,
            "augmentation_families": augmentation_leaks,
        },
        "issues": issues.as_dict(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Additional files; split inferred from filename, otherwise treated as train.",
    )
    parser.add_argument("--train", nargs="+", action="append", type=Path)
    parser.add_argument(
        "--val", "--validation", dest="val", nargs="+", action="append", type=Path
    )
    parser.add_argument("--test", nargs="+", action="append", type=Path)
    parser.add_argument("--json-out", "--output", dest="json_out", type=Path)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any audit error is found.",
    )
    parser.add_argument("--fail-on-warnings", action="store_true")
    parser.add_argument(
        "--require-images",
        action="store_true",
        help="Treat missing/unreadable image paths as errors (recommended on the training pod).",
    )
    return parser


def _infer_split(path: Path) -> str:
    name = path.name.casefold()
    if "test" in name:
        return "test"
    if "val" in name or "valid" in name or "dev" in name:
        return "val"
    return "train"


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    sources = {
        "train": _flatten_paths(args.train),
        "val": _flatten_paths(args.val),
        "test": _flatten_paths(args.test),
    }
    for path in args.paths:
        sources[_infer_split(path)].append(path)
    sources = {split: paths for split, paths in sources.items() if paths}
    if not sources:
        print("ERROR: provide at least one dataset path", file=sys.stderr)
        return 2

    report = audit(
        sources,
        max_examples=max(1, args.max_examples),
        require_images=args.require_images,
    )
    summary = report["summary"]
    print(
        f"Dataset audit: {report['status'].upper()}  records={summary['records']} families={summary['families']} errors={summary['errors']} warnings={summary['warnings']}"
    )
    print(
        f"Leakage: families={report['leakage']['family_count']} sample_ids={report['leakage']['sample_id_count']} augmentation_families={report['leakage']['augmentation_family_count']}"
    )
    for split, values in report["splits"].items():
        print(
            f"  {split:5s}: records={values['records']} augmented={values['augmented_records']}"
        )
    for severity in ("error", "warning"):
        for code, values in report["issues"][severity].items():
            print(f"  {severity.upper():7s} {code}: {values['count']}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Report: {args.json_out}")
    if args.strict and summary["errors"]:
        return 1
    if args.fail_on_warnings and (summary["errors"] or summary["warnings"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
