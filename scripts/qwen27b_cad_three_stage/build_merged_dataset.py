#!/usr/bin/env python3
"""Build a leakage-safe, three-stage CAD fine-tuning dataset.

This utility merges the complementary annotations from the two 5K exports and
the Siemens 788 export.  It intentionally keeps *annotation scope* separate
from positive labels: a category absent from an export that did not annotate
that category must never be learned as a negative.

Outputs (under ``--output-dir``):

* ``merged_annotations.json``: canonical, de-duplicated source annotations.
* ``manifest.jsonl``: one audit record per merged image.
* ``{train,val,test}_three_stage.jsonl``: ms-swift conversations.  Images with
  view annotations use Step1 -> Step2 -> Step3; partial/augmented images use an
  explicitly scoped, standalone Step3 conversation.
* ``{train,val,test}_ground_truth.jsonl``: IDs, structured stage targets, and
  audit metadata kept separate so ms-swift sees only ``messages``/``images``.
* ``train_view_crops.jsonl``: standalone high-resolution Step3 crop records.
* ``crop_manifest.jsonl`` and ``stats.json``: reproducibility/audit metadata.

The crop JSONL always uses deterministic crop filenames and crops are
materialized by default.  With ``--no-materialize-crops``, the active crop
JSONL is empty and records are written to ``train_view_crops_plan.jsonl``;
materialize or otherwise provide those files before activating that plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Iterator, Mapping, Sequence

try:
    from cad_schema import (
        DOCUMENT_CATEGORIES,
        FEATURE_CATEGORIES,
        SYSTEM_PROMPT,
        VIEW_CATEGORIES,
        infer_family_id,
        is_augmented_name,
        normalize_size,
        step1_prompt,
        step2_prompt,
        step3_prompt,
    )
except ImportError as exc:  # pragma: no cover - exercised by CLI users
    raise SystemExit(
        "Cannot import cad_schema.py. Keep it next to build_merged_dataset.py "
        "and run this script directly or add its directory to PYTHONPATH."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_JSON_10 = REPO_ROOT / "data" / "5k_10feats_v2_augmented.json"
DEFAULT_JSON_15 = REPO_ROOT / "data" / "5k_15feats_with_view_v2.json"
DEFAULT_JSON_788 = (
    REPO_ROOT / "scripts" / "qwen3.5-27b_788_pipeline" / "788_11Feats_View.json"
)


def _first_existing_path(candidates: Sequence[Path]) -> Path:
    return next((path for path in candidates if path.exists()), candidates[0])


DEFAULT_IMAGE_10 = _first_existing_path(
    (
        REPO_ROOT / "data" / "IM_D03_PT_5k_Augmented",
        REPO_ROOT / "IM_D03_PT_5k_Augmented",
        REPO_ROOT / "data" / "IM_D03_PT_5K_Augmented",
    )
)
DEFAULT_IMAGE_15 = _first_existing_path(
    (
        REPO_ROOT / "data" / "5k",
        REPO_ROOT / "data" / "IM_D03_PT_5K",
        REPO_ROOT / "5k",
        REPO_ROOT / "IM_D03_PT_5K",
    )
)
DEFAULT_IMAGE_788 = (
    REPO_ROOT / "scripts" / "qwn3.5-27b_788_silver_plate_bend" / "simens_7feats"
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


# Dataset exports omit a task entirely when there is no positive instance.
# This map recovers the source-level exhaustive annotation scope.
TASK_SCOPE: dict[str, tuple[str, ...]] = {
    "round_hole_detection": ("Round Hole", "Round Hole Group"),
    "slotted_hole_detection": ("Slotted Hole", "Slotted Hole Group"),
    "rectangular_hole_detection": ("Rectangular Hole", "Rectangular Hole Group"),
    "threaded_hole_detection": ("Threaded Hole", "Threaded Hole Group"),
    "pin_hole_detection": ("Pin Hole", "Pin Hole Group"),
    "counterbore_hole_detection": ("Counterbore Hole", "Counterbore Hole Group"),
    "fillet_detection": ("Fillet", "Fillet Group"),
    "chamfer_detection": ("Chamfer", "Chamfer Group"),
    "threaded_shaft_detection": ("Threaded Shaft",),
    "bending_detection": ("Bending",),
    "silver_plating_detection": ("Silver Plating",),
}

# These are dataset-contract scopes, not merely the labels that happened to be
# positive.  In particular, identically named tasks do not imply identical
# Group coverage across exports (e.g. Siemens 788 has Fillet but not Fillet
# Group).  They are exposed through CLI overrides for future corrected exports.
DEFAULT_SCOPE_10 = (
    "Threaded Hole",
    "Threaded Hole Group",
    "Round Hole",
    "Round Hole Group",
    "Slotted Hole",
    "Slotted Hole Group",
    "Rectangular Hole",
    "Rectangular Hole Group",
    "Fillet",
    "Fillet Group",
)
DEFAULT_SCOPE_15 = (
    "Threaded Hole",
    "Threaded Hole Group",
    "Round Hole",
    "Round Hole Group",
    "Pin Hole",
    "Pin Hole Group",
    "Counterbore Hole",
    "Counterbore Hole Group",
    "Rectangular Hole",
    "Fillet",
    "Fillet Group",
    "Chamfer",
    "Chamfer Group",
    "Threaded Shaft",
)
DEFAULT_SCOPE_788 = (
    "Round Hole",
    "Rectangular Hole",
    "Threaded Hole",
    "Slotted Hole",
    "Round Hole Group",
    "Rectangular Hole Group",
    "Slotted Hole Group",
    "Fillet",
    "Bending",
    "Silver Plating",
)
DEFAULT_VIEW_SCOPE_10: tuple[str, ...] = ()
DEFAULT_DOCUMENT_SCOPE_10: tuple[str, ...] = ()
DEFAULT_VIEW_SCOPE_15 = (
    "Orthographic Projection - Front View",
    "Orthographic Projection - Top View",
    "Orthographic Projection - Left View",
    "Orthographic Projection - Right View",
    "Isometric View",
    "Auxiliary View",
    "Section View",
)
DEFAULT_DOCUMENT_SCOPE_15 = tuple(DOCUMENT_CATEGORIES)
DEFAULT_VIEW_SCOPE_788 = tuple(VIEW_CATEGORIES)
DEFAULT_DOCUMENT_SCOPE_788 = ("Title Block", "Notes")


@dataclass(frozen=True)
class SourceSpec:
    """Paths and domain semantics for one annotation export."""

    name: str
    json_path: Path
    image_dir: Path | None
    deploy_dir: str
    projection_method: str
    declared_feature_scope: tuple[str, ...] | None = None
    declared_view_scope: tuple[str, ...] = ()
    declared_document_scope: tuple[str, ...] = ()


@dataclass
class SourcePayload:
    """Loaded export plus its source-level exhaustive feature scope."""

    spec: SourceSpec
    items: list[dict[str, Any]]
    feature_scope: set[str]
    task_names: set[str]
    unknown_labels: Counter[str]


@dataclass
class MergedItem:
    """Canonical annotations for one exact image filename."""

    filename: str
    sources: list[str] = field(default_factory=list)
    source_items: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_scopes: dict[str, list[str]] = field(default_factory=dict)
    source_view_scopes: dict[str, list[str]] = field(default_factory=dict)
    source_document_scopes: dict[str, list[str]] = field(default_factory=dict)
    annotations: list[dict[str, Any]] = field(default_factory=list)
    duplicate_annotations: int = 0
    annotation_conflicts: list[dict[str, Any]] = field(default_factory=list)
    unknown_labels: set[str] = field(default_factory=set)

    @property
    def annotation_scope(self) -> list[str]:
        scope: set[str] = set()
        for values in self.source_scopes.values():
            scope.update(values)
        return [category for category in FEATURE_CATEGORIES if category in scope]

    @property
    def view_scope(self) -> list[str]:
        scope: set[str] = set()
        for values in self.source_view_scopes.values():
            scope.update(values)
        return [category for category in VIEW_CATEGORIES if category in scope]

    @property
    def document_scope(self) -> list[str]:
        scope: set[str] = set()
        for values in self.source_document_scopes.values():
            scope.update(values)
        return [category for category in DOCUMENT_CATEGORIES if category in scope]

    @property
    def family_id(self) -> str:
        return str(infer_family_id(self.filename))

    @property
    def is_augmented(self) -> bool:
        return bool(is_augmented_name(self.filename))


@dataclass(frozen=True)
class ImageInfo:
    """Resolved local/deployment image and its dimensions."""

    local_path: Path | None
    deploy_path: str
    width: int | None
    height: int | None
    selected_source: str
    status: str
    error: str = ""

    @property
    def usable(self) -> bool:
        return bool(self.width and self.height and self.width > 0 and self.height > 0)


def _json_dump(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json_dump(row) + "\n")
            count += 1
    return count


def _ms_swift_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep training files on the strict ms-swift messages/images contract."""

    return {"messages": record["messages"], "images": record["images"]}


def _ground_truth_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Remove prompts while preserving stable IDs and structured evaluation GT."""

    return {
        key: record.get(key)
        for key in (
            "sample_id",
            "images",
            "stage1",
            "stage2",
            "stage3",
            "context",
            "metadata",
        )
        if key in record
    }


def _load_json_items(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        for key in ("data", "items", "annotations", "records"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(
            f"Expected a JSON list in {path}, got {type(payload).__name__}"
        )
    result: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: item {index} is not a JSON object")
        result.append(item)
    return result


def _iter_task_values(
    item: Mapping[str, Any],
) -> Iterator[tuple[str, Mapping[str, Any]]]:
    tasks = item.get("tasks", [])
    if not isinstance(tasks, list):
        return
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        task_name = str(task.get("task_name", ""))
        values = task.get("task_values", [])
        if not isinstance(values, list):
            continue
        for task_value in values:
            if isinstance(task_value, Mapping):
                yield task_name, task_value


def _category_of(task_value: Mapping[str, Any]) -> str:
    value = task_value.get("value", {})
    if not isinstance(value, Mapping):
        return ""
    return str(value.get("label", "")).strip()


def _safe_normalize_size(value: Any, category: str) -> str:
    raw = "" if value is None else str(value).strip()
    try:
        normalized = normalize_size(raw, category=category)
    except TypeError:
        normalized = normalize_size(raw, category)
    except Exception:
        normalized = raw
    return raw if normalized is None else str(normalized).strip()


def _valid_bbox(raw_bbox: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(raw_bbox, Sequence) or isinstance(raw_bbox, (str, bytes)):
        return None
    if len(raw_bbox) != 4:
        return None
    try:
        values = [float(v) for v in raw_bbox]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    x1, y1, x2, y2 = values
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _number_for_json(value: float) -> int | float:
    rounded = round(value)
    return int(rounded) if abs(value - rounded) < 1e-9 else round(value, 4)


def _canonical_annotation(
    source_name: str,
    task_name: str,
    task_value: Mapping[str, Any],
) -> dict[str, Any] | None:
    category = _category_of(task_value)
    if not category:
        return None
    bbox = _valid_bbox(task_value.get("bbox"))
    if bbox is None:
        return None
    value = task_value.get("value", {})
    size = value.get("size", "") if isinstance(value, Mapping) else ""
    return {
        "category": category,
        "size": _safe_normalize_size(size, category),
        "bbox": [_number_for_json(v) for v in bbox],
        "sources": [source_name],
        "task_names": [task_name] if task_name else [],
    }


def _bbox_key(bbox: Sequence[int | float]) -> tuple[float, float, float, float]:
    return tuple(round(float(value), 3) for value in bbox)  # type: ignore[return-value]


def _prefer_size(left: str, right: str) -> str:
    """Choose the more informative size deterministically on duplicate boxes."""

    if not left:
        return right
    if not right:
        return left
    if left == right:
        return left

    # Prefer values containing explicit engineering symbols/tolerance/depth,
    # then the longer value.  Lexical order makes reruns deterministic.
    def rank(value: str) -> tuple[int, int, str]:
        markers = sum(
            token in value.upper() for token in ("Ø", "M", "R", "DP", "H7", "°", "X")
        )
        return markers, len(value), value

    return max((left, right), key=rank)


def load_source(spec: SourceSpec) -> SourcePayload:
    items = _load_json_items(spec.json_path)
    feature_set = set(FEATURE_CATEGORIES)
    known_set = feature_set | set(VIEW_CATEGORIES) | set(DOCUMENT_CATEGORIES)
    task_names: set[str] = set()
    positive_feature_labels: set[str] = set()
    unknown_labels: Counter[str] = Counter()
    for item in items:
        for task_name, task_value in _iter_task_values(item):
            if task_name:
                task_names.add(task_name)
            label = _category_of(task_value)
            if label in feature_set:
                positive_feature_labels.add(label)
            elif label and label not in known_set:
                unknown_labels[label] += 1

    scope = set(positive_feature_labels)
    if spec.declared_feature_scope is not None:
        scope.update(
            category
            for category in spec.declared_feature_scope
            if category in feature_set
        )
    else:
        # Auto mode is useful for new exports, but explicit scope is preferred
        # because task names alone cannot reveal whether Group was exhaustive.
        for task_name in task_names:
            scope.update(
                category
                for category in TASK_SCOPE.get(task_name, ())
                if category in feature_set
            )
    return SourcePayload(
        spec=spec,
        items=items,
        feature_scope=scope,
        task_names=task_names,
        unknown_labels=unknown_labels,
    )


def merge_sources(payloads: Sequence[SourcePayload]) -> list[MergedItem]:
    """Merge exact filenames, complementary scopes, and duplicate boxes."""

    known_categories = (
        set(FEATURE_CATEGORIES) | set(VIEW_CATEGORIES) | set(DOCUMENT_CATEGORIES)
    )
    merged: dict[str, MergedItem] = {}
    dedup_index: dict[str, dict[tuple[str, tuple[float, ...]], int]] = defaultdict(dict)

    for payload in payloads:
        source = payload.spec.name
        for raw_item in payload.items:
            filename_value = raw_item.get("dataitem_name") or raw_item.get("filename")
            if not filename_value:
                continue
            filename = Path(str(filename_value)).name
            target = merged.setdefault(filename, MergedItem(filename=filename))
            if source not in target.sources:
                target.sources.append(source)
            target.source_items[source] = dict(raw_item)
            target.source_scopes[source] = sorted(payload.feature_scope)
            target.source_view_scopes[source] = list(payload.spec.declared_view_scope)
            target.source_document_scopes[source] = list(
                payload.spec.declared_document_scope
            )

            for task_name, task_value in _iter_task_values(raw_item):
                annotation = _canonical_annotation(source, task_name, task_value)
                if annotation is None:
                    continue
                category = annotation["category"]
                if category not in known_categories:
                    target.unknown_labels.add(category)
                    continue
                key = (category, _bbox_key(annotation["bbox"]))
                existing_index = dedup_index[filename].get(key)
                if existing_index is None:
                    dedup_index[filename][key] = len(target.annotations)
                    target.annotations.append(annotation)
                    continue

                target.duplicate_annotations += 1
                existing = target.annotations[existing_index]
                for source_name in annotation["sources"]:
                    if source_name not in existing["sources"]:
                        existing["sources"].append(source_name)
                for existing_task in annotation["task_names"]:
                    if existing_task not in existing["task_names"]:
                        existing["task_names"].append(existing_task)
                left_size = str(existing.get("size", ""))
                right_size = str(annotation.get("size", ""))
                if left_size and right_size and left_size != right_size:
                    target.annotation_conflicts.append(
                        {
                            "category": category,
                            "bbox": annotation["bbox"],
                            "sizes": sorted({left_size, right_size}),
                        }
                    )
                existing["size"] = _prefer_size(left_size, right_size)

    for item in merged.values():
        item.sources.sort()
        for annotation in item.annotations:
            annotation["sources"].sort()
            annotation["task_names"].sort()
        item.annotations.sort(
            key=lambda ann: (
                float(ann["bbox"][1]),
                float(ann["bbox"][0]),
                str(ann["category"]),
                str(ann.get("size", "")),
            )
        )
    return sorted(merged.values(), key=lambda item: item.filename)


class ImageResolver:
    """Resolve local images lazily while preserving deployment paths."""

    def __init__(self, specs: Sequence[SourceSpec], recursive: bool = False):
        self.specs = {spec.name: spec for spec in specs}
        self.recursive = recursive
        self._indices: dict[Path, dict[str, Path]] = {}
        self._index_lock = Lock()

    def _index(self, image_dir: Path) -> dict[str, Path]:
        # Several dimension workers may miss a direct path at the same time.
        # Build each fallback directory index once instead of multiplying the
        # expensive CephFS metadata scan.
        with self._index_lock:
            if image_dir in self._indices:
                return self._indices[image_dir]
            index: dict[str, Path] = {}
            if image_dir.is_dir():
                iterator = (
                    image_dir.rglob("*") if self.recursive else image_dir.iterdir()
                )
                for path in iterator:
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                        index.setdefault(path.name, path)
            self._indices[image_dir] = index
            return index

    @staticmethod
    def _metadata_dimensions(raw_item: Mapping[str, Any]) -> tuple[int, int] | None:
        candidates: list[Mapping[str, Any]] = [raw_item]
        for key in ("metadata", "image", "image_metadata"):
            nested = raw_item.get(key)
            if isinstance(nested, Mapping):
                candidates.append(nested)
        key_pairs = (
            ("width", "height"),
            ("image_width", "image_height"),
            ("original_width", "original_height"),
        )
        for mapping in candidates:
            for width_key, height_key in key_pairs:
                try:
                    width = int(mapping.get(width_key, 0))
                    height = int(mapping.get(height_key, 0))
                except (TypeError, ValueError):
                    continue
                if width > 0 and height > 0:
                    return width, height
        return None

    @staticmethod
    def _read_dimensions(path: Path) -> tuple[int, int]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Pillow is required to read CAD image dimensions"
            ) from exc
        with Image.open(path) as image:
            width, height = image.size
        return int(width), int(height)

    def resolve(self, item: MergedItem) -> ImageInfo:
        errors: list[str] = []
        # Prefer an existing local file, retaining source order from the merge.
        for source_name in item.sources:
            spec = self.specs[source_name]
            if spec.image_dir is None:
                continue
            local_path = spec.image_dir / item.filename
            if not local_path.is_file():
                local_path = self._index(spec.image_dir).get(item.filename, local_path)
            if not local_path.is_file():
                continue
            try:
                width, height = self._read_dimensions(local_path)
            except Exception as exc:  # retain audit trail and try another source
                errors.append(f"{local_path}: {exc}")
                continue
            return ImageInfo(
                local_path=local_path.resolve(),
                deploy_path=_join_deploy(spec.deploy_dir, item.filename),
                width=width,
                height=height,
                selected_source=source_name,
                status="local",
                error="; ".join(errors),
            )

        # Metadata dimensions allow generation on a preprocessing host without
        # the pixels.  The deploy path still points to the selected source.
        for source_name in item.sources:
            dimensions = self._metadata_dimensions(item.source_items[source_name])
            if dimensions is None:
                continue
            spec = self.specs[source_name]
            return ImageInfo(
                local_path=None,
                deploy_path=_join_deploy(spec.deploy_dir, item.filename),
                width=dimensions[0],
                height=dimensions[1],
                selected_source=source_name,
                status="metadata_only",
                error="; ".join(errors),
            )

        source_name = item.sources[0]
        spec = self.specs[source_name]
        return ImageInfo(
            local_path=None,
            deploy_path=_join_deploy(spec.deploy_dir, item.filename),
            width=None,
            height=None,
            selected_source=source_name,
            status="missing",
            error="; ".join(errors) or "image missing and no width/height metadata",
        )


def _join_deploy(directory: str, filename: str) -> str:
    base = str(directory).rstrip("/")
    return f"{base}/{filename}" if base else filename


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def normalize_bbox(
    raw_bbox: Sequence[int | float],
    width: int,
    height: int,
    *,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    crop_width: float | None = None,
    crop_height: float | None = None,
) -> list[int]:
    """Map a raw/global bbox to clamped 0..1000 coordinates."""

    x1, y1, x2, y2 = (float(value) for value in raw_bbox)
    target_width = float(crop_width if crop_width is not None else width)
    target_height = float(crop_height if crop_height is not None else height)
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Image/crop dimensions must be positive")
    values = [
        round(_clamp((x1 - origin_x) / target_width * 1000.0, 0.0, 1000.0)),
        round(_clamp((y1 - origin_y) / target_height * 1000.0, 0.0, 1000.0)),
        round(_clamp((x2 - origin_x) / target_width * 1000.0, 0.0, 1000.0)),
        round(_clamp((y2 - origin_y) / target_height * 1000.0, 0.0, 1000.0)),
    ]
    result = [int(value) for value in values]
    if result[2] <= result[0]:
        result[2] = min(1000, result[0] + 1)
        if result[2] <= result[0]:
            result[0] = max(0, result[2] - 1)
    if result[3] <= result[1]:
        result[3] = min(1000, result[1] + 1)
        if result[3] <= result[1]:
            result[1] = max(0, result[3] - 1)
    return result


def _normalized_annotations(
    item: MergedItem, image: ImageInfo
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not image.usable:
        return [], [], []
    features: list[dict[str, Any]] = []
    views: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    feature_set = set(FEATURE_CATEGORIES)
    view_set = set(VIEW_CATEGORIES)
    document_set = set(DOCUMENT_CATEGORIES)
    assert image.width is not None and image.height is not None
    for annotation in item.annotations:
        category = str(annotation["category"])
        bbox = normalize_bbox(annotation["bbox"], image.width, image.height)
        if category in feature_set:
            features.append(
                {
                    "category": category,
                    "size": str(annotation.get("size", "")),
                    "bbox": bbox,
                }
            )
        elif category in view_set:
            views.append(
                {"category": category, "bbox": bbox, "raw_bbox": annotation["bbox"]}
            )
        elif category in document_set:
            documents.append({"category": category, "bbox": bbox})

    def sort_key(value: Mapping[str, Any]) -> tuple[Any, Any, Any]:
        return value["bbox"][1], value["bbox"][0], value["category"]

    features.sort(key=sort_key)
    views.sort(key=sort_key)
    documents.sort(key=sort_key)
    return features, views, documents


def _projection_method(item: MergedItem, specs: Mapping[str, SourceSpec]) -> str:
    methods = {
        specs[source].projection_method
        for source in item.sources
        if specs[source].projection_method != "unknown"
    }
    return next(iter(methods)) if len(methods) == 1 else "unknown"


def build_stage_targets(
    item: MergedItem,
    image: ImageInfo,
    specs: Mapping[str, SourceSpec],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    features, views, documents = _normalized_annotations(item, image)
    stage3 = {"feature_scope": item.annotation_scope, "features": features}
    if not views or item.is_augmented:
        return None, None, stage3

    stage1_regions: list[dict[str, Any]] = []
    stage2_views: list[dict[str, Any]] = []
    for index, view in enumerate(views, start=1):
        region_id = f"r{index:03d}"
        stage1_regions.append(
            {"region_id": region_id, "category": "View Region", "bbox": view["bbox"]}
        )
        stage2_views.append(
            {
                "region_id": region_id,
                "category": view["category"],
                "bbox": view["bbox"],
            }
        )
    for index, document in enumerate(documents, start=1):
        stage1_regions.append(
            {
                "region_id": f"d{index:03d}",
                "category": document["category"],
                "bbox": document["bbox"],
            }
        )
    stage1_regions.sort(
        key=lambda value: (value["bbox"][1], value["bbox"][0], value["region_id"])
    )
    stage1 = {"regions": stage1_regions}
    stage2 = {
        "projection_method": _projection_method(item, specs),
        "views": stage2_views,
    }
    return stage1, stage2, stage3


def _call_step3_prompt(
    feature_scope: Sequence[str],
    *,
    region_id: str | None = None,
    crop_context: bool,
) -> str:
    try:
        return str(
            step3_prompt(
                feature_scope=list(feature_scope),
                region_id=region_id,
                crop_context=crop_context,
            )
        )
    except TypeError:
        return str(step3_prompt(list(feature_scope)))


def _layout_scope_suffix(item: MergedItem, stage: int) -> str:
    """Condition layout supervision on the source's exhaustive class scope."""

    scope = {
        "view_scope": item.view_scope,
        "document_scope": item.document_scope,
    }
    if stage == 1:
        instruction = (
            "本训练样本仅对下列layout_scope中的语义类别做了穷尽标注。"
            "未列入scope的类别不是负样本；只输出已标注范围内对应的View Region/文档区域。"
        )
    else:
        instruction = (
            "本训练样本仅对下列view_scope中的视图语义做了穷尽标注。"
            "不得把未列入scope的类别解释为负样本。"
        )
    return f"\n{instruction}\nlayout_scope={_json_dump(scope)}"


def build_three_stage_record(
    item: MergedItem,
    image: ImageInfo,
    split: str,
    specs: Mapping[str, SourceSpec],
) -> dict[str, Any]:
    stage1, stage2, stage3 = build_stage_targets(item, image, specs)
    metadata = {
        "filename": item.filename,
        "family_id": item.family_id,
        "split": split,
        "sources": item.sources,
        "annotation_scope": item.annotation_scope,
        "view_scope": item.view_scope,
        "document_scope": item.document_scope,
        "is_augmented": item.is_augmented,
        "conversation_mode": "three_stage"
        if stage1 is not None
        else "scoped_feature_only",
    }
    if stage1 is not None and stage2 is not None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"<image>{step1_prompt(document_scope=item.document_scope, view_scope=item.view_scope)}"
                    f"{_layout_scope_suffix(item, 1)}"
                ),
            },
            {"role": "assistant", "content": _json_dump(stage1)},
            {
                "role": "user",
                "content": str(step2_prompt(view_scope=item.view_scope)),
            },
            {"role": "assistant", "content": _json_dump(stage2)},
            {
                "role": "user",
                "content": _call_step3_prompt(
                    stage3["feature_scope"], crop_context=False
                ),
            },
            {"role": "assistant", "content": _json_dump(stage3)},
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"<image>{_call_step3_prompt(stage3['feature_scope'], crop_context=False)}",
            },
            {"role": "assistant", "content": _json_dump(stage3)},
        ]
    return {
        "sample_id": f"full::{item.filename}",
        "messages": messages,
        "images": [image.deploy_path],
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "metadata": metadata,
    }


def _family_labels(items: Sequence[MergedItem]) -> set[str]:
    known = set(FEATURE_CATEGORIES) | set(VIEW_CATEGORIES) | set(DOCUMENT_CATEGORIES)
    return {
        str(annotation["category"])
        for item in items
        for annotation in item.annotations
        if str(annotation["category"]) in known
    }


def split_by_family(
    items: Sequence[MergedItem],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    """Greedy multi-label family split with deterministic seeded tie-breaking."""

    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    total_ratio = sum(ratios.values())
    if any(value < 0 for value in ratios.values()) or total_ratio <= 0:
        raise ValueError("Split ratios must be non-negative and have a positive sum")
    if not math.isclose(total_ratio, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.8f}")

    families: dict[str, list[MergedItem]] = defaultdict(list)
    for item in items:
        families[item.family_id].append(item)
    family_labels = {
        family: _family_labels(group) for family, group in families.items()
    }
    label_frequency: Counter[str] = Counter(
        label for labels in family_labels.values() for label in labels
    )
    original_count = {
        family: max(1, sum(not item.is_augmented for item in group))
        for family, group in families.items()
    }

    rng = random.Random(seed)
    tie_break = {family: rng.random() for family in families}
    ordered = sorted(
        families,
        key=lambda family: (
            -sum(1.0 / label_frequency[label] for label in family_labels[family]),
            -len(family_labels[family]),
            -original_count[family],
            tie_break[family],
            family,
        ),
    )

    total_original = sum(original_count.values())
    target_count = {
        split: max(ratio * total_original, 1e-9) for split, ratio in ratios.items()
    }
    target_labels = {
        split: {
            label: max(ratio * frequency, 1e-9)
            for label, frequency in label_frequency.items()
        }
        for split, ratio in ratios.items()
    }
    current_count: Counter[str] = Counter()
    current_labels: dict[str, Counter[str]] = {split: Counter() for split in ratios}
    assignment: dict[str, str] = {}

    active_splits = [split for split, ratio in ratios.items() if ratio > 0]
    # Seed each requested split when possible, preferring the largest-ratio
    # split for the rarest family and leaving no empty validation/test split.
    seed_order = sorted(active_splits, key=lambda split: (-ratios[split], split))
    for family, split in zip(ordered[: len(seed_order)], seed_order):
        assignment[family] = split
        current_count[split] += original_count[family]
        current_labels[split].update(family_labels[family])

    for family in ordered[len(seed_order) :]:
        labels = family_labels[family]
        family_size = original_count[family]
        candidates: list[tuple[float, float, str]] = []
        for split in active_splits:
            count_deficit = (
                max(target_count[split] - current_count[split], 0.0)
                / target_count[split]
            )
            overflow = (
                max(current_count[split] + family_size - target_count[split], 0.0)
                / target_count[split]
            )
            if labels:
                label_deficit = sum(
                    max(target_labels[split][label] - current_labels[split][label], 0.0)
                    / target_labels[split][label]
                    for label in labels
                ) / len(labels)
            else:
                label_deficit = 0.0
            score = 0.60 * count_deficit + 0.40 * label_deficit - 1.75 * overflow
            candidates.append((score, rng.random(), split))
        chosen = max(candidates)[2]
        assignment[family] = chosen
        current_count[chosen] += family_size
        current_labels[chosen].update(labels)
    return assignment


def _raw_view_annotations(item: MergedItem) -> list[dict[str, Any]]:
    view_set = set(VIEW_CATEGORIES)
    result = [ann for ann in item.annotations if ann["category"] in view_set]
    return sorted(
        result,
        key=lambda ann: (float(ann["bbox"][1]), float(ann["bbox"][0]), ann["category"]),
    )


def _raw_feature_annotations(item: MergedItem) -> list[dict[str, Any]]:
    feature_set = set(FEATURE_CATEGORIES)
    return [ann for ann in item.annotations if ann["category"] in feature_set]


def _intersection_area(
    left: Sequence[int | float], right: Sequence[int | float]
) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_area(bbox: Sequence[int | float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(
        0.0, float(bbox[3]) - float(bbox[1])
    )


def assign_features_to_views(
    features: Sequence[dict[str, Any]],
    views: Sequence[dict[str, Any]],
) -> tuple[dict[int, list[dict[str, Any]]], int]:
    """Assign each feature to at most one best-overlapping semantic view."""

    assigned: dict[int, list[dict[str, Any]]] = defaultdict(list)
    unassigned = 0
    for feature in features:
        feature_bbox = feature["bbox"]
        area = max(_bbox_area(feature_bbox), 1e-9)
        center_x = (float(feature_bbox[0]) + float(feature_bbox[2])) / 2.0
        center_y = (float(feature_bbox[1]) + float(feature_bbox[3])) / 2.0
        candidates: list[tuple[float, int]] = []
        for index, view in enumerate(views):
            view_bbox = view["bbox"]
            overlap = _intersection_area(feature_bbox, view_bbox) / area
            center_inside = float(view_bbox[0]) <= center_x <= float(
                view_bbox[2]
            ) and float(view_bbox[1]) <= center_y <= float(view_bbox[3])
            score = overlap + (1.0 if center_inside else 0.0)
            if center_inside or overlap >= 0.25:
                candidates.append((score, -index))
        if not candidates:
            unassigned += 1
            continue
        best_index = -max(candidates)[1]
        assigned[best_index].append(feature)
    return assigned, unassigned


def _expanded_crop_bbox(
    bbox: Sequence[int | float],
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    left = max(0, int(math.floor(x1 - pad_x)))
    top = max(0, int(math.floor(y1 - pad_y)))
    right = min(image_width, int(math.ceil(x2 + pad_x)))
    bottom = min(image_height, int(math.ceil(y2 + pad_y)))
    if right <= left:
        right = min(image_width, left + 1)
    if bottom <= top:
        bottom = min(image_height, top + 1)
    return left, top, right, bottom


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned[:120] or "image"


def _crop_filename(item: MergedItem, region_id: str, crop_bbox: Sequence[int]) -> str:
    digest = hashlib.sha1(
        f"{item.filename}|{region_id}|{','.join(map(str, crop_bbox))}".encode("utf-8")
    ).hexdigest()[:10]
    return f"{_safe_stem(Path(item.filename).stem)}__{region_id}__{digest}.png"


def _materialize_crop(source: Path, destination: Path, bbox: Sequence[int]) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required with --materialize-crops") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        crop = image.crop(tuple(int(value) for value in bbox))
        if crop.mode not in {"RGB", "RGBA", "L"}:
            crop = crop.convert("RGB")
        crop.save(destination, format="PNG", optimize=True)


def build_crop_records(
    train_items: Sequence[MergedItem],
    image_infos: Mapping[str, ImageInfo],
    specs: Mapping[str, SourceSpec],
    crop_dir: Path,
    crop_deploy_dir: str,
    padding: float,
    materialize: bool,
    max_empty_crops_per_image: int,
    crop_workers: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    candidates: list[tuple[dict[str, Any], dict[str, Any], Future[None] | None]] = []
    crop_pool = (
        ThreadPoolExecutor(max_workers=crop_workers, thread_name_prefix="cad-crop")
        if materialize
        else None
    )

    for item in train_items:
        if item.is_augmented:
            continue
        image = image_infos[item.filename]
        if not image.usable:
            continue
        assert image.width is not None and image.height is not None
        views = _raw_view_annotations(item)
        if not views:
            continue
        features = _raw_feature_annotations(item)
        assigned, unassigned = assign_features_to_views(features, views)
        counters["unassigned_features"] += unassigned
        projection = _projection_method(item, specs)
        empty_crops_emitted = 0

        for view_index, view in enumerate(views):
            region_id = f"r{view_index + 1:03d}"
            view_features = assigned.get(view_index, [])
            if not view_features:
                if max_empty_crops_per_image == 0 or (
                    max_empty_crops_per_image > 0
                    and empty_crops_emitted >= max_empty_crops_per_image
                ):
                    counters["empty_crops_skipped"] += 1
                    continue
                empty_crops_emitted += 1
            crop_bbox = _expanded_crop_bbox(
                view["bbox"], image.width, image.height, padding
            )
            crop_width = crop_bbox[2] - crop_bbox[0]
            crop_height = crop_bbox[3] - crop_bbox[1]
            crop_name = _crop_filename(item, region_id, crop_bbox)
            local_crop = crop_dir / crop_name
            deploy_crop = _join_deploy(crop_deploy_dir, crop_name)
            status = "planned"
            error = ""
            crop_future: Future[None] | None = None
            if materialize:
                if image.local_path is None:
                    status = "skipped_missing_source"
                    error = "local source image is unavailable"
                else:
                    assert crop_pool is not None
                    status = "materializing"
                    crop_future = crop_pool.submit(
                        _materialize_crop, image.local_path, local_crop, crop_bbox
                    )

            crop_features = []
            for feature in sorted(
                view_features,
                key=lambda ann: (
                    float(ann["bbox"][1]),
                    float(ann["bbox"][0]),
                    ann["category"],
                ),
            ):
                crop_features.append(
                    {
                        "category": feature["category"],
                        "size": str(feature.get("size", "")),
                        "bbox": normalize_bbox(
                            feature["bbox"],
                            image.width,
                            image.height,
                            origin_x=crop_bbox[0],
                            origin_y=crop_bbox[1],
                            crop_width=crop_width,
                            crop_height=crop_height,
                        ),
                    }
                )
            stage3 = {"feature_scope": item.annotation_scope, "features": crop_features}
            context = {
                "projection_method": projection,
                "view": {"region_id": region_id, "category": view["category"]},
            }
            context_text = _json_dump(context)
            prompt = _call_step3_prompt(
                stage3["feature_scope"], region_id=region_id, crop_context=True
            )
            prompt = f"{prompt}\n已知整图阶段上下文：{context_text}"
            record = {
                "sample_id": f"crop::{item.filename}::{region_id}",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"<image>{prompt}"},
                    {"role": "assistant", "content": _json_dump(stage3)},
                ],
                "images": [deploy_crop],
                "stage1": None,
                "stage2": None,
                "stage3": stage3,
                "context": context,
                "metadata": {
                    "filename": item.filename,
                    "family_id": item.family_id,
                    "split": "train",
                    "sources": item.sources,
                    "annotation_scope": item.annotation_scope,
                    "conversation_mode": "view_crop_feature",
                    "region_id": region_id,
                    "view_category": view["category"],
                    "global_crop_bbox_pixels": list(crop_bbox),
                    "crop_materialization": status,
                },
            }
            crop_row = {
                "sample_id": record["sample_id"],
                "source_filename": item.filename,
                "source_image_path": str(image.local_path)
                if image.local_path
                else None,
                "crop_filename": crop_name,
                "crop_local_path": str(local_crop.resolve()),
                "crop_deploy_path": deploy_crop,
                "global_crop_bbox_pixels": list(crop_bbox),
                "region_id": region_id,
                "view_category": view["category"],
                "feature_count": len(crop_features),
                "annotation_scope": item.annotation_scope,
                "status": status,
                "error": error,
            }
            candidates.append((record, crop_row, crop_future))

    if crop_pool is not None:
        crop_pool.shutdown(wait=True)

    # Futures are consumed in deterministic source/view order.  Concurrency
    # changes only wall-clock time, never JSONL ordering or sample IDs.
    for record, crop_row, crop_future in candidates:
        status = str(crop_row["status"])
        error = str(crop_row["error"])
        if crop_future is not None:
            try:
                crop_future.result()
                status = "materialized"
                counters["crops_materialized"] += 1
            except Exception as exc:
                status = "materialize_error"
                error = str(exc)
                counters["crop_materialize_failed"] += 1
        elif materialize:
            counters["crop_materialize_failed"] += 1

        record["metadata"]["crop_materialization"] = status
        crop_row["status"] = status
        crop_row["error"] = error
        manifest.append(crop_row)
        if materialize and status != "materialized":
            # Do not emit a training record that points at a crop known to
            # be absent.  The audit manifest retains the failure.
            continue
        records.append(record)
        counters["crop_records"] += 1
        if not record["stage3"]["features"]:
            counters["empty_crop_records"] += 1
    return records, manifest, dict(counters)


def _serializable_merged(item: MergedItem) -> dict[str, Any]:
    return {
        "dataitem_name": item.filename,
        "family_id": item.family_id,
        "is_augmented": item.is_augmented,
        "sources": item.sources,
        "annotation_scope": item.annotation_scope,
        "view_scope": item.view_scope,
        "document_scope": item.document_scope,
        "source_scopes": item.source_scopes,
        "source_view_scopes": item.source_view_scopes,
        "source_document_scopes": item.source_document_scopes,
        "annotations": item.annotations,
        "duplicate_annotations_removed": item.duplicate_annotations,
        "annotation_conflicts": item.annotation_conflicts,
        "unknown_labels": sorted(item.unknown_labels),
    }


def _label_distribution(items: Sequence[MergedItem]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in items:
        counts.update(str(annotation["category"]) for annotation in item.annotations)
    return dict(sorted(counts.items()))


def _scope_distribution(items: Sequence[MergedItem]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in items:
        counts.update(item.annotation_scope)
    return dict(sorted(counts.items()))


def _layout_scope_distribution(items: Sequence[MergedItem]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in items:
        counts.update(item.view_scope)
        counts.update(item.document_scope)
    return dict(sorted(counts.items()))


def _parse_projection(value: str) -> str:
    aliases = {
        "first": "first_angle",
        "first-angle": "first_angle",
        "first_angle": "first_angle",
        "third": "third_angle",
        "third-angle": "third_angle",
        "third_angle": "third_angle",
        "unknown": "unknown",
    }
    normalized = aliases.get(value.strip().lower())
    if normalized is None:
        raise argparse.ArgumentTypeError(
            "projection must be first_angle, third_angle, or unknown"
        )
    return normalized


def _optional_path(value: str) -> Path | None:
    if value.strip().lower() in {"", "none", "null", "-"}:
        return None
    return Path(value).expanduser()


def _parse_scope(value: str) -> tuple[str, ...] | None:
    """Parse comma-separated exhaustive feature categories or ``auto``."""

    if value.strip().lower() == "auto":
        return None
    categories = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(categories) - set(FEATURE_CATEGORIES))
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown feature categories in scope: " + ", ".join(unknown)
        )
    return categories


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge CAD labels and build leakage-safe three-stage ms-swift JSONL"
    )
    parser.add_argument("--json-10", type=Path, default=DEFAULT_JSON_10)
    parser.add_argument("--json-15", type=Path, default=DEFAULT_JSON_15)
    parser.add_argument("--json-788", type=Path, default=DEFAULT_JSON_788)
    parser.add_argument("--image-dir-10", type=_optional_path, default=DEFAULT_IMAGE_10)
    parser.add_argument("--image-dir-15", type=_optional_path, default=DEFAULT_IMAGE_15)
    parser.add_argument(
        "--image-dir-788", type=_optional_path, default=DEFAULT_IMAGE_788
    )
    parser.add_argument(
        "--deploy-dir-10", default="/workspace/data/IM_D03_PT_5k_Augmented"
    )
    parser.add_argument("--deploy-dir-15", default="/workspace/data/IM_D03_PT_5K")
    parser.add_argument("--deploy-dir-788", default="/workspace/data/simens_7feats")
    parser.add_argument("--projection-10", type=_parse_projection, default="unknown")
    parser.add_argument("--projection-15", type=_parse_projection, default="unknown")
    parser.add_argument(
        "--projection-788", type=_parse_projection, default="first_angle"
    )
    parser.add_argument(
        "--scope-10",
        type=_parse_scope,
        default=DEFAULT_SCOPE_10,
        help="Comma-separated exhaustive feature scope, or 'auto'",
    )
    parser.add_argument(
        "--scope-15",
        type=_parse_scope,
        default=DEFAULT_SCOPE_15,
        help="Comma-separated exhaustive feature scope, or 'auto'",
    )
    parser.add_argument(
        "--scope-788",
        type=_parse_scope,
        default=DEFAULT_SCOPE_788,
        help="Comma-separated exhaustive feature scope, or 'auto'",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "dataset")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--recursive-images",
        action="store_true",
        help="Recursively index image directories if a direct filename lookup fails",
    )
    parser.add_argument(
        "--image-workers",
        type=int,
        default=1,
        help="Concurrent image-dimension readers; use 4-8 on network filesystems",
    )
    parser.add_argument(
        "--crop-workers",
        type=int,
        default=1,
        help="Concurrent deterministic crop writers; use 4-8 on network filesystems",
    )
    parser.add_argument(
        "--strict-missing-images",
        action="store_true",
        help="Fail if any merged item lacks pixels and image-size metadata",
    )
    parser.add_argument("--crop-padding", type=float, default=0.12)
    parser.add_argument("--crop-dir", type=Path, default=None)
    parser.add_argument(
        "--crop-deploy-dir",
        default=None,
        help="Deployment crop directory; defaults to the resolved local --crop-dir",
    )
    parser.add_argument(
        "--materialize-crops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write cropped PNGs locally; otherwise emit a deterministic crop plan",
    )
    parser.add_argument(
        "--exclude-empty-crops",
        action="store_true",
        help="Compatibility alias for --max-empty-crops-per-image 0",
    )
    parser.add_argument(
        "--max-empty-crops-per-image",
        type=int,
        default=1,
        help="Maximum deterministic negative crops per train image; -1 keeps all",
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.crop_padding <= 1.0:
        parser.error("--crop-padding must be between 0 and 1")
    if args.max_empty_crops_per_image < -1:
        parser.error("--max-empty-crops-per-image must be -1 or greater")
    if args.image_workers < 1:
        parser.error("--image-workers must be at least 1")
    if args.crop_workers < 1:
        parser.error("--crop-workers must be at least 1")
    if args.exclude_empty_crops:
        args.max_empty_crops_per_image = 0
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = (
        args.crop_dir.expanduser().resolve()
        if args.crop_dir is not None
        else output_dir / "view_crops"
    )
    crop_deploy_dir = args.crop_deploy_dir or str(crop_dir)

    specs = [
        SourceSpec(
            "5k_10feats",
            args.json_10.expanduser().resolve(),
            args.image_dir_10.expanduser().resolve() if args.image_dir_10 else None,
            args.deploy_dir_10,
            args.projection_10,
            args.scope_10,
            DEFAULT_VIEW_SCOPE_10,
            DEFAULT_DOCUMENT_SCOPE_10,
        ),
        SourceSpec(
            "5k_15feats_with_view",
            args.json_15.expanduser().resolve(),
            args.image_dir_15.expanduser().resolve() if args.image_dir_15 else None,
            args.deploy_dir_15,
            args.projection_15,
            args.scope_15,
            DEFAULT_VIEW_SCOPE_15,
            DEFAULT_DOCUMENT_SCOPE_15,
        ),
        SourceSpec(
            "siemens_788",
            args.json_788.expanduser().resolve(),
            args.image_dir_788.expanduser().resolve() if args.image_dir_788 else None,
            args.deploy_dir_788,
            args.projection_788,
            args.scope_788,
            DEFAULT_VIEW_SCOPE_788,
            DEFAULT_DOCUMENT_SCOPE_788,
        ),
    ]
    for spec in specs:
        if not spec.json_path.is_file():
            raise FileNotFoundError(f"Annotation JSON not found: {spec.json_path}")

    payloads = [load_source(spec) for spec in specs]
    merged = merge_sources(payloads)
    assignment = split_by_family(
        merged,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    resolver = ImageResolver(specs, recursive=args.recursive_images)
    if args.image_workers == 1:
        resolved_images = map(resolver.resolve, merged)
    else:
        image_pool = ThreadPoolExecutor(
            max_workers=args.image_workers, thread_name_prefix="cad-image"
        )
        resolved_images = image_pool.map(resolver.resolve, merged)
    try:
        # executor.map preserves input order, so worker count cannot affect the
        # deterministic manifest or split artifacts.
        image_infos = {
            item.filename: image for item, image in zip(merged, resolved_images)
        }
    finally:
        if args.image_workers != 1:
            image_pool.shutdown(wait=True)
    missing = [
        item.filename for item in merged if not image_infos[item.filename].usable
    ]
    if args.strict_missing_images and missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(
            f"{len(missing)} merged images lack usable dimensions; first: {preview}"
        )

    split_items: dict[str, list[MergedItem]] = {"train": [], "val": [], "test": []}
    dropped_augmented: Counter[str] = Counter()
    for item in merged:
        split = assignment[item.family_id]
        if split != "train" and item.is_augmented:
            dropped_augmented[split] += 1
            continue
        split_items[split].append(item)
    for values in split_items.values():
        values.sort(key=lambda item: item.filename)

    spec_map = {spec.name: spec for spec in specs}
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, items in split_items.items():
        records_by_split[split] = [
            build_three_stage_record(item, image_infos[item.filename], split, spec_map)
            for item in items
            if image_infos[item.filename].usable
        ]

    crop_records, crop_manifest, crop_stats = build_crop_records(
        split_items["train"],
        image_infos,
        spec_map,
        crop_dir,
        crop_deploy_dir,
        args.crop_padding,
        args.materialize_crops,
        max_empty_crops_per_image=args.max_empty_crops_per_image,
        crop_workers=args.crop_workers,
    )

    _write_json(
        output_dir / "merged_annotations.json",
        [_serializable_merged(item) for item in merged],
    )
    manifest_rows: list[dict[str, Any]] = []
    for item in merged:
        image = image_infos[item.filename]
        family_split = assignment[item.family_id]
        excluded = family_split != "train" and item.is_augmented
        manifest_rows.append(
            {
                "filename": item.filename,
                "family_id": item.family_id,
                "split": family_split,
                "included": not excluded and image.usable,
                "exclude_reason": (
                    "augmented_removed_from_eval"
                    if excluded
                    else ("missing_image_dimensions" if not image.usable else "")
                ),
                "is_augmented": item.is_augmented,
                "sources": item.sources,
                "annotation_scope": item.annotation_scope,
                "view_scope": item.view_scope,
                "document_scope": item.document_scope,
                "source_scopes": item.source_scopes,
                "source_view_scopes": item.source_view_scopes,
                "source_document_scopes": item.source_document_scopes,
                "image_path": str(image.local_path) if image.local_path else None,
                "deploy_image_path": image.deploy_path,
                "image_width": image.width,
                "image_height": image.height,
                "image_status": image.status,
                "image_error": image.error,
                "annotation_count": len(item.annotations),
                "duplicate_annotations_removed": item.duplicate_annotations,
                "annotation_conflict_count": len(item.annotation_conflicts),
                "unknown_labels": sorted(item.unknown_labels),
            }
        )
    manifest_count = _write_jsonl(output_dir / "manifest.jsonl", manifest_rows)
    crop_manifest_count = _write_jsonl(
        output_dir / "crop_manifest.jsonl", crop_manifest
    )

    output_counts: dict[str, int] = {
        "manifest": manifest_count,
        "crop_manifest": crop_manifest_count,
    }
    for split in ("train", "val", "test"):
        output_counts[f"{split}_three_stage"] = _write_jsonl(
            output_dir / f"{split}_three_stage.jsonl",
            (_ms_swift_row(record) for record in records_by_split[split]),
        )
        output_counts[f"{split}_ground_truth"] = _write_jsonl(
            output_dir / f"{split}_ground_truth.jsonl",
            (_ground_truth_row(record) for record in records_by_split[split]),
        )
    if args.materialize_crops:
        active_crop_records = crop_records
        planned_crop_records: list[dict[str, Any]] = []
    else:
        # Avoid making the default training pipeline consume paths known not to
        # exist yet.  The plan becomes active after rerunning with materialize.
        active_crop_records = []
        planned_crop_records = crop_records
    output_counts["train_view_crops"] = _write_jsonl(
        output_dir / "train_view_crops.jsonl",
        (_ms_swift_row(record) for record in active_crop_records),
    )
    output_counts["train_view_crops_plan"] = _write_jsonl(
        output_dir / "train_view_crops_plan.jsonl",
        (_ms_swift_row(record) for record in planned_crop_records),
    )
    output_counts["train_view_crops_ground_truth"] = _write_jsonl(
        output_dir / "train_view_crops_ground_truth.jsonl",
        (_ground_truth_row(record) for record in crop_records),
    )

    full_three_stage_counts = {
        split: sum(record["stage1"] is not None for record in records)
        for split, records in records_by_split.items()
    }
    scoped_only_counts = {
        split: sum(record["stage1"] is None for record in records)
        for split, records in records_by_split.items()
    }
    stats: dict[str, Any] = {
        "schema_version": "cad-three-stage-v1",
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "sources": {
            payload.spec.name: {
                "json_path": str(payload.spec.json_path),
                "items": len(payload.items),
                "projection_method": payload.spec.projection_method,
                "feature_scope": sorted(payload.feature_scope),
                "view_scope": list(payload.spec.declared_view_scope),
                "document_scope": list(payload.spec.declared_document_scope),
                "task_names": sorted(payload.task_names),
                "unknown_labels": dict(payload.unknown_labels),
            }
            for payload in payloads
        },
        "merge": {
            "input_items": sum(len(payload.items) for payload in payloads),
            "exact_filename_items": len(merged),
            "multi_source_items": sum(len(item.sources) > 1 for item in merged),
            "duplicates_removed": sum(item.duplicate_annotations for item in merged),
            "annotation_conflicts": sum(
                len(item.annotation_conflicts) for item in merged
            ),
            "families": len(assignment),
            "augmented_items": sum(item.is_augmented for item in merged),
            "missing_image_dimensions": len(missing),
        },
        "splits": {
            split: {
                "families": len({item.family_id for item in items}),
                "items_after_policy": len(items),
                "usable_records": len(records_by_split[split]),
                "three_stage_records": full_three_stage_counts[split],
                "scoped_feature_only_records": scoped_only_counts[split],
                "augmented_items": sum(item.is_augmented for item in items),
                "label_distribution": _label_distribution(items),
                "annotation_scope_distribution": _scope_distribution(items),
                "layout_scope_distribution": _layout_scope_distribution(items),
            }
            for split, items in split_items.items()
        },
        "dropped_augmented_from_eval": dict(dropped_augmented),
        "crops": {
            "materialize_requested": bool(args.materialize_crops),
            "crop_dir": str(crop_dir),
            "crop_deploy_dir": crop_deploy_dir,
            "padding": args.crop_padding,
            "max_empty_crops_per_image": args.max_empty_crops_per_image,
            **crop_stats,
        },
        "outputs": output_counts,
    }
    _write_json(output_dir / "stats.json", stats)
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        stats = run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(_json_dump(stats, pretty=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
