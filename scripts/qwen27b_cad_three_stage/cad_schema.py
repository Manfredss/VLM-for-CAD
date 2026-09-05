#!/usr/bin/env python3
"""Canonical schema and prompts for the three-stage CAD extraction pipeline.

This module deliberately has no third-party dependencies.  It is shared by
dataset preparation, inference, auditing and evaluation so that category names,
coordinate transforms and output contracts cannot silently drift apart.

Canonical task contract
-----------------------
Stage 1: ``{"regions": [{"region_id", "category", "bbox"}, ...]}``
    ``category`` is ``View Region`` for a geometric view, otherwise one of the
    four document categories.
Stage 2: ``{"projection_method", "views": [{"region_id", "category", ...}]}``
    Each view reuses a Stage-1 ``region_id`` and receives one of 11 view labels.
Stage 3: ``{"feature_scope": [...], "features": [{"category", "size", "bbox"}]}``
    ``feature_scope`` is the exhaustive set of feature classes annotated for
    this sample.  It prevents a missing label in a partially annotated source
    dataset from being interpreted as a true negative.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


DEFAULT_MODEL = "Qwen/Qwen3.6-27B"
FALLBACK_MODEL = "Qwen/Qwen3.5-27B"
COORDINATE_SCALE = 1000
GENERIC_VIEW_REGION = "View Region"

# The union of the 5K feature data and the Siemens/788 feature data.  Keep this
# ordered: the order is reused in prompts, reports and deterministic manifests.
FEATURE_CATEGORIES = (
    "Threaded Hole",
    "Threaded Hole Group",
    "Fillet",
    "Fillet Group",
    "Round Hole",
    "Round Hole Group",
    "Pin Hole",
    "Pin Hole Group",
    "Chamfer",
    "Chamfer Group",
    "Counterbore Hole",
    "Counterbore Hole Group",
    "Rectangular Hole",
    "Rectangular Hole Group",
    "Slotted Hole",
    "Slotted Hole Group",
    "Threaded Shaft",
    "Bending",
    "Silver Plating",
)

GROUP_CATEGORIES = tuple(c for c in FEATURE_CATEGORIES if c.endswith(" Group"))
BASE_FEATURE_CATEGORIES = tuple(
    c for c in FEATURE_CATEGORIES if c not in GROUP_CATEGORIES
)

DOCUMENT_CATEGORIES = (
    "Title Block",
    "Notes",
    "Revision Table",
    "Bill of Materials",
)

# These are actual drawing-view semantics.  The legacy code called the union of
# these 11 labels and the 4 document labels "15 view categories".  The new names
# make the distinction explicit while retaining a 15-class compatibility union.
VIEW_CATEGORIES = (
    "Orthographic Projection - Front View",
    "Orthographic Projection - Top View",
    "Orthographic Projection - Bottom View",
    "Orthographic Projection - Left View",
    "Orthographic Projection - Right View",
    "Orthographic Projection - Rear View",
    "Isometric View",
    "Flat Pattern View",
    "Section View",
    "Detail View",
    "Auxiliary View",
)

VIEW_LAYOUT_CATEGORIES = DOCUMENT_CATEGORIES + VIEW_CATEGORIES
REGION_CATEGORIES = VIEW_LAYOUT_CATEGORIES  # legacy annotation vocabulary (15)
STAGE1_CATEGORIES = (GENERIC_VIEW_REGION,) + DOCUMENT_CATEGORIES
ALL_CATEGORIES = FEATURE_CATEGORIES + VIEW_LAYOUT_CATEGORIES

PROJECTION_METHODS = ("first_angle", "third_angle", "unknown")


def _alias_key(value: Any) -> str:
    """Return a stable lookup key; this is normalization, never fuzzy matching."""
    text = str(value or "").strip().casefold()
    text = text.replace("–", "-").replace("—", "-").replace("：", ":")
    text = re.sub(r"[\s_/]+", " ", text)
    text = re.sub(r"\s*-\s*", " - ", text)
    return re.sub(r"\s+", " ", text).strip()


_CATEGORY_ALIASES: dict[str, str] = {}


def _add_aliases(canonical: str, *aliases: str) -> None:
    for value in (canonical, *aliases):
        _CATEGORY_ALIASES[_alias_key(value)] = canonical


# Explicit aliases only.  Do not replace this with edit-distance/category
# snapping: a near spelling can be a semantically different CAD feature.
_add_aliases(GENERIC_VIEW_REGION, "view_region", "view area", "视图区域", "视图区")
_add_aliases(
    "Title Block", "title_block", "drawing title block", "标题栏", "图框标题栏"
)
_add_aliases("Notes", "note", "general notes", "technical notes", "注释", "技术要求")
_add_aliases("Revision Table", "revision_table", "revision block", "修订栏", "更改栏")
_add_aliases(
    "Bill of Materials", "BOM", "bill_of_materials", "parts list", "明细栏", "材料表"
)

for direction, zh in (
    ("Front", "正视图"),
    ("Top", "俯视图"),
    ("Bottom", "仰视图"),
    ("Left", "左视图"),
    ("Right", "右视图"),
    ("Rear", "后视图"),
):
    canonical = f"Orthographic Projection - {direction} View"
    _add_aliases(
        canonical,
        f"{direction} View",
        f"Orthographic {direction} View",
        f"orthographic_{direction.lower()}_view",
        zh,
    )

_add_aliases(
    "Isometric View", "isometric", "iso view", "isometric_view", "等轴测图", "轴测图"
)
_add_aliases(
    "Flat Pattern View", "flat pattern", "flat_pattern_view", "展开视图", "展开图"
)
_add_aliases(
    "Section View", "section", "sectional view", "section_view", "剖视图", "剖面图"
)
_add_aliases("Detail View", "detail", "detail_view", "详图", "局部放大图")
_add_aliases(
    "Auxiliary View", "auxiliary", "aux view", "auxiliary_view", "辅助视图", "斜视图"
)

_add_aliases(
    "Threaded Hole", "thread hole", "tapped hole", "threaded_hole", "螺纹孔", "攻丝孔"
)
_add_aliases(
    "Threaded Hole Group",
    "thread hole group",
    "tapped hole group",
    "threaded_hole_group",
    "螺纹孔组",
)
_add_aliases("Fillet", "round", "radius fillet", "fillet_radius", "圆角")
_add_aliases("Fillet Group", "fillets", "fillet_group", "圆角组")
_add_aliases("Round Hole", "circular hole", "plain hole", "round_hole", "圆孔", "通孔")
_add_aliases("Round Hole Group", "circular hole group", "round_hole_group", "圆孔组")
_add_aliases("Pin Hole", "dowel hole", "reamed hole", "pin_hole", "销孔", "定位孔")
_add_aliases("Pin Hole Group", "dowel hole group", "pin_hole_group", "销孔组")
_add_aliases("Chamfer", "bevel", "chamfer_edge", "倒角", "斜角")
_add_aliases("Chamfer Group", "chamfers", "chamfer_group", "倒角组", "斜角组")
_add_aliases(
    "Counterbore Hole",
    "counterbore",
    "counter bored hole",
    "counterbore_hole",
    "沉孔",
    "沉台孔",
)
_add_aliases(
    "Counterbore Hole Group", "counterbore group", "counterbore_hole_group", "沉孔组"
)
_add_aliases(
    "Rectangular Hole",
    "rectangle hole",
    "square hole",
    "rectangular_hole",
    "矩形孔",
    "方孔",
)
_add_aliases(
    "Rectangular Hole Group",
    "rectangle hole group",
    "rectangular_hole_group",
    "矩形孔组",
)
_add_aliases(
    "Slotted Hole", "slot", "slot hole", "oblong hole", "slotted_hole", "腰孔", "长圆孔"
)
_add_aliases(
    "Slotted Hole Group", "slot group", "slotted_hole_group", "腰孔组", "长圆孔组"
)
_add_aliases("Threaded Shaft", "thread shaft", "threaded_shaft", "螺纹轴", "外螺纹")
_add_aliases("Bending", "bend", "bending line", "fold", "折弯", "折弯线")
_add_aliases("Silver Plating", "silver plate", "silver_plating", "镀银", "银镀层")

CATEGORY_ALIASES = dict(_CATEGORY_ALIASES)

_PROJECTION_ALIASES = {
    _alias_key("first_angle"): "first_angle",
    _alias_key("first-angle"): "first_angle",
    _alias_key("first angle projection"): "first_angle",
    _alias_key("第一角投影"): "first_angle",
    _alias_key("third_angle"): "third_angle",
    _alias_key("third-angle"): "third_angle",
    _alias_key("third angle projection"): "third_angle",
    _alias_key("第三角投影"): "third_angle",
    _alias_key("unknown"): "unknown",
    _alias_key("未知"): "unknown",
    "": "unknown",
}


def categories_for_kind(kind: Optional[str]) -> tuple[str, ...]:
    key = str(kind or "all").strip().casefold().replace("-", "_")
    mapping = {
        "feature": FEATURE_CATEGORIES,
        "features": FEATURE_CATEGORIES,
        "base_feature": BASE_FEATURE_CATEGORIES,
        "view": VIEW_CATEGORIES,
        "views": VIEW_CATEGORIES,
        "document": DOCUMENT_CATEGORIES,
        "documents": DOCUMENT_CATEGORIES,
        "layout": VIEW_LAYOUT_CATEGORIES,
        "view_layout": VIEW_LAYOUT_CATEGORIES,
        "region": STAGE1_CATEGORIES,
        "stage1": STAGE1_CATEGORIES,
        "all": ALL_CATEGORIES + (GENERIC_VIEW_REGION,),
    }
    if key not in mapping:
        raise ValueError(f"unknown category kind: {kind!r}")
    return mapping[key]


def normalize_category(
    value: Any,
    allowed: Optional[Iterable[str]] = None,
    kind: Optional[str] = None,
    *,
    strict: bool = True,
) -> Optional[str]:
    """Resolve a category through the explicit alias table.

    Unknown labels return ``None`` in strict mode and the stripped original
    value otherwise.  No fuzzy matching is performed.
    """
    raw = str(value or "").strip()
    canonical = _CATEGORY_ALIASES.get(_alias_key(raw))
    if canonical is None:
        return None if strict else raw
    expected = set(allowed if allowed is not None else categories_for_kind(kind))
    if canonical not in expected:
        return None if strict else canonical
    return canonical


def normalize_projection_method(value: Any) -> Optional[str]:
    return _PROJECTION_ALIASES.get(_alias_key(value))


def normalize_size(value: Any, category: Optional[str] = None) -> str:
    """Canonicalize common CAD size notation for comparison, not presentation."""
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.translate(
        str.maketrans(
            {
                "，": ",",
                "。": ".",
                "×": "X",
                "＊": "X",
                "*": "X",
                "－": "-",
                "–": "-",
                "—": "-",
                "φ": "Ø",
                "Φ": "Ø",
                "ø": "Ø",
                "⌀": "Ø",
                "∅": "Ø",
            }
        )
    )
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    text = re.sub(r"\bDIA(?:METER)?\.?\b", "Ø", text, flags=re.IGNORECASE)
    text = re.sub(r"\bDEG(?:REE)?S?\b", "°", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d)\s*MM\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", "", text).upper()
    canonical_category = (
        normalize_category(category, kind="feature") if category else None
    )

    # Source labels variously encode a group count as 4x, 4-, or 4×.
    if canonical_category in GROUP_CATEGORIES:
        text = re.sub(r"^(\d+)[-X](?=[A-ZØRCDM\d])", r"\1X", text)

    # Diameter symbols are inconsistently omitted for plain round holes.  The
    # category already carries the diameter semantics, so ignore the symbol.
    if canonical_category in {"Round Hole", "Round Hole Group"}:
        text = re.sub(r"(^|[X-])Ø(?=\d)", r"\1", text)
    if canonical_category in {"Slotted Hole", "Slotted Hole Group"}:
        text = text.replace("LL", "")

    text = re.sub(r"(?<=\d)\.0+(?=$|[^0-9])", "", text)
    return text


_AUGMENTATION_PATTERNS = (
    # The 5K export contains many non-right-angle rotations (rot5, rot45,
    # rot355, ...), not only rot90/180/270.  Treat every numeric rot suffix as
    # augmentation so all derivatives stay in the same split family.
    r"(?:^|[_-])rot(?:ated|ation)?[_-]?[-+]?\d+(?:\.\d+)?(?:deg)?$",
    r"(?:^|[_-])r(?:90|180|270)$",
    r"(?:^|[_-])flip(?:ped)?[_-]?(?:h|v|horizontal|vertical|lr|ud)?$",
    r"(?:^|[_-])(?:aug|augment|augmented)(?:[_-]?\d+)?$",
    r"(?:^|[_-])(?:brightness|contrast|noise|blur|skew|perspective)(?:[_-]?[\d.]+)?$",
)
_AUGMENTATION_RE = re.compile(
    "|".join(f"(?:{p})" for p in _AUGMENTATION_PATTERNS), re.IGNORECASE
)


def is_augmented_name(filename: Any) -> bool:
    stem = Path(str(filename or "")).stem
    return bool(_AUGMENTATION_RE.search(stem))


def infer_family_id(filename: Any) -> str:
    """Infer a leakage-safe drawing family from an image/sample name.

    Augmentation and page suffixes are stripped repeatedly.  The result is
    intentionally conservative: revision/sheet identifiers embedded inside a
    part number are kept because removing them without source metadata can join
    unrelated drawings.
    """
    stem = Path(str(filename or "")).stem.strip().casefold()
    stem = re.sub(r"[\s]+", "_", stem)
    previous = None
    while stem and stem != previous:
        previous = stem
        stem = _AUGMENTATION_RE.sub("", stem).rstrip("_-")
    stem = re.sub(r"(?:[_-])page(?:[_-]?\d+)$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"(?:[_-])p(?:age)?[_-]?\d+$", "", stem, flags=re.IGNORECASE)
    # Siemens exports sometimes prepend a three-digit catalog row number.  The
    # same drawing can appear under different row numbers (for example
    # 226_A7... and 227_A7...), so the prefix is not part of drawing identity.
    # Limit stripping to the two known drawing-ID shapes to avoid joining
    # unrelated filenames that legitimately begin with three digits.
    stem = re.sub(r"^\d{3}_(?=(?:a7e[a-z0-9]+|\d{8})(?:_|$))", "", stem)
    stem = re.sub(r"[_-]+$", "", stem)
    return stem or "unknown_family"


def _box_values(box: Any) -> Optional[list[float]]:
    if isinstance(box, Mapping):
        key_sets = (
            ("x_min", "y_min", "x_max", "y_max"),
            ("xmin", "ymin", "xmax", "ymax"),
            ("x1", "y1", "x2", "y2"),
        )
        for keys in key_sets:
            if all(k in box for k in keys):
                box = [box[k] for k in keys]
                break
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        values = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    return values if all(math.isfinite(v) for v in values) else None


def normalize_bbox(
    box: Any,
    image_width: Optional[float] = None,
    image_height: Optional[float] = None,
    *,
    assume_pixels: Optional[bool] = None,
    integer: bool = True,
    clip: bool = True,
) -> list[int] | list[float]:
    """Normalize a bbox to ``[0, 1000]`` xyxy coordinates.

    Fractional 0..1 boxes are detected automatically.  Pixel boxes are scaled
    when image dimensions are supplied and either ``assume_pixels=True`` or any
    coordinate exceeds 1000.
    """
    values = _box_values(box)
    if values is None:
        return []
    x1, y1, x2, y2 = values
    if max(abs(v) for v in values) <= 1.5:
        x1, y1, x2, y2 = (v * COORDINATE_SCALE for v in values)
    elif (
        image_width
        and image_height
        and (assume_pixels is True or max(abs(v) for v in values) > COORDINATE_SCALE)
    ):
        if image_width <= 0 or image_height <= 0:
            return []
        x1, x2 = (
            x1 / image_width * COORDINATE_SCALE,
            x2 / image_width * COORDINATE_SCALE,
        )
        y1, y2 = (
            y1 / image_height * COORDINATE_SCALE,
            y2 / image_height * COORDINATE_SCALE,
        )
    if clip:
        x1, y1, x2, y2 = (
            max(0.0, min(float(COORDINATE_SCALE), v)) for v in (x1, y1, x2, y2)
        )
    if x1 >= x2 or y1 >= y2:
        return []
    result: list[float] = [x1, y1, x2, y2]
    return [int(round(v)) for v in result] if integer else result


# Backward-compatible name used by older metric scripts.
normalize_box = normalize_bbox


def valid_bbox(box: Any, *, normalized: bool = True) -> bool:
    values = _box_values(box)
    if values is None:
        return False
    x1, y1, x2, y2 = values
    if x1 >= x2 or y1 >= y2:
        return False
    return not normalized or (min(values) >= 0 and max(values) <= COORDINATE_SCALE)


def bbox_area(box: Sequence[float]) -> float:
    values = _box_values(box)
    if values is None:
        return 0.0
    return max(0.0, values[2] - values[0]) * max(0.0, values[3] - values[1])


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    a, b = _box_values(box_a), _box_values(box_b)
    if a is None or b is None:
        return 0.0
    ix1, iy1, ix2, iy2 = (
        max(a[0], b[0]),
        max(a[1], b[1]),
        min(a[2], b[2]),
        min(a[3], b[3]),
    )
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def bbox_center(box: Sequence[float]) -> Optional[tuple[float, float]]:
    values = _box_values(box)
    if values is None:
        return None
    return ((values[0] + values[2]) / 2.0, (values[1] + values[3]) / 2.0)


def bbox_contains(
    outer: Sequence[float], inner: Sequence[float], *, center_only: bool = False
) -> bool:
    a, b = _box_values(outer), _box_values(inner)
    if a is None or b is None:
        return False
    if center_only:
        center = bbox_center(b)
        return bool(center and a[0] <= center[0] <= a[2] and a[1] <= center[1] <= a[3])
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def expand_bbox(box: Sequence[float], padding: float = 0.10) -> list[int]:
    values = _box_values(box)
    if values is None or values[0] >= values[2] or values[1] >= values[3]:
        return []
    dx, dy = (values[2] - values[0]) * padding, (values[3] - values[1]) * padding
    return normalize_bbox(
        [values[0] - dx, values[1] - dy, values[2] + dx, values[3] + dy]
    )


def crop_to_global_bbox(
    local_bbox: Sequence[float], crop_bbox: Sequence[float]
) -> list[int]:
    """Map crop-local 0..1000 coordinates into global 0..1000 coordinates."""
    local = normalize_bbox(local_bbox, integer=False)
    crop = normalize_bbox(crop_bbox, integer=False)
    if not local or not crop:
        return []
    width, height = crop[2] - crop[0], crop[3] - crop[1]
    result = [
        crop[0] + local[0] / COORDINATE_SCALE * width,
        crop[1] + local[1] / COORDINATE_SCALE * height,
        crop[0] + local[2] / COORDINATE_SCALE * width,
        crop[1] + local[3] / COORDINATE_SCALE * height,
    ]
    return normalize_bbox(result)


def global_to_crop_bbox(
    global_bbox: Sequence[float], crop_bbox: Sequence[float]
) -> list[int]:
    """Map global 0..1000 coordinates into crop-local 0..1000 coordinates."""
    box = normalize_bbox(global_bbox, integer=False)
    crop = normalize_bbox(crop_bbox, integer=False)
    if not box or not crop:
        return []
    width, height = crop[2] - crop[0], crop[3] - crop[1]
    result = [
        (box[0] - crop[0]) / width * COORDINATE_SCALE,
        (box[1] - crop[1]) / height * COORDINATE_SCALE,
        (box[2] - crop[0]) / width * COORDINATE_SCALE,
        (box[3] - crop[1]) / height * COORDINATE_SCALE,
    ]
    return normalize_bbox(result)


STAGE1_SCHEMA = {
    "type": "object",
    "required": ["regions"],
    "properties": {
        "regions": {
            "type": "array",
            "items": {"type": "object", "required": ["region_id", "category", "bbox"]},
        }
    },
}
STAGE2_SCHEMA = {
    "type": "object",
    "required": ["projection_method", "views"],
    "properties": {
        "projection_method": {"enum": list(PROJECTION_METHODS)},
        "views": {"type": "array"},
    },
}
STAGE3_SCHEMA = {
    "type": "object",
    "required": ["feature_scope", "features"],
    "properties": {"feature_scope": {"type": "array"}, "features": {"type": "array"}},
}


def parse_json_object(value: Any) -> Optional[dict[str, Any]]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return None
    text = re.sub(r"<think>[\s\S]*?</think>", "", value).strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return dict(parsed) if isinstance(parsed, Mapping) else None


def _stage_number(stage: Any) -> int:
    text = str(stage).strip().casefold().replace("_", "")
    aliases = {
        "1": 1,
        "stage1": 1,
        "step1": 1,
        "layout": 1,
        "2": 2,
        "stage2": 2,
        "step2": 2,
        "view": 2,
        "3": 3,
        "stage3": 3,
        "step3": 3,
        "feature": 3,
    }
    if text not in aliases:
        raise ValueError(f"unknown stage: {stage!r}")
    return aliases[text]


def validate_stage_output(stage: Any, value: Any, *, strict: bool = True) -> list[str]:
    """Return human-readable validation errors; never raises for bad payloads."""
    try:
        number = _stage_number(stage)
    except ValueError as exc:
        return [str(exc)]
    obj = parse_json_object(value)
    if obj is None:
        return ["output is not a JSON object"]
    errors: list[str] = []

    if number == 1:
        if strict and set(obj) - {"regions"}:
            errors.append(
                f"unexpected top-level keys: {sorted(set(obj) - {'regions'})}"
            )
        regions = obj.get("regions")
        if not isinstance(regions, list):
            return errors + ["regions must be a list"]
        seen: set[str] = set()
        for index, item in enumerate(regions):
            prefix = f"regions[{index}]"
            if not isinstance(item, Mapping):
                errors.append(f"{prefix} must be an object")
                continue
            region_id = str(item.get("region_id", "")).strip()
            if not region_id:
                errors.append(f"{prefix}.region_id is required")
            elif region_id in seen:
                errors.append(f"duplicate region_id: {region_id}")
            seen.add(region_id)
            if normalize_category(item.get("category"), kind="stage1") is None:
                errors.append(f"{prefix}.category is not an allowed Stage-1 category")
            if not valid_bbox(item.get("bbox")):
                errors.append(f"{prefix}.bbox must be valid normalized xyxy")
            if strict and set(item) - {"region_id", "category", "bbox"}:
                errors.append(
                    f"{prefix} has unexpected keys: {sorted(set(item) - {'region_id', 'category', 'bbox'})}"
                )

    elif number == 2:
        if strict and set(obj) - {"projection_method", "views"}:
            errors.append(
                f"unexpected top-level keys: {sorted(set(obj) - {'projection_method', 'views'})}"
            )
        if normalize_projection_method(obj.get("projection_method")) is None:
            errors.append(
                "projection_method must be first_angle, third_angle, or unknown"
            )
        views = obj.get("views")
        if not isinstance(views, list):
            return errors + ["views must be a list"]
        seen = set()
        for index, item in enumerate(views):
            prefix = f"views[{index}]"
            if not isinstance(item, Mapping):
                errors.append(f"{prefix} must be an object")
                continue
            region_id = str(item.get("region_id", "")).strip()
            if not region_id:
                errors.append(f"{prefix}.region_id is required")
            elif region_id in seen:
                errors.append(f"duplicate region_id: {region_id}")
            seen.add(region_id)
            if normalize_category(item.get("category"), kind="view") is None:
                errors.append(f"{prefix}.category is not an allowed view category")
            if "bbox" in item and not valid_bbox(item.get("bbox")):
                errors.append(
                    f"{prefix}.bbox must be valid normalized xyxy when present"
                )
            allowed_keys = {"region_id", "category", "bbox"}
            if strict and set(item) - allowed_keys:
                errors.append(
                    f"{prefix} has unexpected keys: {sorted(set(item) - allowed_keys)}"
                )

    else:
        if strict and set(obj) - {"feature_scope", "features"}:
            errors.append(
                f"unexpected top-level keys: {sorted(set(obj) - {'feature_scope', 'features'})}"
            )
        scope = obj.get("feature_scope")
        if not isinstance(scope, list):
            errors.append(
                "feature_scope must be a list of exhaustive feature categories"
            )
            scope_set: set[str] = set()
        else:
            canonical_scope = [normalize_category(c, kind="feature") for c in scope]
            if any(c is None for c in canonical_scope):
                errors.append("feature_scope contains an unknown feature category")
            scope_set = {c for c in canonical_scope if c is not None}
            if len(scope_set) != len(scope):
                errors.append("feature_scope contains duplicate categories")
        features = obj.get("features")
        if not isinstance(features, list):
            return errors + ["features must be a list"]
        for index, item in enumerate(features):
            prefix = f"features[{index}]"
            if not isinstance(item, Mapping):
                errors.append(f"{prefix} must be an object")
                continue
            category = normalize_category(item.get("category"), kind="feature")
            if category is None:
                errors.append(f"{prefix}.category is not an allowed feature category")
            elif scope_set and category not in scope_set:
                errors.append(f"{prefix}.category is outside feature_scope")
            if "size" not in item or not isinstance(item.get("size"), str):
                errors.append(f"{prefix}.size must be a string (empty is allowed)")
            if not valid_bbox(item.get("bbox")):
                errors.append(f"{prefix}.bbox must be valid normalized xyxy")
            allowed_keys = {"category", "size", "bbox", "region_id"}
            if strict and set(item) - allowed_keys:
                errors.append(
                    f"{prefix} has unexpected keys: {sorted(set(item) - allowed_keys)}"
                )
    return errors


def validate_cross_stage_outputs(
    stage1: Any = None,
    stage2: Any = None,
    stage3: Any = None,
) -> list[str]:
    """Validate relationships that isolated JSON Schemas cannot express.

    Missing stages are allowed because crop and partial-supervision records may
    intentionally contain only a subset of the pipeline.  When both Stage 1
    and Stage 2 are present, however, every generic view region must be
    classified exactly once and Stage 2 may not invent region IDs.  A Stage-3
    ``region_id`` is optional, but when supplied it must reference a Stage-1
    view region (or a Stage-2 view when Stage 1 is unavailable).
    """

    first = parse_json_object(stage1) if stage1 is not None else None
    second = parse_json_object(stage2) if stage2 is not None else None
    third = parse_json_object(stage3) if stage3 is not None else None
    errors: list[str] = []

    stage1_view_ids: set[str] = set()
    if first is not None and isinstance(first.get("regions"), list):
        for item in first["regions"]:
            if not isinstance(item, Mapping):
                continue
            if (
                normalize_category(item.get("category"), kind="stage1")
                != GENERIC_VIEW_REGION
            ):
                continue
            region_id = str(item.get("region_id", "")).strip()
            if region_id:
                stage1_view_ids.add(region_id)

    stage2_view_ids: set[str] = set()
    if second is not None and isinstance(second.get("views"), list):
        for item in second["views"]:
            if not isinstance(item, Mapping):
                continue
            region_id = str(item.get("region_id", "")).strip()
            if region_id:
                stage2_view_ids.add(region_id)

    if first is not None and second is not None:
        missing = sorted(stage1_view_ids - stage2_view_ids)
        extra = sorted(stage2_view_ids - stage1_view_ids)
        if missing:
            errors.append(
                f"Stage 2 does not classify Stage-1 view region IDs: {missing}"
            )
        if extra:
            errors.append(
                f"Stage 2 references unknown Stage-1 view region IDs: {extra}"
            )

    valid_feature_region_ids = stage1_view_ids if first is not None else stage2_view_ids
    has_region_context = first is not None or second is not None
    if (
        has_region_context
        and third is not None
        and isinstance(third.get("features"), list)
    ):
        for index, item in enumerate(third["features"]):
            if not isinstance(item, Mapping):
                continue
            region_id = str(item.get("region_id", "")).strip()
            if region_id and region_id not in valid_feature_region_ids:
                errors.append(
                    f"Stage 3 features[{index}].region_id references unknown view region: {region_id}"
                )
    return errors


SYSTEM_PROMPT = """你是工业二维 CAD 工程图解析专家。请严格按指定阶段完成任务，遵守给定类别枚举和 JSON Schema。不得猜测不可见对象，不得输出解释、Markdown、思维过程或枚举外类别。所有 bbox 使用 [x_min,y_min,x_max,y_max]，归一化至 0-1000。"""


def _category_lines(categories: Iterable[str]) -> str:
    return "\n".join(f"- {category}" for category in categories)


def step1_prompt(
    document_scope: Optional[Iterable[str]] = None,
    view_scope: Optional[Iterable[str]] = None,
) -> str:
    scoped_documents: list[str] = []
    for category in DOCUMENT_CATEGORIES if document_scope is None else document_scope:
        canonical = normalize_category(category, kind="document")
        if canonical and canonical not in scoped_documents:
            scoped_documents.append(canonical)
    has_view_scope = view_scope is None or any(
        normalize_category(category, kind="view") for category in view_scope
    )
    document_lines = (
        _category_lines(scoped_documents)
        if scoped_documents
        else "- （本样本不标注文档元素）"
    )
    view_rule = (
        f"找出每个几何视图区，统一标为 `{GENERIC_VIEW_REGION}`。"
        if has_view_scope
        else "本样本不标注几何视图区；不要输出 `View Region`。"
    )
    return f"""阶段 1/3：区域定位（不做视图方向分类）。

{view_rule}
同时找出本样本 document_scope 内的文档元素；文档标注范围为：
{document_lines}

未列入上述范围的文档类别既不是正样本，也不是负样本，不要输出。

规则：
1. 为每个区域生成稳定且唯一的 region_id：视图区 r001...，文档区 d001...。
2. bbox 要覆盖区域主体，但不要把相邻视图、尺寸链或图框吞入。
3. 此阶段严禁输出 Front/Top/Left 等语义；这些由阶段 2 判断。
4. 没有区域时返回空 regions。

仅返回严格 JSON 对象：
{{"regions":[{{"region_id":"r001","category":"View Region","bbox":[0,0,1000,1000]}}]}}"""


def step2_prompt(
    regions: Optional[Any] = None, view_scope: Optional[Iterable[str]] = None
) -> str:
    scoped_views: list[str] = []
    for category in VIEW_CATEGORIES if view_scope is None else view_scope:
        canonical = normalize_category(category, kind="view")
        if canonical and canonical not in scoped_views:
            scoped_views.append(canonical)
    view_lines = (
        _category_lines(scoped_views)
        if scoped_views
        else "- （本样本不标注任何视图语义；views 必须为空）"
    )
    context = ""
    if regions is not None:
        context = "\n阶段 1 结果（region_id 必须原样复用）：\n" + json.dumps(
            regions, ensure_ascii=False, separators=(",", ":")
        )
    return f"""阶段 2/3：投影法与视图语义分类。

只处理阶段 1 中 category=`{GENERIC_VIEW_REGION}` 的区域。先判断 projection_method 为 `first_angle`、`third_angle` 或 `unknown`，再为每个几何区域选择且仅选择以下一种视图类别：
{view_lines}

规则：
1. region_id 必须与阶段 1 完全一致，不得增删几何区域。
2. 综合几何内容、剖切/局部符号及视图相对位置判断；投影法不确定时必须输出 unknown，不可强猜。
3. 可回传 bbox，但若回传必须与阶段 1 对应区域一致。
4. 文档区域不得出现在 views。

仅返回严格 JSON 对象：
{{"projection_method":"unknown","views":[{{"region_id":"r001","category":"Orthographic Projection - Front View"}}]}}{context}"""


def layout_view_prompt(
    document_scope: Optional[Iterable[str]] = None,
    view_scope: Optional[Iterable[str]] = None,
) -> str:
    """Build the compact first prompt for the two-stage P2 protocol."""

    scoped_documents: list[str] = []
    for category in DOCUMENT_CATEGORIES if document_scope is None else document_scope:
        canonical = normalize_category(category, kind="document")
        if canonical and canonical not in scoped_documents:
            scoped_documents.append(canonical)
    scoped_views: list[str] = []
    for category in VIEW_CATEGORIES if view_scope is None else view_scope:
        canonical = normalize_category(category, kind="view")
        if canonical and canonical not in scoped_views:
            scoped_views.append(canonical)
    document_lines = (
        _category_lines(scoped_documents)
        if scoped_documents
        else "- （本样本不标注文档元素）"
    )
    view_lines = (
        _category_lines(scoped_views)
        if scoped_views
        else "- （本样本不标注任何视图语义）"
    )
    return f"""P2 阶段 1/2：一次完成区域定位、投影法与视图语义分类。

先在 regions 中定位每个几何视图和本样本范围内的文档区域；几何视图统一标为 `{GENERIC_VIEW_REGION}`。再在同一个 JSON 对象的 views 中为每个几何视图选择一种语义类别。

允许的几何视图类别：
{view_lines}

允许的文档类别：
{document_lines}

规则：
1. 先判断 projection_method 为 `first_angle`、`third_angle` 或 `unknown`。
2. 每个区域只在 regions 中输出一次。几何视图使用 r001...；文档区域使用 d001...；region_id 必须稳定且唯一。
3. views 必须逐一复用 regions 中所有 `{GENERIC_VIEW_REGION}` 的 region_id，不得增删或重复；文档区域不得进入 views。
4. 综合几何内容、剖切/局部符号及相对位置判断视图语义；不确定投影法时输出 unknown，不可强猜。
5. bbox 覆盖区域主体，使用整图 0-1000 坐标；不要吞入相邻视图、尺寸链或图框。
6. 没有区域时返回空 regions 和空 views；不得输出枚举外类别。

仅返回严格 JSON 对象：
{{"projection_method":"unknown","regions":[{{"region_id":"r001","category":"View Region","bbox":[0,0,1000,1000]}},{{"region_id":"d001","category":"Title Block","bbox":[0,0,1000,1000]}}],"views":[{{"region_id":"r001","category":"Orthographic Projection - Front View"}}]}}"""


def step3_prompt(
    feature_scope: Optional[Iterable[str]] = None,
    region_id: Optional[str] = None,
    *,
    crop_context: bool = True,
) -> str:
    scope = []
    for category in FEATURE_CATEGORIES if feature_scope is None else feature_scope:
        canonical = normalize_category(category, kind="feature")
        if canonical and canonical not in scope:
            scope.append(canonical)
    coordinate_rule = (
        "输入是高分辨率视图裁剪；bbox 必须相对该裁剪归一化到 0-1000。"
        if crop_context
        else "输入是整图；bbox 必须相对整张图归一化到 0-1000。"
    )
    region_rule = (
        f"所有 features 可带相同的 region_id=`{region_id}`。"
        if region_id
        else "region_id 可省略。"
    )
    scope_json = json.dumps(scope, ensure_ascii=False, separators=(",", ":"))
    return f"""阶段 3/3：结构特征检测与尺寸提取。

本样本只对 feature_scope 中的类别做穷尽检测；未列入 scope 的类别既不是正样本，也不是负样本。允许类别如下：
{_category_lines(scope)}

{coordinate_rule}
{region_rule}
规则：
1. 检出每个可见实例；Group 是描述性目标，Group 子实例仍需逐个输出。
2. size 保留 CAD 语义（M、R、Ø、H7、角度、公差、数量）；图中无明确尺寸时用空字符串，禁止猜测。
3. Round/Threaded/Pin/Counterbore、Slotted/Rectangular、Fillet/Chamfer 必须根据符号与几何证据区分。
4. 没有目标时返回空 features；不得输出 scope 外类别。

仅返回严格 JSON 对象：
{{"feature_scope":{scope_json},"features":[{{"category":"Round Hole","size":"Ø8","bbox":[0,0,1000,1000]}}]}}"""


def crop_prompt(
    feature_scope: Optional[Iterable[str]] = None, region_id: Optional[str] = None
) -> str:
    return step3_prompt(feature_scope, region_id, crop_context=True)


def get_stage_prompt(stage: Any, **kwargs: Any) -> str:
    number = _stage_number(stage)
    if number == 1:
        return step1_prompt(kwargs.get("document_scope"), kwargs.get("view_scope"))
    if number == 2:
        return step2_prompt(kwargs.get("regions"), kwargs.get("view_scope"))
    return step3_prompt(
        kwargs.get("feature_scope"),
        kwargs.get("region_id"),
        crop_context=kwargs.get("crop_context", True),
    )


__all__ = [
    "DEFAULT_MODEL",
    "FALLBACK_MODEL",
    "COORDINATE_SCALE",
    "GENERIC_VIEW_REGION",
    "FEATURE_CATEGORIES",
    "BASE_FEATURE_CATEGORIES",
    "GROUP_CATEGORIES",
    "DOCUMENT_CATEGORIES",
    "VIEW_CATEGORIES",
    "VIEW_LAYOUT_CATEGORIES",
    "REGION_CATEGORIES",
    "STAGE1_CATEGORIES",
    "ALL_CATEGORIES",
    "PROJECTION_METHODS",
    "CATEGORY_ALIASES",
    "STAGE1_SCHEMA",
    "STAGE2_SCHEMA",
    "STAGE3_SCHEMA",
    "SYSTEM_PROMPT",
    "categories_for_kind",
    "normalize_category",
    "normalize_projection_method",
    "normalize_size",
    "infer_family_id",
    "is_augmented_name",
    "normalize_bbox",
    "normalize_box",
    "valid_bbox",
    "bbox_area",
    "bbox_iou",
    "bbox_center",
    "bbox_contains",
    "expand_bbox",
    "crop_to_global_bbox",
    "global_to_crop_bbox",
    "parse_json_object",
    "validate_stage_output",
    "validate_cross_stage_outputs",
    "step1_prompt",
    "step2_prompt",
    "layout_view_prompt",
    "step3_prompt",
    "crop_prompt",
    "get_stage_prompt",
]
