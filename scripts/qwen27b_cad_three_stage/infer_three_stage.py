#!/usr/bin/env python3
"""Robust three-stage inference for 2D-CAD feature extraction.

Stage 1 locates generic view/document regions, Stage 2 classifies view
semantics and projection method, and Stage 3 extracts structural features.
Stage 3 can run on the full drawing, on high-resolution view crops, or in a
hybrid mode.  Crop-local 0..1000 boxes are mapped back to full-image 0..1000
coordinates before cross-pass deduplication.

The output is resumable.  Both JSON and JSONL files are rewritten atomically at
``--save-every`` boundaries; completed images are skipped on the next run while
failed images are retried.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from PIL import Image, ImageOps

from cad_schema import (
    DEFAULT_MODEL,
    DOCUMENT_CATEGORIES,
    FEATURE_CATEGORIES,
    GENERIC_VIEW_REGION,
    STAGE1_CATEGORIES,
    SYSTEM_PROMPT,
    VIEW_CATEGORIES,
    bbox_iou,
    crop_to_global_bbox,
    normalize_bbox,
    normalize_category,
    normalize_projection_method,
    normalize_size,
    layout_view_prompt,
    step1_prompt,
    step2_prompt,
    step3_prompt,
    validate_cross_stage_outputs,
    validate_stage_output,
)


LOGGER = logging.getLogger("cad_three_stage_inference")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    # A truncated reasoning block must not make the subsequent JSON invisible.
    if "<think>" in text.lower():
        starts = [pos for pos in (text.find("{"), text.find("[")) if pos >= 0]
        if starts:
            text = text[min(starts) :]
    return text.strip().lstrip("\ufeff")


def _json_candidates(text: str) -> Iterable[str]:
    """Yield strict JSON candidates without using fuzzy Python-literal repair."""

    yielded: set[str] = set()
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE):
        candidate = match.group(1).strip()
        if candidate and candidate not in yielded:
            yielded.add(candidate)
            yield candidate
    stripped = text.strip()
    if stripped and stripped not in yielded:
        yielded.add(stripped)
        yield stripped

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        candidate = text[index : index + end]
        if candidate not in yielded:
            yielded.add(candidate)
            yield candidate

    # Limited recovery for a response cut immediately after a complete item.
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidate = text[first : last + 1].rstrip().rstrip(",")
        opens = candidate.count("{") - candidate.count("}")
        brackets = candidate.count("[") - candidate.count("]")
        if opens >= 0 and brackets >= 0:
            repaired = candidate + ("]" * brackets) + ("}" * opens)
            if repaired not in yielded:
                yield repaired


def parse_stage_json(raw_text: str, stage: int) -> tuple[dict[str, Any], list[str]]:
    """Parse a model response and accept a narrow legacy-list fallback."""

    text = _strip_thinking(str(raw_text or ""))
    errors: list[str] = []
    parsed: Any = None
    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
            break
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    if parsed is None:
        return {}, [f"step{stage}: JSON parse failed"]
    if isinstance(parsed, list):
        # The new protocol is object-only.  Wrapping old adapters' list output
        # is deterministic and lets users compare a Qwen3.5 fallback checkpoint.
        if stage == 1:
            parsed = {"regions": parsed}
        elif stage == 2:
            parsed = {"projection_method": "unknown", "views": parsed}
        else:
            parsed = {"feature_scope": list(FEATURE_CATEGORIES), "features": parsed}
        errors.append(f"step{stage}: wrapped legacy top-level list")
    if not isinstance(parsed, Mapping):
        return {}, errors + [f"step{stage}: top-level value is not an object"]
    return dict(parsed), errors


def raw_stage_json_valid(raw_text: str, stage: int) -> bool:
    """True only for a direct, schema-valid JSON object before any repair."""

    text = str(raw_text or "").strip().lstrip("\ufeff")
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    return isinstance(value, Mapping) and not validate_stage_output(
        stage, value, strict=True
    )


def parse_layout_view_json(raw_text: str) -> tuple[dict[str, Any], list[str]]:
    """Parse the compact layout+view response used by the P2 protocol."""

    text = _strip_thinking(str(raw_text or ""))
    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(parsed, Mapping):
            return dict(parsed), []
        return {}, ["layout_view: top-level value is not an object"]
    return {}, ["layout_view: JSON parse failed"]


def _canonical_category(value: Any, *, kind: str, allowed: Sequence[str]) -> str | None:
    return normalize_category(value, allowed=allowed, kind=kind, strict=True)


def normalize_step1(payload: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    output: list[dict[str, Any]] = []
    errors: list[str] = []
    used_ids: set[str] = set()
    raw_regions = payload.get("regions", [])
    if not isinstance(raw_regions, list):
        raw_regions = []
        errors.append("step1.regions is not a list")

    view_index = doc_index = 0
    for index, item in enumerate(raw_regions):
        if not isinstance(item, Mapping):
            errors.append(f"step1.regions[{index}] is not an object")
            continue
        category = _canonical_category(
            item.get("category"), kind="stage1", allowed=STAGE1_CATEGORIES
        )
        bbox = normalize_bbox(item.get("bbox", item.get("bbox_2d")))
        if category is None or not bbox:
            errors.append(
                f"step1.regions[{index}] has unknown category or invalid bbox"
            )
            continue
        if category == GENERIC_VIEW_REGION:
            view_index += 1
            generated_id = f"r{view_index:03d}"
        else:
            doc_index += 1
            generated_id = f"d{doc_index:03d}"
        region_id = str(item.get("region_id") or generated_id).strip()
        if not region_id:
            region_id = generated_id
        if region_id in used_ids:
            base = region_id
            suffix = 2
            while f"{base}_{suffix}" in used_ids:
                suffix += 1
            region_id = f"{base}_{suffix}"
            errors.append(f"step1.regions[{index}] had duplicate region_id; renamed")
        used_ids.add(region_id)
        output.append({"region_id": region_id, "category": category, "bbox": bbox})

    result = {"regions": output}
    errors.extend(
        f"step1 schema: {e}" for e in validate_stage_output(1, result, strict=True)
    )
    return result, errors


def _best_unassigned_region(
    bbox: Sequence[float],
    candidates: list[Mapping[str, Any]],
    assigned: set[str],
) -> str | None:
    scores = [
        (bbox_iou(bbox, region["bbox"]), str(region["region_id"]))
        for region in candidates
        if str(region["region_id"]) not in assigned
    ]
    if not scores:
        return None
    score, region_id = max(scores)
    return region_id if score > 0 else None


def normalize_step2(
    payload: Mapping[str, Any], step1: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    projection = normalize_projection_method(payload.get("projection_method"))
    if projection is None:
        projection = "unknown"
        errors.append("step2.projection_method was unknown; normalized to unknown")

    regions = [
        region
        for region in step1.get("regions", [])
        if region.get("category") == GENERIC_VIEW_REGION
    ]
    region_by_id = {str(region["region_id"]): region for region in regions}
    assigned: set[str] = set()
    views: list[dict[str, Any]] = []
    raw_views = payload.get("views", [])
    if not isinstance(raw_views, list):
        raw_views = []
        errors.append("step2.views is not a list")

    for index, item in enumerate(raw_views):
        if not isinstance(item, Mapping):
            errors.append(f"step2.views[{index}] is not an object")
            continue
        category = _canonical_category(
            item.get("category"), kind="view", allowed=VIEW_CATEGORIES
        )
        if category is None:
            errors.append(f"step2.views[{index}] has an unknown view category")
            continue
        region_id = str(item.get("region_id") or "").strip()
        bbox = normalize_bbox(item.get("bbox", item.get("bbox_2d")))
        if region_id not in region_by_id:
            matched = _best_unassigned_region(bbox, regions, assigned) if bbox else None
            if matched is None:
                remaining = [rid for rid in region_by_id if rid not in assigned]
                matched = remaining[0] if remaining else None
            if matched is None:
                errors.append(
                    f"step2.views[{index}] cannot be linked to a Stage-1 region"
                )
                continue
            errors.append(f"step2.views[{index}] region_id repaired to {matched}")
            region_id = matched
        if region_id in assigned:
            errors.append(
                f"step2.views[{index}] duplicates region_id {region_id}; dropped"
            )
            continue
        assigned.add(region_id)
        canonical_bbox = normalize_bbox(region_by_id[region_id]["bbox"])
        if bbox and bbox_iou(bbox, canonical_bbox) < 0.80:
            errors.append(
                f"step2.views[{index}] bbox drifted from Step 1; Step-1 bbox retained"
            )
        views.append(
            {"region_id": region_id, "category": category, "bbox": canonical_bbox}
        )

    # A missing classification is preserved as a diagnostic, not fabricated.
    missing = sorted(set(region_by_id) - assigned)
    if missing:
        errors.append(f"step2 did not classify Stage-1 view regions: {missing}")

    result = {"projection_method": projection, "views": views}
    errors.extend(
        f"step2 schema: {e}" for e in validate_stage_output(2, result, strict=True)
    )
    return result, errors


def normalize_layout_view(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Split one compact P2 response into the canonical Stage-1/Stage-2 forms."""

    errors: list[str] = []
    raw_regions = payload.get("regions", [])
    if not isinstance(raw_regions, list):
        raw_regions = []
        errors.append("layout_view.regions is not a list")

    step1_regions: list[dict[str, Any]] = []
    compact_views: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    view_index = doc_index = 0
    for index, item in enumerate(raw_regions):
        if not isinstance(item, Mapping):
            errors.append(f"layout_view.regions[{index}] is not an object")
            continue
        view_category = _canonical_category(
            item.get("category"), kind="view", allowed=VIEW_CATEGORIES
        )
        stage1_category = _canonical_category(
            item.get("category"), kind="stage1", allowed=STAGE1_CATEGORIES
        )
        bbox = normalize_bbox(item.get("bbox", item.get("bbox_2d")))
        if not bbox or (view_category is None and stage1_category is None):
            errors.append(
                f"layout_view.regions[{index}] has unknown category or invalid bbox"
            )
            continue

        is_view_region = (
            view_category is not None or stage1_category == GENERIC_VIEW_REGION
        )
        if is_view_region:
            view_index += 1
            generated_id = f"r{view_index:03d}"
        else:
            doc_index += 1
            generated_id = f"d{doc_index:03d}"
        region_id = str(item.get("region_id") or generated_id).strip() or generated_id
        if region_id in used_ids:
            base = region_id
            suffix = 2
            while f"{base}_{suffix}" in used_ids:
                suffix += 1
            region_id = f"{base}_{suffix}"
            errors.append(
                f"layout_view.regions[{index}] had duplicate region_id; renamed"
            )
        used_ids.add(region_id)

        if is_view_region:
            step1_regions.append(
                {
                    "region_id": region_id,
                    "category": GENERIC_VIEW_REGION,
                    "bbox": bbox,
                }
            )
            if view_category is not None:
                compact_views.append(
                    {"region_id": region_id, "category": view_category, "bbox": bbox}
                )
        else:
            step1_regions.append(
                {
                    "region_id": region_id,
                    "category": stage1_category,
                    "bbox": bbox,
                }
            )

    stage1, stage1_errors = normalize_step1({"regions": step1_regions})
    explicit_views = payload.get("views", [])
    if not isinstance(explicit_views, list):
        explicit_views = []
        errors.append("layout_view.views is not a list")
    explicit_ids = {
        str(item.get("region_id") or "").strip()
        for item in explicit_views
        if isinstance(item, Mapping)
    }
    combined_views = list(explicit_views)
    combined_views.extend(
        view
        for view in compact_views
        if str(view.get("region_id") or "") not in explicit_ids
    )
    stage2, stage2_errors = normalize_step2(
        {
            "projection_method": payload.get("projection_method"),
            "views": combined_views,
        },
        stage1,
    )
    errors.extend(stage1_errors + stage2_errors)
    return stage1, stage2, errors


def raw_layout_view_json_valid(raw_text: str) -> bool:
    """Require direct JSON and a repair-free compact P2 response."""

    text = str(raw_text or "").strip().lstrip("\ufeff")
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    if not isinstance(value, Mapping):
        return False
    if "projection_method" not in value or "regions" not in value:
        return False
    _, _, errors = normalize_layout_view(value)
    return not errors


def compact_layout_view(
    step1: Mapping[str, Any], step2: Mapping[str, Any]
) -> dict[str, Any]:
    """Rebuild the compact object used as canonical history for P2 Stage 2."""

    return {
        "projection_method": step2.get("projection_method", "unknown"),
        "regions": [
            {
                "region_id": region["region_id"],
                "category": region["category"],
                "bbox": region["bbox"],
            }
            for region in step1.get("regions", [])
        ],
        "views": [
            {"region_id": view["region_id"], "category": view["category"]}
            for view in step2.get("views", [])
        ],
    }


def normalize_feature_scope(values: Any, fallback: Sequence[str]) -> list[str]:
    raw = values if isinstance(values, list) else list(fallback)
    result: list[str] = []
    for value in raw:
        category = _canonical_category(
            value, kind="feature", allowed=FEATURE_CATEGORIES
        )
        if category and category not in result:
            result.append(category)
    return result or list(fallback)


def normalize_step3(
    payload: Mapping[str, Any],
    requested_scope: Sequence[str],
    *,
    default_region_id: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    reported_scope = normalize_feature_scope(
        payload.get("feature_scope"), requested_scope
    )
    requested = list(requested_scope)
    if set(reported_scope) != set(requested):
        errors.append(
            "step3.feature_scope differed from the requested exhaustive scope; request retained"
        )

    features: list[dict[str, Any]] = []
    raw_features = payload.get("features", [])
    if not isinstance(raw_features, list):
        raw_features = []
        errors.append("step3.features is not a list")
    allowed = set(requested)
    for index, item in enumerate(raw_features):
        if not isinstance(item, Mapping):
            errors.append(f"step3.features[{index}] is not an object")
            continue
        category = _canonical_category(
            item.get("category"), kind="feature", allowed=FEATURE_CATEGORIES
        )
        bbox = normalize_bbox(item.get("bbox", item.get("bbox_2d")))
        if category is None or category not in allowed or not bbox:
            errors.append(f"step3.features[{index}] is out of scope or malformed")
            continue
        feature = {
            "category": category,
            "size": str(item.get("size") or "").strip(),
            "bbox": bbox,
        }
        region_id = str(item.get("region_id") or default_region_id or "").strip()
        if region_id:
            feature["region_id"] = region_id
        features.append(feature)

    result = {"feature_scope": requested, "features": features}
    errors.extend(
        f"step3 schema: {e}" for e in validate_stage_output(3, result, strict=True)
    )
    return result, errors


def _resolve_device_map(value: str) -> Any:
    if value != "single":
        return value
    if torch.cuda.is_available():
        return {"": 0}
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return {"": "mps"}
    return {"": "cpu"}


def load_model(
    model_path: str,
    adapter_path: str,
    *,
    load_in_4bit: bool,
    device_map: str,
    attn_impl: str,
):
    """Load Qwen3.6 through its official auto class, with Qwen3.5 fallback."""

    import transformers
    from peft import PeftModel
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": _resolve_device_map(device_map),
    }
    try:
        transformers_major = int(str(transformers.__version__).split(".", 1)[0])
    except (AttributeError, TypeError, ValueError):
        transformers_major = 4
    # Transformers 5 renamed the public loading keyword; retain the v4 keyword
    # for Qwen3.5 fallback images.
    kwargs["dtype" if transformers_major >= 5 else "torch_dtype"] = torch.bfloat16
    if attn_impl != "auto":
        kwargs["attn_implementation"] = attn_impl
    else:
        try:
            import flash_attn  # noqa: F401

            kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            kwargs["attn_implementation"] = "sdpa"
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    class_names = (
        "AutoModelForMultimodalLM",  # official Qwen3.6 Transformers class
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    )
    available = [(name, getattr(transformers, name, None)) for name in class_names]
    available = [(name, cls) for name, cls in available if cls is not None]
    if not available:
        raise RuntimeError(
            "No compatible multimodal auto-model class exists in transformers"
        )

    model = None
    mapping_errors: list[str] = []
    for name, model_class in available:
        LOGGER.info("Loading %s with %s", model_path, name)
        try:
            model = model_class.from_pretrained(model_path, **kwargs)
            break
        except (KeyError, TypeError, ValueError) as exc:
            # Only mapping/signature failures justify another auto class.  CUDA
            # OOM, corrupt weights and network failures must surface directly.
            mapping_errors.append(f"{name}: {exc}")
    if model is None:
        raise RuntimeError(
            "No auto-model class accepted this config: " + " | ".join(mapping_errors)
        )

    LOGGER.info("Loading PEFT adapter: %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, processor


def _input_device(model: Any) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, RuntimeError):
        return next(model.parameters()).device


def generate(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int,
) -> str:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise RuntimeError(
            "qwen-vl-utils>=0.0.14 is required for multimodal inference"
        ) from exc

    try:
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:  # older Qwen3.5 processor
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )

    image_inputs, video_inputs = process_vision_info(messages)
    processor_kwargs: dict[str, Any] = {
        "text": [prompt],
        "images": image_inputs,
        "return_tensors": "pt",
        "padding": True,
    }
    if video_inputs:
        processor_kwargs["videos"] = video_inputs
    inputs = processor(**processor_kwargs)
    device = _input_device(model)
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    prompt_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    new_tokens = generated[0, prompt_length:]
    decoder = getattr(processor, "tokenizer", processor)
    return decoder.decode(new_tokens, skip_special_tokens=True).strip()


def _image_message(image: Any, text: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ],
    }


def _crop_image(
    image: Image.Image, region_bbox: Sequence[float], padding: float
) -> tuple[Image.Image, list[int]]:
    expanded = normalize_bbox(
        [
            region_bbox[0] - (region_bbox[2] - region_bbox[0]) * padding,
            region_bbox[1] - (region_bbox[3] - region_bbox[1]) * padding,
            region_bbox[2] + (region_bbox[2] - region_bbox[0]) * padding,
            region_bbox[3] + (region_bbox[3] - region_bbox[1]) * padding,
        ]
    )
    width, height = image.size
    pixel_box = (
        max(0, int(math.floor(expanded[0] / 1000 * width))),
        max(0, int(math.floor(expanded[1] / 1000 * height))),
        min(width, int(math.ceil(expanded[2] / 1000 * width))),
        min(height, int(math.ceil(expanded[3] / 1000 * height))),
    )
    if pixel_box[0] >= pixel_box[2] or pixel_box[1] >= pixel_box[3]:
        raise ValueError(f"invalid crop for bbox={region_bbox}")
    actual_global_box = normalize_bbox(
        [
            pixel_box[0] / width * 1000,
            pixel_box[1] / height * 1000,
            pixel_box[2] / width * 1000,
            pixel_box[3] / height * 1000,
        ]
    )
    return image.crop(pixel_box), actual_global_box


def _feature_key_size(value: Any, category: Any) -> str:
    return normalize_size(value, category=str(category or ""))


def deduplicate_features(
    features: list[dict[str, Any]],
    iou_threshold: float,
    *,
    size_conflicts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Class+IoU dedup, preferring crop geometry/size over full-image output."""

    ordered = sorted(
        features, key=lambda item: int(item.get("_priority", 0)), reverse=True
    )
    kept: list[dict[str, Any]] = []
    for feature in ordered:
        duplicate = False
        for existing in kept:
            if feature.get("category") != existing.get("category"):
                continue
            if (
                bbox_iou(feature.get("bbox", []), existing.get("bbox", []))
                >= iou_threshold
            ):
                size_a = _feature_key_size(feature.get("size"), feature.get("category"))
                size_b = _feature_key_size(
                    existing.get("size"), existing.get("category")
                )
                if not size_b and size_a:
                    existing["size"] = feature["size"]
                elif (
                    size_a
                    and size_b
                    and size_a != size_b
                    and size_conflicts is not None
                ):
                    size_conflicts.append(
                        {
                            "category": existing.get("category"),
                            "kept_size": existing.get("size", ""),
                            "discarded_size": feature.get("size", ""),
                            "kept_bbox": existing.get("bbox", []),
                            "discarded_bbox": feature.get("bbox", []),
                        }
                    )
                duplicate = True
                break
        if not duplicate:
            cleaned = {
                key: value for key, value in feature.items() if not key.startswith("_")
            }
            kept.append(cleaned)
    return kept


def _crop_targets(
    step1: Mapping[str, Any], step2: Mapping[str, Any]
) -> list[dict[str, Any]]:
    classified = [dict(view) for view in step2.get("views", []) if view.get("bbox")]
    classified_ids = {str(view.get("region_id")) for view in classified}
    # If Step 2 misses a region, still run feature extraction on it.  We do not
    # fabricate a view label; the context explicitly says unclassified.
    for region in step1.get("regions", []):
        if (
            region.get("category") == GENERIC_VIEW_REGION
            and str(region.get("region_id")) not in classified_ids
            and region.get("bbox")
        ):
            classified.append(
                {
                    "region_id": region["region_id"],
                    "category": GENERIC_VIEW_REGION,
                    "bbox": region["bbox"],
                }
            )
    return classified


def infer_one(
    model: Any,
    processor: Any,
    image_path: Path,
    *,
    protocol_mode: str,
    feature_mode: str,
    feature_scope: Sequence[str],
    crop_padding: float,
    dedup_iou: float,
    max_crops: int,
    layout_view_tokens: int,
    step1_tokens: int,
    step2_tokens: int,
    step3_tokens: int,
    save_raw: bool,
    projection_override: str | None,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "protocol_mode": protocol_mode,
        "warnings": [],
        "crop_count": 0,
        "raw_json_valid": {"step3_crops": {}},
        "timings_seconds": {},
    }
    raw_outputs: dict[str, Any] = {}
    image_uri = image_path.resolve().as_uri()

    if protocol_mode == "three_stage":
        messages1 = [
            {"role": "system", "content": SYSTEM_PROMPT},
            _image_message(image_uri, step1_prompt()),
        ]
        call_started = time.monotonic()
        raw1 = generate(model, processor, messages1, max_new_tokens=step1_tokens)
        diagnostics["timings_seconds"]["step1"] = round(
            time.monotonic() - call_started, 3
        )
        diagnostics["raw_json_valid"]["step1"] = raw_stage_json_valid(raw1, 1)
        parsed1, parse_errors = parse_stage_json(raw1, 1)
        stage1, normal_errors = normalize_step1(parsed1)
        diagnostics["warnings"].extend(parse_errors + normal_errors)
        raw_outputs["step1"] = raw1

        canonical1 = _json_text(stage1)
        messages2 = messages1 + [
            {"role": "assistant", "content": canonical1},
            {"role": "user", "content": step2_prompt()},
        ]
        call_started = time.monotonic()
        raw2 = generate(model, processor, messages2, max_new_tokens=step2_tokens)
        diagnostics["timings_seconds"]["step2"] = round(
            time.monotonic() - call_started, 3
        )
        diagnostics["raw_json_valid"]["step2"] = raw_stage_json_valid(raw2, 2)
        parsed2, parse_errors = parse_stage_json(raw2, 2)
        stage2, normal_errors = normalize_step2(parsed2, stage1)
        diagnostics["warnings"].extend(parse_errors + normal_errors)
        raw_outputs["step2"] = raw2
        feature_history = messages2 + [
            {"role": "assistant", "content": _json_text(stage2)}
        ]
    elif protocol_mode == "two_stage":
        layout_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            _image_message(image_uri, layout_view_prompt()),
        ]
        call_started = time.monotonic()
        raw_layout_view = generate(
            model,
            processor,
            layout_messages,
            max_new_tokens=layout_view_tokens,
        )
        diagnostics["timings_seconds"]["layout_view"] = round(
            time.monotonic() - call_started, 3
        )
        diagnostics["raw_json_valid"]["layout_view"] = raw_layout_view_json_valid(
            raw_layout_view
        )
        parsed_layout_view, parse_errors = parse_layout_view_json(raw_layout_view)
        stage1, stage2, normal_errors = normalize_layout_view(parsed_layout_view)
        diagnostics["warnings"].extend(parse_errors + normal_errors)
        raw_outputs["layout_view"] = raw_layout_view
        feature_history = layout_messages + [
            {
                "role": "assistant",
                "content": _json_text(compact_layout_view(stage1, stage2)),
            }
        ]
    else:
        raise ValueError(f"unsupported protocol_mode: {protocol_mode!r}")

    if projection_override is not None:
        stage2["projection_method"] = projection_override
        diagnostics["projection_override"] = projection_override
        feature_history[-1]["content"] = _json_text(
            stage2
            if protocol_mode == "three_stage"
            else compact_layout_view(stage1, stage2)
        )

    candidates: list[dict[str, Any]] = []
    if feature_mode in {"full", "hybrid"}:
        messages3 = feature_history + [
            {
                "role": "user",
                "content": step3_prompt(feature_scope, crop_context=False),
            },
        ]
        call_started = time.monotonic()
        raw3 = generate(model, processor, messages3, max_new_tokens=step3_tokens)
        diagnostics["timings_seconds"]["step3_full"] = round(
            time.monotonic() - call_started, 3
        )
        diagnostics["raw_json_valid"]["step3_full"] = raw_stage_json_valid(raw3, 3)
        parsed3, parse_errors = parse_stage_json(raw3, 3)
        full_stage3, normal_errors = normalize_step3(parsed3, feature_scope)
        diagnostics["warnings"].extend(parse_errors + normal_errors)
        for feature in full_stage3["features"]:
            candidates.append({**feature, "_priority": 1, "_source": "full"})
        raw_outputs["step3_full"] = raw3

    if feature_mode in {"crops", "hybrid"}:
        with Image.open(image_path) as opened:
            full_image = ImageOps.exif_transpose(opened).convert("RGB")
        targets = _crop_targets(stage1, stage2)
        if max_crops > 0:
            targets = targets[:max_crops]
        raw_crop_outputs: dict[str, str] = {}
        for view in targets:
            region_id = str(view["region_id"])
            try:
                crop, crop_global_bbox = _crop_image(
                    full_image, view["bbox"], crop_padding
                )
                context = {
                    "projection_method": stage2["projection_method"],
                    "view": {"region_id": region_id, "category": view["category"]},
                }
                prompt = (
                    step3_prompt(feature_scope, region_id=region_id, crop_context=True)
                    + "\n已知整图阶段上下文："
                    + _json_text(context)
                )
                crop_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    _image_message(crop, prompt),
                ]
                call_started = time.monotonic()
                raw_crop = generate(
                    model, processor, crop_messages, max_new_tokens=step3_tokens
                )
                diagnostics["timings_seconds"][f"step3_crop_{region_id}"] = round(
                    time.monotonic() - call_started, 3
                )
                diagnostics["raw_json_valid"]["step3_crops"][region_id] = (
                    raw_stage_json_valid(raw_crop, 3)
                )
                parsed_crop, parse_errors = parse_stage_json(raw_crop, 3)
                crop_stage3, normal_errors = normalize_step3(
                    parsed_crop, feature_scope, default_region_id=region_id
                )
                diagnostics["warnings"].extend(
                    f"crop {region_id}: {error}"
                    for error in parse_errors + normal_errors
                )
                for feature in crop_stage3["features"]:
                    mapped = crop_to_global_bbox(feature["bbox"], crop_global_bbox)
                    if not mapped:
                        diagnostics["warnings"].append(
                            f"crop {region_id}: feature bbox could not be mapped"
                        )
                        continue
                    candidates.append(
                        {
                            **feature,
                            "bbox": mapped,
                            "region_id": region_id,
                            "_priority": 2,
                            "_source": "crop",
                        }
                    )
                raw_crop_outputs[region_id] = raw_crop
                diagnostics["crop_count"] += 1
            except Exception as exc:  # one bad crop must not discard the drawing
                LOGGER.exception(
                    "Crop inference failed for %s/%s", image_path.name, region_id
                )
                diagnostics["warnings"].append(f"crop {region_id} failed: {exc}")
                diagnostics["raw_json_valid"]["step3_crops"][region_id] = False
        raw_outputs["step3_crops"] = raw_crop_outputs

    size_conflicts: list[dict[str, Any]] = []
    final_features = deduplicate_features(
        candidates, dedup_iou, size_conflicts=size_conflicts
    )
    stage3 = {"feature_scope": list(feature_scope), "features": final_features}
    diagnostics["warnings"].extend(
        f"final step3 schema: {e}"
        for e in validate_stage_output(3, stage3, strict=True)
    )
    diagnostics["feature_candidates"] = len(candidates)
    diagnostics["feature_count"] = len(final_features)
    diagnostics["size_conflicts"] = size_conflicts
    validity_values = [
        bool(value)
        for key, value in diagnostics["raw_json_valid"].items()
        if key != "step3_crops"
    ]
    validity_values.extend(diagnostics["raw_json_valid"]["step3_crops"].values())
    diagnostics["raw_json_valid"]["overall"] = bool(validity_values) and all(
        validity_values
    )
    constraint_errors = validate_cross_stage_outputs(stage1, stage2, stage3)
    diagnostics["constraint_errors"] = constraint_errors
    diagnostics["constraint_valid"] = not constraint_errors
    diagnostics["warnings"].extend(
        f"cross-stage constraint: {error}" for error in constraint_errors
    )

    # Compatibility projection for existing platform/evaluation consumers.  A
    # generic View Region is not a semantic prediction and is therefore omitted.
    legacy_result: list[dict[str, Any]] = []
    for region in stage1["regions"]:
        if region["category"] in DOCUMENT_CATEGORIES:
            legacy_result.append(
                {"category": region["category"], "size": "", "bbox": region["bbox"]}
            )
    for view in stage2["views"]:
        legacy_result.append(
            {"category": view["category"], "size": "", "bbox": view["bbox"]}
        )
    legacy_result.extend(
        {
            "category": feature["category"],
            "size": feature["size"],
            "bbox": feature["bbox"],
        }
        for feature in final_features
    )

    result: dict[str, Any] = {
        "step1": stage1,
        "step2": stage2,
        "step3": stage3,
        "result": legacy_result,
        "diagnostics": diagnostics,
    }
    if save_raw:
        result["raw_outputs"] = raw_outputs
    return result


def _extract_image_from_record(
    record: Mapping[str, Any], base_dir: Path
) -> Path | None:
    candidates: list[Any] = []
    if record.get("image"):
        candidates.append(record["image"])
    if record.get("image_path"):
        candidates.append(record["image_path"])
    if isinstance(record.get("images"), list):
        candidates.extend(record["images"])
    for message in (
        record.get("messages", []) if isinstance(record.get("messages"), list) else []
    ):
        content = message.get("content", []) if isinstance(message, Mapping) else []
        if isinstance(content, list):
            for part in content:
                if isinstance(part, Mapping) and part.get("type") in {
                    "image",
                    "image_url",
                }:
                    candidates.append(part.get("image") or part.get("image_url"))
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            candidate = candidate.get("url")
        if not isinstance(candidate, str) or candidate.startswith(
            ("http://", "https://")
        ):
            continue
        path_text = candidate.removeprefix("file://")
        path = Path(path_text).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        if path.exists() and path.suffix.lower() in IMAGE_SUFFIXES:
            return path.resolve()
    return None


def _read_records(path: Path) -> list[Mapping[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        value = json.loads(text)
        return [item for item in value if isinstance(item, Mapping)]
    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
        if isinstance(value, Mapping):
            records.append(value)
    return records


def collect_images(args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    items: OrderedDict[str, tuple[str, Path]] = OrderedDict()
    for value in args.image:
        path = Path(value).expanduser().resolve()
        items[str(path)] = (path.name, path)
    if args.image_dir:
        image_dir = Path(args.image_dir).expanduser().resolve()
        iterator = image_dir.rglob("*") if args.recursive else image_dir.glob("*")
        for path in sorted(iterator):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                resolved = path.resolve()
                items[str(resolved)] = (resolved.name, resolved)
    if args.input_jsonl:
        input_path = Path(args.input_jsonl).expanduser().resolve()
        for record in _read_records(input_path):
            image = _extract_image_from_record(record, input_path.parent)
            if image is None:
                LOGGER.warning(
                    "No accessible image in input record: %s",
                    record.get("sample_id", "?"),
                )
                continue
            sample_id = str(record.get("sample_id") or image.name)
            items[str(image)] = (sample_id, image)
    result = [(sample_id, path.name, path) for sample_id, path in items.values()]
    if args.limit > 0:
        result = result[: args.limit]
    return result


def load_existing(path: Path) -> OrderedDict[str, dict[str, Any]]:
    existing: OrderedDict[str, dict[str, Any]] = OrderedDict()
    if not path.exists() or path.stat().st_size == 0:
        return existing
    for record in _read_records(path):
        key = str(record.get("image") or record.get("dataitem_name") or "")
        if key:
            existing[key] = dict(record)
    return existing


def atomic_save(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    records_list = list(records)
    with temporary.open("w", encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            for record in records_list:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            json.dump(records_list, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_scope(values: list[str] | None) -> list[str]:
    if not values:
        return list(FEATURE_CATEGORIES)
    flattened: list[str] = []
    for value in values:
        flattened.extend(part.strip() for part in value.split(",") if part.strip())
    result: list[str] = []
    unknown: list[str] = []
    for value in flattened:
        category = _canonical_category(
            value, kind="feature", allowed=FEATURE_CATEGORIES
        )
        if category is None:
            unknown.append(value)
        elif category not in result:
            result.append(category)
    if unknown:
        raise ValueError(
            f"Unknown feature_scope categories (explicit aliases only): {unknown}"
        )
    if not result:
        raise ValueError("feature_scope cannot be empty")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", "--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", "--adapter_path", required=True)
    parser.add_argument("--image", action="append", default=[], help="May be repeated")
    parser.add_argument("--image-dir", "--image_dir")
    parser.add_argument("--input-jsonl", "--input_jsonl", help="Dataset JSONL/JSON")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--output-file", "--output_file", default="predictions_three_stage.jsonl"
    )
    parser.add_argument(
        "--feature-mode", choices=("full", "crops", "hybrid"), default="hybrid"
    )
    parser.add_argument(
        "--protocol-mode",
        choices=("three_stage", "two_stage"),
        default="three_stage",
        help="three_stage=layout->view->features; two_stage=layout+view->features",
    )
    parser.add_argument(
        "--projection-method",
        choices=("auto", "first_angle", "third_angle", "unknown"),
        default="auto",
        help="Override only when drawing metadata/project standard has been verified",
    )
    parser.add_argument(
        "--feature-scope",
        nargs="*",
        help="Canonical/explicit-alias categories; default is all 19 features",
    )
    parser.add_argument("--crop-padding", type=float, default=0.12)
    parser.add_argument("--dedup-iou", type=float, default=0.55)
    parser.add_argument(
        "--max-crops", type=int, default=0, help="0 means all view regions"
    )
    parser.add_argument("--step1-max-new-tokens", type=int, default=1536)
    parser.add_argument("--step2-max-new-tokens", type=int, default=1536)
    parser.add_argument("--layout-view-max-new-tokens", type=int, default=2048)
    parser.add_argument("--step3-max-new-tokens", type=int, default=3072)
    parser.add_argument(
        "--device-map", choices=("auto", "balanced", "single"), default="auto"
    )
    parser.add_argument("--attn-impl", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if not (0 <= args.crop_padding <= 1):
        raise ValueError("--crop-padding must be in [0,1]")
    if not (0 < args.dedup_iou <= 1):
        raise ValueError("--dedup-iou must be in (0,1]")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")

    feature_scope = parse_scope(args.feature_scope)
    images = collect_images(args)
    if not images:
        LOGGER.error(
            "No accessible images found; use --image, --image-dir, or --input-jsonl"
        )
        return 2

    output_path = Path(args.output_file).expanduser().resolve()
    if args.overwrite:
        existing: OrderedDict[str, dict[str, Any]] = OrderedDict()
    else:
        existing = load_existing(output_path)
    complete_paths = {
        str(record.get("image"))
        for record in existing.values()
        if record.get("status") == "ok" and record.get("image")
    }
    pending = [
        (sample_id, name, path)
        for sample_id, name, path in images
        if str(path) not in complete_paths
    ]
    LOGGER.info(
        "Images total=%d complete=%d pending=%d protocol=%s mode=%s scope=%d",
        len(images),
        len(images) - len(pending),
        len(pending),
        args.protocol_mode,
        args.feature_mode,
        len(feature_scope),
    )
    if not pending:
        LOGGER.info("Nothing to do; output already contains all successful images")
        return 0

    model, processor = load_model(
        args.model_path,
        args.adapter_path,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
        attn_impl=args.attn_impl,
    )
    started = time.time()
    dirty = 0
    try:
        for index, (sample_id, name, image_path) in enumerate(pending, 1):
            item_started = time.time()
            LOGGER.info("[%d/%d] %s", index, len(pending), image_path)
            try:
                inference = infer_one(
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
                        None
                        if args.projection_method == "auto"
                        else args.projection_method
                    ),
                )
                record = {
                    "sample_id": sample_id,
                    "dataitem_name": name,
                    "image": str(image_path),
                    "status": "ok",
                    "protocol_mode": args.protocol_mode,
                    "feature_mode": args.feature_mode,
                    **inference,
                    "elapsed_seconds": round(time.time() - item_started, 3),
                }
            except Exception as exc:
                LOGGER.exception("Inference failed: %s", image_path)
                record = {
                    "sample_id": sample_id,
                    "dataitem_name": name,
                    "image": str(image_path),
                    "status": "error",
                    "protocol_mode": args.protocol_mode,
                    "feature_mode": args.feature_mode,
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": round(time.time() - item_started, 3),
                }
                if args.fail_fast:
                    existing[str(image_path)] = record
                    atomic_save(output_path, existing.values())
                    raise
            existing[str(image_path)] = record
            dirty += 1
            if dirty >= args.save_every:
                atomic_save(output_path, existing.values())
                dirty = 0
            average = (time.time() - started) / index
            eta = average * (len(pending) - index)
            LOGGER.info(
                "[%d/%d] status=%s elapsed=%.1fs ETA=%.1fmin",
                index,
                len(pending),
                record["status"],
                record["elapsed_seconds"],
                eta / 60,
            )
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; saving completed records before exit")
        atomic_save(output_path, existing.values())
        return 130
    finally:
        if dirty:
            atomic_save(output_path, existing.values())

    LOGGER.info("Done: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
