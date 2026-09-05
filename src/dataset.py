"""
Dataset module for industrial drawing feature recognition training.

Supports multiple JSON annotation formats, including Label Studio export.

--- Format 1: Label Studio JSON export (primary) ---
[
    {
        "id": 1,
        "data": {
            "image": "/data/upload/1/drawing.png"
        },
        "annotations": [
            {
                "result": [
                    {
                        "type": "textarea",
                        "value": {"text": ["这张图纸中包含以下工件特征：..."]},
                        "from_name": "answer",
                        "to_name": "image"
                    },
                    {
                        "type": "rectanglelabels",
                        "value": {
                            "x": 10.0, "y": 20.0,
                            "width": 5.0, "height": 8.0,
                            "rectanglelabels": ["螺纹孔"]
                        },
                        "from_name": "label",
                        "to_name": "image"
                    }
                ]
            }
        ]
    }
]

--- Format 2: Conversation format ---
[
    {
        "image": "path/to/image.png",
        "conversations": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
]

--- Format 3: Simple Q&A ---
[
    {
        "image": "path/to/image.png",
        "question": "...",
        "answer": "..."
    }
]

--- Format 4: Structured with bounding boxes ---
[
    {
        "image": "path/to/image.png",
        "question": "...",
        "answer": "...",
        "features": [
            {"label": "螺纹孔", "bbox": [x1, y1, x2, y2]}
        ]
    }
]
"""

import json
import os
import copy
import logging
from typing import Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)


def _is_label_studio_format(data: list) -> bool:
    """Check if json data is in Label Studio export format."""
    if not data or not isinstance(data[0], dict):
        return False
    sample = data[0]
    return "data" in sample and ("annotations" in sample or "completions" in sample)


def _parse_label_studio_sample(sample: dict, default_prompt: str) -> dict:
    """
    Convert a single Label Studio annotation into our normalized format.

    Handles common Label Studio result types:
    - textarea: Text answers / descriptions
    - choices: Classification / multiple-choice labels
    - rectanglelabels: Bounding box annotations
    - polygonlabels: Polygon annotations
    - labels (NER-style): Span labels on text
    """
    # --- Extract image path ---
    data = sample.get("data", {})
    image_path = None
    for k in ["image", "image_url", "img", "img_url", "image_path"]:
        if k in data:
            image_path = data[k]
            break
    if image_path is None:
        # Fallback: take first value that looks like an image path
        for v in data.values():
            if isinstance(v, str) and any(
                v.lower().endswith(ext)
                for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")
            ):
                image_path = v
                break
    if image_path is None:
        raise ValueError(f"No image found in Label Studio sample data: {list(data.keys())}")

    # Clean up Label Studio image path prefixes
    # e.g. "/data/upload/1/image.png" -> "image.png"
    #      "/data/local-files/?d=images/img.png" -> "images/img.png"
    if image_path.startswith("/data/upload/"):
        # /data/upload/<project_id>/<filename>
        parts = image_path.split("/")
        image_path = "/".join(parts[4:])  # keep only the filename part
    elif image_path.startswith("/data/local-files/"):
        # /data/local-files/?d=<path>
        if "?d=" in image_path:
            image_path = image_path.split("?d=", 1)[1]
        else:
            image_path = image_path.split("/data/local-files/", 1)[1]
    elif image_path.startswith("s3://") or image_path.startswith("gs://"):
        # Cloud storage: keep filename only
        image_path = image_path.split("/")[-1]

    # --- Extract annotations ---
    annotations = sample.get("annotations", sample.get("completions", []))
    if not annotations:
        raise ValueError("No annotations found in Label Studio sample")

    # Use the first (or latest) annotation
    annotation = annotations[0]
    results = annotation.get("result", [])

    # Categorize results by type
    text_answers = []
    choices = []
    bboxes = []
    polygons = []
    other_labels = []

    for result in results:
        rtype = result.get("type", "")
        value = result.get("value", {})

        if rtype == "textarea":
            texts = value.get("text", [])
            if isinstance(texts, list):
                text_answers.extend(texts)
            elif isinstance(texts, str):
                text_answers.append(texts)

        elif rtype == "choices":
            ch = value.get("choices", [])
            choices.extend(ch)

        elif rtype == "rectanglelabels":
            labels = value.get("rectanglelabels", [])
            bbox_info = {
                "label": labels[0] if labels else "unknown",
                "x": value.get("x", 0),
                "y": value.get("y", 0),
                "width": value.get("width", 0),
                "height": value.get("height", 0),
                # Original dimensions for absolute coordinate computation
                "original_width": value.get("original_width"),
                "original_height": value.get("original_height"),
            }
            bboxes.append(bbox_info)

        elif rtype == "polygonlabels":
            labels = value.get("polygonlabels", [])
            points = value.get("points", [])
            polygons.append({
                "label": labels[0] if labels else "unknown",
                "points": points,
            })

        elif rtype == "labels":
            # NER-style span annotations on text
            labels = value.get("labels", [])
            other_labels.extend(labels)

    # --- Build the question and answer ---
    # Extract user prompt from data if available
    prompt = None
    for k in ["prompt", "question", "instruction", "text"]:
        if k in data and isinstance(data[k], str):
            prompt = data[k]
            break
    if prompt is None:
        prompt = default_prompt

    # Build answer from annotation results
    answer_parts = []

    # Text answers from textarea
    if text_answers:
        answer_parts.extend(text_answers)

    # Classification choices
    if choices:
        answer_parts.append("分类标签: " + ", ".join(choices))

    # Bounding box detections
    if bboxes:
        bbox_strs = []
        for b in bboxes:
            ow = b.get("original_width")
            oh = b.get("original_height")
            if ow and oh:
                # Convert percentage to absolute pixels
                x1 = round(b["x"] * ow / 100)
                y1 = round(b["y"] * oh / 100)
                x2 = round((b["x"] + b["width"]) * ow / 100)
                y2 = round((b["y"] + b["height"]) * oh / 100)
                bbox_strs.append(f"- {b['label']}: 位置 [{x1}, {y1}, {x2}, {y2}]")
            else:
                # Keep percentage coordinates
                bbox_strs.append(
                    f"- {b['label']}: 位置 (x={b['x']:.1f}%, y={b['y']:.1f}%, "
                    f"w={b['width']:.1f}%, h={b['height']:.1f}%)"
                )
        answer_parts.append("检测到的特征:\n" + "\n".join(bbox_strs))

    # Polygon annotations
    if polygons:
        poly_strs = []
        for p in polygons:
            poly_strs.append(f"- {p['label']}: 多边形区域 ({len(p['points'])} 个点)")
        answer_parts.append("区域标注:\n" + "\n".join(poly_strs))

    if not answer_parts:
        raise ValueError("No usable annotation results found in Label Studio sample")

    answer = "\n\n".join(answer_parts)

    return {
        "image": image_path,
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def load_json_data(json_path: str, default_prompt: str = "请识别这张图纸中的所有工件特征。") -> list:
    """Load and validate JSON annotation data. Auto-detects Label Studio format."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Handle {"data": [...]} or {"samples": [...]} wrapper
        for key in ["data", "samples", "annotations", "items"]:
            if key in data:
                data = data[key]
                break
        else:
            raise ValueError(
                f"JSON root is a dict but doesn't contain expected keys. "
                f"Found keys: {list(data.keys())}"
            )

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of samples, got {type(data)}")

    # Auto-detect Label Studio format and pre-convert
    if _is_label_studio_format(data):
        logger.info("Detected Label Studio export format, converting...")
        converted = []
        for i, sample in enumerate(data):
            try:
                converted.append(_parse_label_studio_sample(sample, default_prompt))
            except Exception as e:
                logger.warning(f"Label Studio sample {i} (id={sample.get('id', '?')}): {e}")
        data = converted
        logger.info(f"Converted {len(data)} Label Studio samples")

    logger.info(f"Loaded {len(data)} samples from {json_path}")
    return data


def normalize_sample(sample: dict) -> dict:
    """
    Normalize different JSON formats into a unified conversation format.

    Returns dict with keys: "image", "conversations"
    """
    normalized = {}

    # --- Image path ---
    image_key = None
    for k in ["image", "image_path", "img", "img_path", "file_name", "filename"]:
        if k in sample:
            image_key = k
            break
    # Also check nested "data" dict (Label Studio already pre-converted, but just in case)
    if image_key is None and "data" in sample and isinstance(sample["data"], dict):
        for k in ["image", "image_url", "img", "img_path", "image_path"]:
            if k in sample["data"]:
                image_key = k
                sample[image_key] = sample["data"][k]
                break
    if image_key is None:
        raise ValueError(f"No image field found in sample. Keys: {list(sample.keys())}")
    normalized["image"] = sample[image_key]

    # --- Conversations ---
    # Accept both "conversations" (plural) and "conversation" (singular, sample.json style)
    _conv_raw = sample.get("conversations") if "conversations" in sample else sample.get("conversation")
    if _conv_raw is not None:
        normalized["conversations"] = []
        for turn in _conv_raw:
            role = turn.get("role", turn.get("from", ""))
            content = turn.get("content", turn.get("value", ""))
            # Normalize role names; "qwen" is the assistant in sample.json format
            if role in ("user", "human"):
                role = "user"
            elif role in ("assistant", "gpt", "bot", "model", "qwen"):
                role = "assistant"
            # Serialize list content (JSON detection arrays) to a string
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            normalized["conversations"].append({"role": role, "content": content})
    elif "question" in sample and "answer" in sample:
        question = sample["question"]
        answer = sample["answer"]

        # If structured features exist, append them to the answer
        if "features" in sample and isinstance(sample["features"], list):
            feature_strs = []
            for feat in sample["features"]:
                label = feat.get("label", "unknown")
                bbox = feat.get("bbox", None)
                if bbox:
                    feature_strs.append(f"- {label}: 位置 {bbox}")
                else:
                    feature_strs.append(f"- {label}")
            answer = answer + "\n\n检测到的特征:\n" + "\n".join(feature_strs)

        normalized["conversations"] = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    elif "instruction" in sample and "output" in sample:
        # Alpaca-like format
        instruction = sample["instruction"]
        inp = sample.get("input", "")
        output = sample["output"]
        question = f"{instruction}\n{inp}".strip() if inp else instruction
        normalized["conversations"] = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": output},
        ]
    else:
        raise ValueError(
            f"Cannot parse sample format. Keys: {list(sample.keys())}. "
            f"Expected 'conversations', 'question'+'answer', or 'instruction'+'output'."
        )

    return normalized


class DrawingFeatureDataset(Dataset):
    """
    Dataset for fine-tuning VLMs on industrial drawing feature recognition.

    Handles loading images and formatting conversations for Qwen3-VL / Qwen2.5-VL.
    Supports Label Studio export, conversation, Q&A, and instruction formats.
    """

    def __init__(
        self,
        json_path: str,
        image_root: str,
        processor,
        system_prompt: str = "",
        max_seq_length: int = 4096,
        max_samples: Optional[int] = None,
        min_pixels: int = 256,
        max_pixels: int = 1280,
    ):
        self.image_root = Path(image_root)
        self.processor = processor
        self.system_prompt = system_prompt
        self.max_seq_length = max_seq_length
        # min_pixels / max_pixels in config = number of 28×28 visual-token patches.
        # Total pixel count = n_patches * 28 * 28 (Qwen-VL convention).
        # E.g. 256 patches → 200,704 px (≈448×448); 1280 patches → 1,003,520 px (≈1002×1002).
        self.min_pixels = min_pixels * 28 * 28
        self.max_pixels = max_pixels * 28 * 28

        # Load and normalize data
        raw_data = load_json_data(json_path, default_prompt=self._get_default_prompt())
        self.data = []
        for i, sample in enumerate(raw_data):
            try:
                normalized = normalize_sample(sample)
                # Verify image exists
                img_path = self._resolve_image_path(normalized["image"])
                if img_path.exists():
                    self.data.append(normalized)
                else:
                    logger.warning(f"Sample {i}: Image not found: {img_path}")
            except Exception as e:
                logger.warning(f"Sample {i}: Failed to normalize: {e}")

        if max_samples is not None:
            self.data = self.data[:max_samples]

        logger.info(f"Dataset initialized with {len(self.data)} valid samples")

    def _get_default_prompt(self) -> str:
        """Get default prompt for Label Studio samples that don't have a question."""
        return "请识别这张图纸中的所有工件特征，并详细描述。"

    def _resolve_image_path(self, image_name: str) -> Path:
        """Resolve image path relative to image_root.

        Tries the path as-is first (handles full relative paths like
        "data/IM_D03_PT_5K/image.png" from sample.json format), then falls
        back to joining with image_root.
        """
        p = Path(image_name)
        if p.is_absolute() and p.exists():
            return p
        if p.exists():
            return p
        return self.image_root / image_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]
        image_path = str(self._resolve_image_path(sample["image"]))
        conversations = sample["conversations"]

        # Validate image is readable before sending to the processor
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as e:
            logger.warning(f"Corrupt or unreadable image at index {idx} ({image_path}): {e}. "
                           "Falling back to a neighbouring sample.")
            fallback_idx = (idx + 1) % len(self.data)
            return self.__getitem__(fallback_idx)

        # Build Qwen3-VL / Qwen2.5-VL message format
        messages = []

        # System message
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})

        # User/Assistant turns
        for i, turn in enumerate(conversations):
            role = turn["role"]
            content_parts = []

            # Add image to the first user message
            if role == "user" and i == 0:
                content_parts.append({
                    "type": "image",
                    "image": f"file://{image_path}",
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                })

            # Strip <image> placeholder — the image is already added as a
            # content part above, so the placeholder would cause a duplicate.
            text = turn["content"].replace("<image>", "")
            content_parts.append({"type": "text", "text": text})
            messages.append({"role": role, "content": content_parts})

        # Apply chat template to get the full text
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        # Tokenize
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )

        # Squeeze batch dimension for sequence tensors.
        # image_grid_thw has shape [n_images, 3] — do NOT squeeze it, otherwise
        # a single-image sample becomes shape [3] and torch.cat in the collator
        # produces a flat [batch*3] tensor instead of [batch, 3], causing
        # "iteration over a 0-d tensor" in Qwen2VL's rot_pos_emb.
        inputs = {k: v.squeeze(0) if k != "image_grid_thw" else v
                  for k, v in inputs.items()}

        # Create labels: mask everything except the assistant response
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()

        # Find assistant response boundaries and mask non-target tokens
        # Mask system + user tokens with -100
        labels = self._create_labels(input_ids, messages)
        inputs["labels"] = labels

        return inputs

    def _create_labels(self, input_ids: torch.Tensor, messages: list) -> torch.Tensor:
        """
        Create labels tensor, masking non-assistant tokens with -100.
        Only compute loss on assistant responses.

        Uses token-level pattern matching instead of text decoding so that
        visual tokens interspersed in input_ids do not corrupt the boundary
        calculation (the previous text-decode approach broke for VLMs because
        re-encoding the decoded prefix does not reproduce visual token positions).
        """
        labels = torch.full_like(input_ids, -100)
        tokenizer = self.processor.tokenizer

        # Special boundary token IDs
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        # Encode "assistant\n" without adding BOS/EOS so we get the exact
        # tokens that appear right after <|im_start|> in assistant turns.
        assistant_header = tokenizer.encode("assistant\n", add_special_tokens=False)
        header_len = len(assistant_header)

        ids = input_ids.tolist()
        n = len(ids)

        i = 0
        while i < n:
            # Look for <|im_start|> followed by the "assistant\n" header
            if ids[i] == im_start_id and i + header_len < n:
                if ids[i + 1 : i + 1 + header_len] == assistant_header:
                    # Content starts right after <|im_start|>assistant\n
                    content_start = i + 1 + header_len
                    # Scan forward to find the matching <|im_end|>
                    content_end = content_start
                    while content_end < n and ids[content_end] != im_end_id:
                        content_end += 1
                    # Include <|im_end|> in labels so the model learns to stop
                    end_incl = min(content_end + 1, n)
                    labels[content_start:end_incl] = input_ids[content_start:end_incl]
                    i = end_incl
                    continue
            i += 1

        return labels


class DataCollatorForVLM:
    """
    Data collator that handles dynamic padding for VLM inputs.
    """

    def __init__(self, processor, max_seq_length: int = 4096):
        self.processor = processor
        self.max_seq_length = max_seq_length

    def __call__(self, batch: list) -> dict:
        # Separate different types of inputs
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        pixel_values_list = []
        image_grid_thw_list = []

        for sample in batch:
            seq_len = sample["input_ids"].shape[0]
            if seq_len > self.max_seq_length:
                # Truncate
                sample["input_ids"] = sample["input_ids"][:self.max_seq_length]
                sample["attention_mask"] = sample["attention_mask"][:self.max_seq_length]
                sample["labels"] = sample["labels"][:self.max_seq_length]

            input_ids_list.append(sample["input_ids"])
            attention_mask_list.append(sample["attention_mask"])
            labels_list.append(sample["labels"])

            if "pixel_values" in sample:
                pixel_values_list.append(sample["pixel_values"])
            if "image_grid_thw" in sample:
                image_grid_thw_list.append(sample["image_grid_thw"])

        # Pad sequences
        max_len = max(ids.shape[0] for ids in input_ids_list)
        pad_token_id = self.processor.tokenizer.pad_token_id or 0

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ids, mask, lbls in zip(input_ids_list, attention_mask_list, labels_list):
            pad_len = max_len - ids.shape[0]
            # Left padding for generation compatibility
            padded_input_ids.append(
                torch.cat([torch.full((pad_len,), pad_token_id, dtype=ids.dtype), ids])
            )
            padded_attention_mask.append(
                torch.cat([torch.zeros(pad_len, dtype=mask.dtype), mask])
            )
            padded_labels.append(
                torch.cat([torch.full((pad_len,), -100, dtype=lbls.dtype), lbls])
            )

        result = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }

        if pixel_values_list:
            result["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        if image_grid_thw_list:
            result["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)

        return result
