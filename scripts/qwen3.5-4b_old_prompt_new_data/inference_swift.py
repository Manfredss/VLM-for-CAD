#!/usr/bin/env python3
"""
Inference script for Qwen3.5-4B with OLD prompt (10-feature detection).
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# OLD prompt
DEFAULT_SYSTEM_PROMPT = (
    "你是一个专业的工业图纸分析助手。你的任务是识别和分析工程图纸中的工件特征，"
    "包括但不限于：螺纹孔、圆角、矩形孔、圆孔、长圆孔、倒角、槽等特征。"
    "对于每种特征，请输出其类型、规格尺寸（如M8、R3）以及在图纸中的位置坐标。"
    "请根据图纸内容给出准确、完整的分析结果。"
)

DETECTION_PROMPT = (
    "你是一个专门用于分析工程图纸的AI视觉系统。\n\n"
    "任务：\n对提供的工程图纸图像进行目标检测。\n\n"
    "对每个检测到的对象，提取：\n"
    "- category（类别）\n"
    "- size（规格尺寸，如 M8、R3、ø10）\n"
    "- bbox_2d（归一化坐标边界框，0-1000 范围）\n\n"
    "需要提取的类别有：\n"
    "- Threaded Hole, Threaded Hole Group\n"
    "- Fillet, Fillet Group\n"
    "- Round Hole, Round Hole Group\n"
    "- Slotted Hole, Slotted Hole Group\n"
    "- Rectangular Hole, Rectangular Hole Group\n"
    "其中，Group 仅为描述性特征，Group 内包含的特征仍需单独列举\n\n"
    "边界框格式（整数，归一化到 0-1000）：\n[x_min, y_min, x_max, y_max]\n\n"
    "规则：\n"
    "- 检测所有可见的符号、标注、尺寸、表格、标题栏以及几何元素。\n"
    "- 使用归一化坐标（0-1000 范围）。\n"
    "- bbox_2d 必须由四个整数值组成，且满足 x_min < x_max 且 y_min < y_max。\n"
    "- 不要输出解释。\n"
    "- 仅输出严格 json。\n"
    "- 不要添加任何额外字段。\n"
    "- 若未检测到对象，返回空数组。\n"
    "- 输出必须严格遵守以下格式。\n\n"
    "输出示例（禁止修改）\n"
    "```json\n"
    "[\n"
    "  {\n"
    '    "category": "Round Hole",\n'
    '    "size": "ø10",\n'
    '    "bbox_2d": [x_min, y_min, x_max, y_max]\n'
    "  },\n"
    "  ...\n"
    "]\n"
    "```\n"
    "图纸："
)


def load_model(adapter_path: str, base_model: str = "Qwen/Qwen3.5-4B"):
    print(f"Loading base model: {base_model}")
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading LoRA adapter: {adapter_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("Adapter merged.")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model.eval()
    return model, processor


def run_inference_single(model, processor, image_path: str,
                         max_new_tokens: int = 4096) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": DETECTION_PROMPT},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
        )
    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    return processor.decode(generated_ids, skip_special_tokens=True)


def parse_output(text: str) -> list:
    text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    pattern = re.compile(r'```json\s*([\s\S]*?)```', re.MULTILINE)
    for m in pattern.finditer(text):
        try:
            parsed = json.loads(m.group(1).strip())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            continue
    match = re.search(r'\[\s*\{[\s\S]*?\}\s*\]', text)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return []


def calculate_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter) if (aa + ab - inter) > 0 else 0.0


def _deduplicate(items: list) -> list:
    if len(items) <= 1:
        return items
    unique = []
    for item in items:
        is_dup = False
        for existing in unique:
            if item.get("category") == existing.get("category"):
                bbox1 = item.get("bbox_2d", item.get("bbox", []))
                bbox2 = existing.get("bbox_2d", existing.get("bbox", []))
                if bbox1 and bbox2 and len(bbox1) == 4 and len(bbox2) == 4:
                    if calculate_iou(bbox1, bbox2) > 0.8:
                        is_dup = True
                        break
        if not is_dup:
            unique.append(item)
    return unique


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--base_model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    args = parser.parse_args()
    model, processor = load_model(args.adapter_path, args.base_model)
    result = run_inference_single(model, processor, args.image_path, args.max_new_tokens)
    features = parse_output(result)
    features = _deduplicate(features)
    print(json.dumps(features, ensure_ascii=False, indent=2))
