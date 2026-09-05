#!/usr/bin/env python3
"""
Inference script for Qwen3.5-4B with NEW prompt (10-feature detection).
Supports both standalone inference and checkpoint-based eval.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# NEW prompt (structured 10-feature)
DEFAULT_SYSTEM_PROMPT = """任务：
在输入的工程图纸中定位、分类并提取以下结构特征的实例及其组，并提取对应尺寸参数，输出 JSON 列表。

需要识别两大类结构：
1. 孔类结构
2. 圆角结构

========================
1. 类别定义
========================

1.1 孔类结构（8 类）
  - 圆孔 (Round Hole)
  - 腰孔 (Slotted Hole)
  - 矩形孔 (Rectangular Hole)
  - 螺纹孔 (Threaded Hole)
  - 圆孔组 (Round Hole Group)
  - 腰孔组 (Slotted Hole Group)
  - 矩形孔组 (Rectangular Hole Group)
  - 螺纹孔组 (Threaded Hole Group)

1.2 圆角结构（2 类）
  - 圆角 (Fillet)
  - 圆角组 (Fillet Group)


========================
2. 几何定义与尺寸参数提取规则
========================

--------------------------------
2.1 圆孔
--------------------------------
- 轮廓：闭合圆
- 尺寸参数 D：直径
  例：18 → "18"

--------------------------------
2.2 腰孔
--------------------------------
- 轮廓：形似椭圆，由两平行直边 + 两对称半圆弧构成
- 尺寸参数 W×L：
  - W：直边间距
  - L：两圆弧中心距
  例：14×30 → "14*30"

--------------------------------
2.3 矩形孔
--------------------------------
- 轮廓：四边形（含正方形）

尺寸参数：
- 正方形：A → "□A"
  例：□18 → "□18"

- 长方形：L×W → "L*W"
  例：20×14 → "20*14"

--------------------------------
2.4 螺纹孔
--------------------------------
- 轮廓：闭合圆
- 特征：尺寸参数包含字母 M 或存在螺纹线
- 尺寸参数 M
  例：M8 → "M8"

--------------------------------
2.5 圆孔组
--------------------------------
- 定义：多个相同尺寸圆孔组成并统一标注
- 标注形式：N x D 或 N - D

示例：
4x18 → "4x18"
4-18 → "4-18"

bbox 应覆盖该组内所有对应圆孔实例

--------------------------------
2.6 腰孔组
--------------------------------
标注形式：
N x W × L 或 N - W × L

示例：
2x14×30 → "2x14*30"
2x14×30 → "2-14*30"

bbox 应覆盖该组内所有腰孔实例

--------------------------------
2.7 矩形孔组
--------------------------------
标注形式：
N x L × W 或 N - L × W

示例：
2x□18 → "2x□18mm"
2x18x18 → "2x18x18"
3-20×14 → "3-20*14mm"

bbox 应覆盖该组内所有矩形孔实例

--------------------------------
2.8 螺纹孔组
--------------------------------
标注形式：
N x M 或 N - M

示例：
2xM8 → "2xM8"
2-M8 → "2-M8"

bbox 应覆盖该组内所有螺纹孔实例


--------------------------------
2.9 圆角
--------------------------------
几何定义：
一段圆弧，用于将两条相交直线光滑连接。

可能存在：
- 1/4 圆角
- 1/2 圆角（半圆）

规则：
半圆圆角必须整体提取，不得拆分为两个 1/4 圆角。

尺寸参数：
- 半径：R3 → "R3"
- 直径形式：Ø6 → "Ø6"

--------------------------------
2.10 圆角组
--------------------------------
定义：
多个相同尺寸圆角组成并统一标注

标注形式：
N x R 或 N - R

示例：
2xR3 → "2xR3"
4xØ6 → "4xØ6"
2-R3 → "2-R3"

bbox 应覆盖该组内所有圆角实例


========================
3. 输出格式
========================

严格返回 JSON 列表：

[
  {
    "category": "<类别>",
    "bbox": [x_min, y_min, x_max, y_max],
    "size": "<尺寸参数字符串>"
  }
]


允许的 category：

Round Hole
Slotted Hole
Rectangular Hole
Threaded Hole
Round Hole Group
Slotted Hole Group
Rectangular Hole Group
Threaded Hole Group
Fillet
Fillet Group


bbox：
- 归一化坐标（0-1000 范围）
- 整数
- [x_min, y_min, x_max, y_max]


========================
4. 补充规则
========================

1. 所有单实例必须可见且闭合。

2. "组"是独立检测对象：
   当多个相同尺寸特征由统一标注描述时：
   - 必须检测所有单实例
   - 还必须额外检测该组

3. 尺寸提取优先级：
   优先从特征附近标注提取。

4. 若单实例附近没有尺寸标注：
   - 必须找到其所属组
   - 继承组内单个特征尺寸

   注意：
   - 单实例 size 不包含数量
   - 组 size 必须包含数量

5. 组识别条件：
   - 类型一致
   - 尺寸一致
   - 与组标注一致

6. 螺纹孔识别优先规则：
   若尺寸包含 M，则归类为螺纹孔，而非圆孔。

7. 同一特征若属于某组：
   - 单实例仍必须输出
   - 组也必须输出

8. 同一层级不得重复框。

9. 必须检测：
   - 所有孔实例
   - 所有孔组
   - 所有圆角实例
   - 所有圆角组

10. 无置信度阈值要求。

11. 图像分辨率为当前像素分辨率，
    bbox 必须在归一化坐标系（0-1000）下返回。


========================
输出要求
========================

仅返回 JSON 结果。
不得附加任何解释、说明或额外文本。"""

DETECTION_PROMPT = "请分析这张工程图纸，识别所有结构特征。"


def load_model(adapter_path: str, base_model: str = "Qwen/Qwen3.5-4B"):
    """Load model with optional LoRA adapter."""
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
    """Run inference on a single image."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": DETECTION_PROMPT},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    image = Image.open(image_path).convert("RGB")

    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    result = processor.decode(generated_ids, skip_special_tokens=True)
    return result


def parse_output(text: str) -> list:
    """Parse model output to list of feature dicts."""
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


def _deduplicate(items: list) -> list:
    """Remove duplicate detections (same category + overlapping bbox)."""
    if len(items) <= 1:
        return items

    unique = []
    for item in items:
        is_dup = False
        for existing in unique:
            if item.get("category") == existing.get("category"):
                bbox1 = item.get("bbox", item.get("bbox_2d", []))
                bbox2 = existing.get("bbox", existing.get("bbox_2d", []))
                if bbox1 and bbox2 and len(bbox1) == 4 and len(bbox2) == 4:
                    from metric import calculate_iou
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
