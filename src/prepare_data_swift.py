"""
Prepare JSONL data for MS-Swift fine-tuning of Qwen3-VL-4B.

Converts drawing_IM_03_PT_5K_R1.json -> data/train.jsonl + data/val.jsonl
in Swift conversation JSONL format:

  {
    "dataitem_name": "image.png",
    "images": ["data/IM_D03_PT_5K/image.png"],
    "messages": [
      {"role": "user",      "content": "<fixed detection prompt ending with <image>>"},
      {"role": "assistant", "content": "[{\"category\": \"...\", \"size\": \"...\", \"bbox\": [...]}]"}
    ]
  }

Notes:
  - "images" must be a list (Swift requirement).
  - Assistant content must be a JSON string, not a Python list.
  - <image> in the user content is the Swift image placeholder.

Usage:
    python src/prepare_data_swift.py
    python src/prepare_data_swift.py --val_ratio 0.1 --seed 42
"""

import json
import random
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Prompt (must stay in sync with prepare_data.py / inference.py) ────────────

HUMAN_PROMPT = (
    "你是一个专门用于分析工程图纸的AI视觉系统。\n\n"
    "任务：\n对提供的工程图纸图像进行目标检测。\n\n"
    "对于每个检测到的对象，提取：\n"
    "- category（类别）\n"
    "- bbox（图像坐标边界框）\n\n"
    "需要提取的类别有：\n"
    "- Threaded Hole, Threaded Hole Group\n"
    "- Fillet, Fillet Group\n"
    "- Round Hole, Round Hole Group\n"
    "- Slotted Hole, Slotted Hole Group\n"
    "- Rectangular Hole, Rectangular Hole Group\n"
    "其中，Group 仅为描述性特征，Group 内包含的特征仍需单独列举\n\n"
    "边界框格式（整数）：\n[x_min, y_min, x_max, y_max]\n\n"
    "规则：\n"
    "- 检测所有可见的符号、标注、尺寸、表格、标题栏以及几何元素。\n"
    "- 使用像素坐标。\n"
    "- bbox 必须由四个整数值组成，且满足 x_min < x_max 且 y_min < y_max。\n"
    "- 不要输出解释。\n"
    "- 不要输出 Markdown 格式。\n"
    "- 仅输出严格 json。\n"
    "- 不要添加任何额外字段。\n"
    "- 若未检测到对象，返回空数组。\n"
    "- 输出必须严格遵守以下格式。\n\n"
    "输出示例（禁止修改）\n"
    "[\n"
    "    {\n"
    '        "category": "Round Hole",\n'
    '        "bbox": [x_min, y_min, x_max, y_max]\n'
    "    },\n"
    "    ...\n"
    "]\n\n"
    "图纸：\n<image>\n"
)


# ── Detection builder ─────────────────────────────────────────────────────────

def _clean_size(size: str) -> str:
    return size.strip() if size else ""


def build_detections(tasks: list) -> list:
    detections = []
    for task in tasks:
        for tv in task.get("task_values", []):
            value = tv.get("value", {})
            detections.append({
                "category": value.get("label", "Unknown"),
                "size":     _clean_size(value.get("size", "")),
                "bbox":     tv.get("bbox", []),
            })
    return detections


# ── Sample conversion ─────────────────────────────────────────────────────────

def convert_sample(raw: dict, image_dir: Path) -> dict | None:
    name = raw.get("dataitem_name", "")
    if not name:
        return None
    if not (image_dir / name).exists():
        return None

    non_empty = [t for t in raw.get("tasks", []) if t.get("task_values")]
    if not non_empty:
        return None

    detections = build_detections(non_empty)
    image_path = (image_dir / name).as_posix()

    return {
        "dataitem_name": name,
        # Swift requires "images" as a list
        "images": [image_path],
        "messages": [
            {"role": "user",      "content": HUMAN_PROMPT},
            # Assistant content must be a JSON string, not a Python list
            {"role": "assistant", "content": json.dumps(detections, ensure_ascii=False)},
        ],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert drawing annotations to Swift JSONL format for Qwen3-VL-4B."
    )
    parser.add_argument("--input",      default="data/drawing_IM_03_PT_5K_R1.json")
    parser.add_argument("--image_dir",  default="data/IM_D03_PT_5K")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--val_ratio",  type=float, default=0.1)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    logger.info(f"Loaded {len(raw_data)} raw entries")

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, skipped = [], 0
    for raw in raw_data:
        s = convert_sample(raw, image_dir)
        if s:
            samples.append(s)
        else:
            skipped += 1

    logger.info(f"Converted: {len(samples):,}  |  Skipped: {skipped}")
    if not samples:
        raise RuntimeError("No samples converted. Check --image_dir.")

    random.shuffle(samples)
    val_n = max(1, int(len(samples) * args.val_ratio))
    val_samples   = samples[:val_n]
    train_samples = samples[val_n:]

    def write_jsonl(path: Path, data: list):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    train_path = output_dir / "train.jsonl"
    val_path   = output_dir / "val.jsonl"
    write_jsonl(train_path, train_samples)
    write_jsonl(val_path,   val_samples)

    logger.info(f"Train: {len(train_samples):,}  ->  {train_path}")
    logger.info(f"Val:   {len(val_samples):,}  ->  {val_path}")
    logger.info("\nNext: bash scripts/train-swift-wsl.sh")


if __name__ == "__main__":
    main()
