"""
Prepare training data from the custom drawing annotation JSON.

Converts drawing_IM_03_PT_5K_R1.json -> data/train.json + data/val.json
in the conversation format matching sample.json:

  {
    "dataitem_name": "image.png",
    "image": "data/IM_D03_PT_5K/image.png",
    "conversation": [
      {"from": "human", "value": "<fixed detection prompt with <image>>"},
      {"from": "qwen",  "value": [{"category": "...", "size": "...", "bbox": [...]}]}
    ]
  }

Usage:
    python src/prepare_data.py
    python src/prepare_data.py --val_ratio 0.1 --seed 42
    python src/prepare_data.py --input data/drawing_IM_03_PT_5K_R1.json \\
        --image_dir data/IM_D03_PT_5K --output_dir data
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


# ─── Fixed detection prompt (matches sample.json) ────────────────────────────

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
    "    ,\n"
    "       ...\n"
    "]\n\n"
    "图纸：\n<image>\n"
)


# ─── Detection list generation ────────────────────────────────────────────────

def _clean_size(size: str) -> str:
    """Strip stray whitespace from size strings."""
    if not size:
        return ""
    return size.strip()


def build_detections(tasks: list) -> list:
    """Convert structured task annotations to a list of detection dicts.

    Each detection: {"category": "<English label>", "size": "...", "bbox": [...]}
    """
    detections = []
    for task in tasks:
        for tv in task.get("task_values", []):
            value = tv.get("value", {})
            label = value.get("label", "Unknown")
            size = _clean_size(value.get("size", ""))
            bbox = tv.get("bbox", [])
            detections.append({"category": label, "size": size, "bbox": bbox})
    return detections


# ─── Sample conversion ────────────────────────────────────────────────────────

def convert_sample(raw: dict, image_dir: Path) -> dict | None:
    """Convert one raw JSON entry to conversation format. Returns None if invalid."""
    name = raw.get("dataitem_name", "")
    if not name:
        logger.debug("Entry missing 'dataitem_name', skipping.")
        return None

    if not (image_dir / name).exists():
        logger.debug(f"Image not found: {image_dir / name}")
        return None

    tasks = raw.get("tasks", [])
    non_empty_tasks = [t for t in tasks if t.get("task_values")]
    if not non_empty_tasks:
        logger.debug(f"All tasks empty for {name}, skipping.")
        return None

    detections = build_detections(non_empty_tasks)
    image_path = (image_dir / name).as_posix()

    return {
        "dataitem_name": name,
        "image": image_path,
        "conversation": [
            {"from": "human", "value": HUMAN_PROMPT},
            {"from": "qwen",  "value": detections},
        ],
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert drawing JSON annotations to conversation format for VLM training."
    )
    parser.add_argument(
        "--input",
        default="data/drawing_IM_03_PT_5K_R1.json",
        help="Path to raw annotation JSON (default: data/drawing_IM_03_PT_5K_R1.json)",
    )
    parser.add_argument(
        "--image_dir",
        default="data/IM_D03_PT_5K",
        help="Directory containing the PNG images (default: data/IM_D03_PT_5K)",
    )
    parser.add_argument(
        "--output_dir",
        default="data",
        help="Output directory for train.json / val.json (default: data)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Load raw data
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    logger.info(f"Loaded {len(raw_data)} raw entries from {input_path}")

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert all samples
    samples = []
    skipped = 0
    for raw in raw_data:
        converted = convert_sample(raw, image_dir)
        if converted:
            samples.append(converted)
        else:
            skipped += 1

    logger.info(f"Converted: {len(samples):,}  |  Skipped (missing image/tasks): {skipped}")

    if not samples:
        raise RuntimeError(
            "No samples were converted. Check that --image_dir points to the correct folder."
        )

    # Shuffle and split
    random.shuffle(samples)
    val_n = max(1, int(len(samples) * args.val_ratio))
    val_samples = samples[:val_n]
    train_samples = samples[val_n:]

    # Save
    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)

    logger.info(f"Train: {len(train_samples):,} samples  ->  {train_path}")
    logger.info(f"Val:   {len(val_samples):,} samples  ->  {val_path}")
    logger.info("\nAll done! Next step:")
    logger.info("  python src/train.py --config configs/train_config.yaml")


if __name__ == "__main__":
    main()
