"""
Prepare training data from the custom drawing annotation JSON.

Converts drawing_IM_03_PT_5K_R1.json → data/train.json + data/val.json
in conversation format compatible with DrawingFeatureDataset.

Data statistics:
  - 4,092 annotated images
  - Task types: threaded_hole, fillet, rectangular_hole, round_hole, slotted_hole
  - Label types: Threaded Hole, Threaded Hole Group, Fillet, Fillet Group,
                 Rectangular Hole, Round Hole, Round Hole Group, Slotted Hole

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


# ─── Label and task name translations (English → Chinese) ────────────────────

LABEL_ZH = {
    "Threaded Hole": "螺纹孔",
    "Threaded Hole Group": "螺纹孔组",
    "Fillet": "圆角",
    "Fillet Group": "圆角组",
    "Rectangular Hole": "矩形孔",
    "Rectangular Hole Group": "矩形孔组",
    "Round Hole": "圆孔",
    "Round Hole Group": "圆孔组",
    "Slotted Hole": "长圆孔",
    "Slotted Hole Group": "长圆孔组",
    "Chamfer": "倒角",
    "Chamfer Group": "倒角组",
    "Slot": "槽",
    "Slot Group": "槽组",
}

TASK_ZH = {
    "threaded_hole_detection": "螺纹孔",
    "fillet_detection": "圆角",
    "rectangular_hole_detection": "矩形孔",
    "round_hole_detection": "圆孔",
    "slotted_hole_detection": "长圆孔",
    "chamfer_detection": "倒角",
    "slot_detection": "槽",
}

# Varied prompts for data augmentation (same image, different phrasing)
PROMPTS = [
    "请识别这张图纸中的所有工件特征，包括类型、规格和位置坐标。",
    "分析这张工程图，列出所有检测到的工件特征，需包含每个特征的尺寸规格和边界框坐标。",
    "这张工业图纸中包含哪些工件特征？请逐一描述各特征的类型、规格和所在位置（边界框）。",
    "对这张图纸进行工件特征识别，输出所有特征的名称、规格尺寸和像素坐标范围。",
]


# ─── Answer generation ───────────────────────────────────────────────────────

def _clean_size(size: str) -> str:
    """Strip stray whitespace and fix encoding artifacts in size strings."""
    if not size:
        return ""
    return size.strip()


def build_answer(tasks: list) -> str:
    """Convert structured task annotations to a structured Chinese description."""
    sections = []
    total_features = 0

    for task in tasks:
        task_name = task.get("task_name", "")
        task_values = task.get("task_values", [])
        if not task_values:
            continue

        task_zh = TASK_ZH.get(task_name, task_name)
        lines = [f"【{task_zh}检测结果】"]

        for tv in task_values:
            value = tv.get("value", {})
            label = value.get("label", "Unknown")
            size = _clean_size(value.get("size", ""))
            bbox = tv.get("bbox", [])

            label_zh = LABEL_ZH.get(label, label)
            size_part = f"（{size}）" if size else ""
            bbox_part = f"，位置坐标：{bbox}" if bbox else ""
            lines.append(f"  - {label_zh}{size_part}{bbox_part}")
            total_features += 1

        sections.append("\n".join(lines))

    if not sections:
        return "该图纸中未检测到明显的工件特征。"

    header = f"这张图纸中共识别出 {total_features} 个工件特征：\n"
    return header + "\n\n".join(sections)


# ─── Sample conversion ────────────────────────────────────────────────────────

def convert_sample(raw: dict, image_dir: Path, prompt: str) -> dict | None:
    """Convert one raw JSON entry to conversation format. Returns None if invalid."""
    name = raw.get("dataitem_name", "")
    if not name:
        logger.debug("Entry missing 'dataitem_name', skipping.")
        return None

    if not (image_dir / name).exists():
        logger.debug(f"Image not found: {image_dir / name}")
        return None

    tasks = raw.get("tasks", [])
    if not tasks:
        logger.debug(f"No tasks for {name}, skipping.")
        return None

    # Filter out tasks with no annotations
    non_empty_tasks = [t for t in tasks if t.get("task_values")]
    if not non_empty_tasks:
        logger.debug(f"All tasks empty for {name}, skipping.")
        return None

    answer = build_answer(non_empty_tasks)
    return {
        "image": name,
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
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
        prompt = random.choice(PROMPTS)
        converted = convert_sample(raw, image_dir, prompt)
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

    logger.info(f"Train: {len(train_samples):,} samples  →  {train_path}")
    logger.info(f"Val:   {len(val_samples):,} samples  →  {val_path}")
    logger.info("\nAll done! Next step:")
    logger.info("  python src/train.py --config configs/train_config.yaml")


if __name__ == "__main__":
    main()
