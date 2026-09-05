"""
Prepare JSONL data for MS-Swift fine-tuning of Qwen3-VL-4B (v2).

Key differences from prepare_data_swift.py (v1):
  - bbox_2d normalized to 0-1000 range (using PIL image dimensions)
  - Native multimodal format: user content is a list of
      [{"type":"text","text":...}, {"type":"image","image":path}]
    (no <image> placeholder; no custom template needed)
  - System message included
  - Assistant answer wrapped in ```json ... ``` code block
  - Stratified sampling: rare labels (slotted_hole, rectangular_hole)
    split independently so they're evenly distributed
  - channel / objects.ref fields included for future loss_scale use
  - Optional test split (--test_ratio)

Output files:
  data/train_v2.jsonl
  data/val_v2.jsonl
  data/test_v2.jsonl  (only if --test_ratio > 0)

Usage:
    python src/prepare_data_swift_v2.py
    python src/prepare_data_swift_v2.py --val_ratio 0.1 --test_ratio 0.1 --seed 42
"""

import json
import os
import random
import argparse
import logging
from collections import Counter
from pathlib import Path
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一个专业的工业图纸分析助手。你的任务是识别和分析工程图纸中的工件特征，"
    "包括但不限于：螺纹孔、圆角、矩形孔、圆孔、长圆孔、倒角、槽等特征。"
    "对于每种特征，请输出其类型、规格尺寸（如M8、R3）以及在图纸中的位置坐标。"
    "请根据图纸内容给出准确、完整的分析结果。"
)

USER_PROMPT = (
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

# ── Label mappings ─────────────────────────────────────────────────────────────

# Task names considered "rare" for stratified sampling
RARE_TASK_NAMES = {"slotted_hole_detection", "rectangular_hole_detection"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def normalize_bbox(bbox, img_w, img_h):
    """Convert pixel [x1,y1,x2,y2] to 0-1000 normalized integers."""
    x1, y1, x2, y2 = bbox
    nx1 = max(0, min(1000, int(round(x1 / img_w * 1000))))
    ny1 = max(0, min(1000, int(round(y1 / img_h * 1000))))
    nx2 = max(0, min(1000, int(round(x2 / img_w * 1000))))
    ny2 = max(0, min(1000, int(round(y2 / img_h * 1000))))
    return [nx1, ny1, nx2, ny2]


def convert_sample(raw: dict, image_dir: Path) -> dict | None:
    name = raw.get("dataitem_name", "")
    if not name:
        return None
    image_path = image_dir / name
    if not image_path.exists():
        return None

    non_empty = [t for t in raw.get("tasks", []) if t.get("task_values")]
    if not non_empty:
        return None

    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
    except Exception as e:
        logger.warning(f"Cannot open {image_path}: {e} — skipping normalization")
        img_w, img_h = 1, 1

    features = []

    for task in non_empty:
        for tv in task.get("task_values", []):
            value = tv.get("value", {})
            bbox = tv.get("bbox", [])
            if len(bbox) != 4:
                continue
            norm_bbox = normalize_bbox(bbox, img_w, img_h)
            label = value.get("label", "Unknown")
            features.append({
                "category": label,
                "size":     value.get("size", ""),
                "bbox_2d":  norm_bbox,
            })

    if not features:
        return None

    features_json = json.dumps(features, ensure_ascii=False, indent=2)
    answer = f"```json\n{features_json}\n```"

    # System message is passed via --system in the training script to avoid
    # PyArrow schema conflict (system=string vs user=array in the same column).
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text",  "text":  USER_PROMPT},
                    {"type": "image", "image": image_path.as_posix()},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ],
        # Kept for traceability
        "dataitem_name": name,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert drawing annotations to Swift JSONL v2 format (normalized bbox_2d)."
    )
    parser.add_argument("--input",      default="data/drawing_IM_03_PT_5K_R1.json")
    parser.add_argument("--image_dir",  default="data/IM_D03_PT_5K")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--val_ratio",  type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Fraction held out as test set (0 = no test split).")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    assert args.val_ratio + args.test_ratio < 1.0, "val_ratio + test_ratio must be < 1"

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

    # ── Convert ───────────────────────────────────────────────────────────────
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

    # ── Stratified split ──────────────────────────────────────────────────────
    # Build index → raw_data mapping to check task names for rare-label detection
    name_to_raw = {r.get("dataitem_name", ""): r for r in raw_data}

    rare_samples, common_samples = [], []
    for s in samples:
        raw = name_to_raw.get(s["dataitem_name"], {})
        task_names = {t.get("task_name", "") for t in raw.get("tasks", [])}
        if task_names & RARE_TASK_NAMES:
            rare_samples.append(s)
        else:
            common_samples.append(s)

    random.shuffle(rare_samples)
    random.shuffle(common_samples)

    logger.info(f"Rare samples (slotted/rect): {len(rare_samples)}")
    logger.info(f"Common samples:              {len(common_samples)}")

    def split_group(group):
        n = len(group)
        n_val  = max(1, int(n * args.val_ratio))  if args.val_ratio  > 0 else 0
        n_test = max(1, int(n * args.test_ratio)) if args.test_ratio > 0 else 0
        test  = group[:n_test]
        val   = group[n_test:n_test + n_val]
        train = group[n_test + n_val:]
        return train, val, test

    r_train, r_val, r_test = split_group(rare_samples)
    c_train, c_val, c_test = split_group(common_samples)

    train_samples = r_train + c_train
    val_samples   = r_val   + c_val
    test_samples  = r_test  + c_test

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    logger.info(f"Train: {len(train_samples):,}  (rare {len(r_train)}, common {len(c_train)})")
    logger.info(f"Val:   {len(val_samples):,}  (rare {len(r_val)},   common {len(c_val)})")
    if test_samples:
        logger.info(f"Test:  {len(test_samples):,}  (rare {len(r_test)},  common {len(c_test)})")

    # ── Write ─────────────────────────────────────────────────────────────────
    def write_jsonl(path: Path, data: list):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"  -> {path}")

    write_jsonl(output_dir / "train_v2.jsonl", train_samples)
    write_jsonl(output_dir / "val_v2.jsonl",   val_samples)
    if test_samples:
        write_jsonl(output_dir / "test_v2.jsonl", test_samples)

    # ── Label distribution ────────────────────────────────────────────────────
    label_dist: Counter = Counter()
    for s in samples:
        for msg in s["messages"]:
            if msg["role"] == "assistant":
                try:
                    text = msg["content"]
                    # strip ```json ... ```
                    lines = text.splitlines()
                    inner = "\n".join(lines[1:-1])
                    for det in json.loads(inner):
                        label_dist[det.get("category", "?")] += 1
                except Exception:
                    pass
    logger.info("\n=== Label distribution (all converted) ===")
    total = sum(label_dist.values())
    for label, cnt in label_dist.most_common():
        logger.info(f"  {label}: {cnt} ({cnt/total*100:.1f}%)")

    logger.info("\nNext: bash scripts/train-swift-wsl-v2.sh")


if __name__ == "__main__":
    main()
