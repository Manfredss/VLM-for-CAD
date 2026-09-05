"""
Prepare a rebalanced training dataset for continued fine-tuning from checkpoint-1400.

Strategy: Oversample images containing weak categories so the model sees them more often.
The val set stays the same (no oversampling).

Weak categories (F1 < 0.55 from checkpoint-1400 eval):
  - Chamfer (0.437), Chamfer Group (0.432)
  - Counterbore Hole (0.400), Counterbore Hole Group (0.471)
  - Fillet (0.545)
  - Revision Table (0.340)
  - Orthographic Projection - Right View (0.516)
  - Rectangular Hole (0.000)
  - Threaded Shaft (0.000)
  - Auxiliary View (0.000)
  - Section View (N/A - too rare)
  - Bill of Materials (N/A - too rare)

Oversampling ratios based on F1 gap:
  - F1 = 0: 5x oversampling
  - F1 < 0.4: 4x
  - F1 < 0.5: 3x
  - F1 < 0.6: 2x
"""

import json
import os
import random
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

DATA_DIR = Path(__file__).parent.parent
DEFAULT_JSON_PATH = DATA_DIR / "data" / "5k_15feats_with_view_v2.json"
IMAGE_DIR = DATA_DIR / "5k"
if not IMAGE_DIR.exists():
    IMAGE_DIR = DATA_DIR / "IM_D03_PT_5K"
OUTPUT_DIR = Path(__file__).parent / "dataset"
TRAIN_RATIO = 0.85
SEED = 42

# Oversampling config: category -> multiplier
# Based on checkpoint-1400 evaluation F1 scores
OVERSAMPLE_MAP = {
    # F1 = 0.000
    "Rectangular Hole": 5,
    "Threaded Shaft": 5,
    "Auxiliary View": 5,
    "Section View": 5,
    "Bill of Materials": 5,
    # F1 < 0.4
    "Revision Table": 4,
    # F1 < 0.5
    "Chamfer": 3,
    "Chamfer Group": 3,
    "Counterbore Hole": 3,
    "Counterbore Hole Group": 3,
    # F1 < 0.6
    "Fillet": 2,
    "Orthographic Projection - Right View": 2,
}

SYSTEM_PROMPT = (
    "你是一个专业的工业图纸分析助手。你的任务是识别和分析工程图纸中的所有结构特征和视图布局，"
    "包括各类孔、圆角、倒角、沉头孔、销孔、螺纹轴等结构特征，以及标题栏、注释、各方向视图等布局元素。"
    "对于每种特征，请输出其类型、规格尺寸以及在图纸中的位置坐标。"
    "请根据图纸内容给出准确、完整的分析结果。"
)

USER_PROMPT = (
    "你是一个专门用于分析工程图纸的AI视觉系统。\n\n"
    "任务：\n对提供的工程图纸图像进行目标检测。\n\n"
    "对每个检测到的对象，提取：\n"
    "- category（类别）\n"
    "- size（规格尺寸，如 M8、R3、ø10；视图类别无尺寸则留空）\n"
    "- bbox（归一化坐标边界框，0-1000 范围）\n\n"
    "需要提取的类别有：\n"
    "结构特征（14 类）：\n"
    "- Threaded Hole, Threaded Hole Group\n"
    "- Fillet, Fillet Group\n"
    "- Round Hole, Round Hole Group\n"
    "- Pin Hole, Pin Hole Group\n"
    "- Chamfer, Chamfer Group\n"
    "- Counterbore Hole, Counterbore Hole Group\n"
    "- Rectangular Hole\n"
    "- Threaded Shaft\n\n"
    "视图与布局（11 类）：\n"
    "- Title Block, Notes, Revision Table, Bill of Materials\n"
    "- Orthographic Projection - Front View\n"
    "- Orthographic Projection - Top View\n"
    "- Orthographic Projection - Left View\n"
    "- Orthographic Projection - Right View\n"
    "- Isometric View, Auxiliary View, Section View\n\n"
    "其中，Group 仅为描述性特征，Group 内包含的特征仍需单独列举\n\n"
    "边界框格式（整数，归一化到 0-1000）：\n[x_min, y_min, x_max, y_max]\n\n"
    "规则：\n"
    "- 检测所有可见的符号、标注、尺寸、表格、标题栏以及几何元素。\n"
    "- 使用归一化坐标（0-1000 范围）。\n"
    "- bbox 必须由四个整数值组成，且满足 x_min < x_max 且 y_min < y_max。\n"
    "- 不要输出解释。\n"
    "- 仅输出严格 json。\n"
    "- 不要添加任何额外字段。\n"
    "- 若未检测到对象，返回空数组。\n"
    "- 输出必须是纯 JSON 数组，不要包裹 markdown 代码块。\n\n"
    "输出示例（禁止修改）\n"
    "[\n"
    "  {\n"
    '    "category": "Round Hole",\n'
    '    "size": "ø10",\n'
    '    "bbox": [x_min, y_min, x_max, y_max]\n'
    "  },\n"
    "  ...\n"
    "]\n"
    "图纸："
)

LABEL_TO_CHANNEL = {
    "Threaded Hole": "holes", "Threaded Hole Group": "holes",
    "Round Hole": "holes", "Round Hole Group": "holes",
    "Pin Hole": "holes", "Pin Hole Group": "holes",
    "Counterbore Hole": "holes", "Counterbore Hole Group": "holes",
    "Rectangular Hole": "holes",
    "Threaded Shaft": "holes",
    "Fillet": "yuanjiao", "Fillet Group": "yuanjiao",
    "Chamfer": "chamfer", "Chamfer Group": "chamfer",
    "Title Block": "view", "Notes": "view",
    "Revision Table": "view", "Bill of Materials": "view",
    "Orthographic Projection - Front View": "view",
    "Orthographic Projection - Top View": "view",
    "Orthographic Projection - Left View": "view",
    "Orthographic Projection - Right View": "view",
    "Isometric View": "view", "Auxiliary View": "view",
    "Section View": "view",
}

RARE_LABELS = {
    "Threaded Shaft", "Section View", "Auxiliary View",
    "Revision Table", "Bill of Materials",
    "Counterbore Hole", "Counterbore Hole Group",
    "Chamfer", "Chamfer Group",
    "Rectangular Hole",
}


def normalize_bbox(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    nx1 = max(0, min(1000, int(round(x1 / img_w * 1000))))
    ny1 = max(0, min(1000, int(round(y1 / img_h * 1000))))
    nx2 = max(0, min(1000, int(round(x2 / img_w * 1000))))
    ny2 = max(0, min(1000, int(round(y2 / img_h * 1000))))
    return [nx1, ny1, nx2, ny2]


def convert_sample(item, image_dir):
    image_name = item["dataitem_name"]
    image_path = str(image_dir / image_name)

    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
    except Exception as e:
        print(f"WARNING: Cannot open {image_path}: {e}")
        img_w, img_h = 1, 1

    features = []
    channels = set()
    for task in item.get("tasks", []):
        for tv in task.get("task_values", []):
            val = tv["value"]
            bbox = tv["bbox"]
            norm_bbox = normalize_bbox(bbox, img_w, img_h)
            label = val["label"]
            features.append({
                "category": label,
                "size": val.get("size", ""),
                "bbox": norm_bbox,
            })
            channels.add(LABEL_TO_CHANNEL.get(label, "default"))

    if "holes" in channels:
        channel = "holes"
    elif "yuanjiao" in channels:
        channel = "yuanjiao"
    elif "chamfer" in channels:
        channel = "chamfer"
    else:
        channel = "default"

    answer = json.dumps(features, ensure_ascii=False, indent=2)

    conversation = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>{USER_PROMPT}"},
            {"role": "assistant", "content": answer},
        ],
        "images": [image_path],
        "channel": channel,
    }
    return conversation


def get_labels_in_item(item):
    labels = set()
    for t in item.get("tasks", []):
        for tv in t.get("task_values", []):
            labels.add(tv["value"]["label"])
    return labels


def compute_oversample_count(labels):
    """Return how many times this sample should appear (1 = no oversampling)."""
    max_mult = 1
    for label in labels:
        mult = OVERSAMPLE_MAP.get(label, 1)
        max_mult = max(max_mult, mult)
    return max_mult


def main():
    parser = argparse.ArgumentParser(description="Generate rebalanced ms-swift JSONL for continued fine-tuning")
    parser.add_argument("--json_path", type=str, default=str(DEFAULT_JSON_PATH))
    parser.add_argument("--image_dir", type=str, default=str(IMAGE_DIR))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    json_path = Path(args.json_path)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    random.seed(args.seed)

    with open(json_path, "r") as f:
        data = json.load(f)

    available_images = set(f for f in os.listdir(image_dir) if f.endswith(".png"))
    valid_data = [item for item in data if item["dataitem_name"] in available_images]
    print(f"Total annotated samples: {len(data)}")
    print(f"Samples with local images: {len(valid_data)}")

    # Stratified split (same as original to keep val set identical)
    rare_indices = []
    common_indices = []
    for i, item in enumerate(valid_data):
        labels = get_labels_in_item(item)
        if labels & RARE_LABELS:
            rare_indices.append(i)
        else:
            common_indices.append(i)

    random.shuffle(rare_indices)
    random.shuffle(common_indices)

    rare_split = int(len(rare_indices) * args.train_ratio)
    common_split = int(len(common_indices) * args.train_ratio)

    train_indices = rare_indices[:rare_split] + common_indices[:common_split]
    val_indices = rare_indices[rare_split:] + common_indices[common_split:]

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    # Build oversampled training set
    oversampled_train = []
    oversample_stats = Counter()
    for idx in train_indices:
        item = valid_data[idx]
        labels = get_labels_in_item(item)
        count = compute_oversample_count(labels)
        for _ in range(count):
            oversampled_train.append(idx)
        if count > 1:
            oversample_stats[count] += 1

    random.shuffle(oversampled_train)

    print(f"\nOriginal train: {len(train_indices)}")
    print(f"Oversampled train: {len(oversampled_train)} ({len(oversampled_train)/len(train_indices):.2f}x)")
    print(f"Val: {len(val_indices)} (unchanged)")
    for mult, cnt in sorted(oversample_stats.items()):
        print(f"  {mult}x oversampled: {cnt} unique images")

    # Category distribution in oversampled training set
    label_dist_orig = Counter()
    label_dist_over = Counter()
    for idx in train_indices:
        for t in valid_data[idx].get("tasks", []):
            for tv in t.get("task_values", []):
                label_dist_orig[tv["value"]["label"]] += 1

    for idx in oversampled_train:
        for t in valid_data[idx].get("tasks", []):
            for tv in t.get("task_values", []):
                label_dist_over[tv["value"]["label"]] += 1

    print("\n=== Category distribution (original → oversampled) ===")
    for label in sorted(label_dist_over.keys(), key=lambda x: -label_dist_over[x]):
        orig = label_dist_orig.get(label, 0)
        over = label_dist_over[label]
        ratio = over / orig if orig > 0 else 0
        marker = " ★" if OVERSAMPLE_MAP.get(label, 1) > 1 else ""
        print(f"  {label}: {orig} → {over} ({ratio:.1f}x){marker}")

    # Write datasets
    os.makedirs(output_dir, exist_ok=True)

    train_name = "train_15feats_view_rebalanced.jsonl"
    val_name = "val_15feats_view.jsonl"

    train_count = 0
    with open(output_dir / train_name, "w", encoding="utf-8") as f:
        for idx in oversampled_train:
            sample = convert_sample(valid_data[idx], image_dir)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            train_count += 1

    val_count = 0
    with open(output_dir / val_name, "w", encoding="utf-8") as f:
        for idx in val_indices:
            sample = convert_sample(valid_data[idx], image_dir)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            val_count += 1

    print(f"\nSaved to {output_dir}")
    print(f"  {train_name}: {train_count} samples")
    print(f"  {val_name}:   {val_count} samples")


if __name__ == "__main__":
    main()
