"""
Step 1 (Swift 版): 将原始 JSON 标注转换为 ms-swift 训练格式
输出 train.jsonl / val.jsonl（每行一个 JSON），供 swift sft 使用

适配 Qwen3.5-27B（与 30B 版本数据格式完全相同，可共用数据）

与 prepare_dataset.py 的区别：
    - 输出 JSONL 格式（非 JSON 数组）
        - bbox 统一输出为 bbox
        - 助手回复改为纯 JSON（不再包裹 markdown 代码块）
        - user content 统一输出为字符串，图片路径放到顶层 images 字段
  - 内置坐标归一化（0-1000 范围）
    - 保留 channel 字段
  - 保留分层采样逻辑
"""

import json
import os
import random
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

# ============ 配置 ============
DATA_DIR = Path(__file__).parent.parent  # 上级目录（OneDrive_1_2-19-2026/）
DEFAULT_JSON_PATH = DATA_DIR / "augmented_dataset.json"
if not DEFAULT_JSON_PATH.exists():
    DEFAULT_JSON_PATH = DATA_DIR / "drawing_IM_03_PT_5K_R1.json"
IMAGE_DIR = DATA_DIR / "5k"
if not IMAGE_DIR.exists():
    IMAGE_DIR = DATA_DIR / "IM_D03_PT_5K"
OUTPUT_DIR = DATA_DIR / "qwen_vl_dataset_swift"
TRAIN_RATIO = 0.85   # 85% train, 15% val
SEED = 42

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
    "- bbox（归一化坐标边界框，0-1000 范围）\n\n"
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

# 类别 → channel 映射（供 loss_scale.py 用）
LABEL_TO_CHANNEL = {
    "Threaded Hole": "holes",
    "Threaded Hole Group": "holes",
    "Round Hole": "holes",
    "Round Hole Group": "holes",
    "Slotted Hole": "holes",
    "Slotted Hole Group": "holes",
    "Rectangular Hole": "holes",
    "Rectangular Hole Group": "holes",
    "Fillet": "yuanjiao",
    "Fillet Group": "yuanjiao",
}

# 类别 → 中文名（供 loss_scale.py 的 objects.ref 用）
LABEL_TO_CN = {
    "Threaded Hole": "螺纹孔",
    "Threaded Hole Group": "螺纹孔组",
    "Round Hole": "圆孔",
    "Round Hole Group": "圆孔组",
    "Slotted Hole": "长圆孔",
    "Slotted Hole Group": "长圆孔组",
    "Rectangular Hole": "矩形孔",
    "Rectangular Hole Group": "矩形孔组",
    "Fillet": "圆角",
    "Fillet Group": "圆角组",
}


def normalize_bbox(bbox, img_w, img_h):
    """将像素坐标归一化到 0-1000 范围"""
    x1, y1, x2, y2 = bbox
    nx1 = int(round(x1 / img_w * 1000))
    ny1 = int(round(y1 / img_h * 1000))
    nx2 = int(round(x2 / img_w * 1000))
    ny2 = int(round(y2 / img_h * 1000))
    # 钳位到 [0, 1000]
    nx1 = max(0, min(1000, nx1))
    ny1 = max(0, min(1000, ny1))
    nx2 = max(0, min(1000, nx2))
    ny2 = max(0, min(1000, ny2))
    return [nx1, ny1, nx2, ny2]


def convert_sample(item, image_dir):
    """将一条原始标注转为 swift 训练格式（JSONL 单行）"""
    image_name = item["dataitem_name"]
    image_path = str(image_dir / image_name)

    # 读取图片尺寸用于归一化
    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
    except Exception as e:
        print(f"WARNING: Cannot open {image_path}: {e}, skipping normalization")
        img_w, img_h = 1, 1  # fallback，不归一化

    # 构建 ground truth
    features = []
    channels = set()
    for task in item.get("tasks", []):
        for tv in task.get("task_values", []):
            val = tv["value"]
            bbox = tv["bbox"]  # [x1, y1, x2, y2] 像素
            norm_bbox = normalize_bbox(bbox, img_w, img_h)
            label = val["label"]
            features.append({
                "category": label,
                "size": val.get("size", ""),
                "bbox": norm_bbox,
            })
            channels.add(LABEL_TO_CHANNEL.get(label, "default"))

    # 确定 channel（如果有多种，优先 holes）
    if "holes" in channels:
        channel = "holes"
    elif "yuanjiao" in channels:
        channel = "yuanjiao"
    else:
        channel = "default"

    # 助手回复：纯 JSON 数组，避免训练和推理输出格式漂移
    answer = json.dumps(features, ensure_ascii=False, indent=2)

    # ms-swift 4.x 原生多模态格式：messages.content 全部为字符串，图片放到顶层 images
    conversation = {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"<image>{USER_PROMPT}",
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ],
        "images": [image_path],
        "channel": channel,
    }
    return conversation


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ms-swift JSONL datasets for Qwen finetuning")
    parser.add_argument("--json_path", type=str, default=str(DEFAULT_JSON_PATH), help="Input annotation json")
    parser.add_argument("--image_dir", type=str, default=str(IMAGE_DIR), help="Directory containing png images")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Output directory for generated jsonl")
    parser.add_argument("--train_name", type=str, default="", help="Output train jsonl file name")
    parser.add_argument("--val_name", type=str, default="", help="Output val jsonl file name")
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    json_path = Path(args.json_path)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    if not args.train_name or not args.val_name:
        stem = json_path.stem.lower()
        if "augmented" in stem:
            train_name = "train_augmented.jsonl"
            val_name = "val_augmented.jsonl"
        else:
            train_name = "train.jsonl"
            val_name = "val.jsonl"
    else:
        train_name = args.train_name
        val_name = args.val_name

    random.seed(args.seed)

    with open(json_path, "r") as f:
        data = json.load(f)

    # 只保留本地有图片的样本
    available_images = set(f for f in os.listdir(image_dir) if f.endswith(".png"))
    valid_data = [item for item in data if item["dataitem_name"] in available_images]
    print(f"Total annotated samples: {len(data)}")
    print(f"Samples with local images: {len(valid_data)}")

    # 分层采样：确保 rare labels (Slotted Hole, Rectangular Hole) 均匀分布
    rare_indices = []
    common_indices = []
    for i, item in enumerate(valid_data):
        task_names = set(t["task_name"] for t in item.get("tasks", []))
        if "slotted_hole_detection" in task_names or "rectangular_hole_detection" in task_names:
            rare_indices.append(i)
        else:
            common_indices.append(i)

    random.shuffle(rare_indices)
    random.shuffle(common_indices)

    # 分层划分
    rare_split = int(len(rare_indices) * args.train_ratio)
    common_split = int(len(common_indices) * args.train_ratio)

    train_indices = rare_indices[:rare_split] + common_indices[:common_split]
    val_indices = rare_indices[rare_split:] + common_indices[common_split:]

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    print(f"\nTrain: {len(train_indices)} (rare: {rare_split}, common: {common_split})")
    print(f"Val:   {len(val_indices)} (rare: {len(rare_indices)-rare_split}, common: {len(common_indices)-common_split})")

    # 转换
    os.makedirs(output_dir, exist_ok=True)

    # 输出 JSONL 格式
    train_count = 0
    with open(output_dir / train_name, "w", encoding="utf-8") as f:
        for i in train_indices:
            sample = convert_sample(valid_data[i], image_dir)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            train_count += 1

    val_count = 0
    with open(output_dir / val_name, "w", encoding="utf-8") as f:
        for i in val_indices:
            sample = convert_sample(valid_data[i], image_dir)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            val_count += 1

    print(f"\nSaved to {output_dir}")
    print(f"  {train_name}: {train_count} samples")
    print(f"  {val_name}:   {val_count} samples")

    # 统计
    label_dist = Counter()
    for item in valid_data:
        for task in item.get("tasks", []):
            for tv in task.get("task_values", []):
                label_dist[tv["value"]["label"]] += 1
    print("\n=== Label distribution (all) ===")
    for label, cnt in label_dist.most_common():
        pct = cnt / sum(label_dist.values()) * 100
        print(f"  {label}: {cnt} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
