#!/usr/bin/env python3
"""Prepare training data with OLD prompt from 5k_10feats_v2_augmented.json.
80:20 train:val split, balanced benchmark subset."""

import json
import random
from collections import defaultdict
from pathlib import Path
from PIL import Image

random.seed(42)

JSON_PATH = "/workspace/data/5k_10feats_v2_augmented.json"
IMAGE_DIR = Path("/workspace/data/5k")
OUTPUT_DIR = Path("/workspace/data")
TRAIN_RATIO = 0.80

# OLD prompt
SYSTEM_PROMPT = (
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

print("Loading augmented data...")
with open(JSON_PATH) as f:
    raw_data = json.load(f)
print(f"Loaded {len(raw_data)} items")

# Convert to JSONL records
records = []
skipped = 0
for item in raw_data:
    img_name = item["dataitem_name"]
    img_path = IMAGE_DIR / img_name
    if not img_path.exists():
        skipped += 1
        continue

    with Image.open(img_path) as img:
        w, h = img.size

    features = []
    for task in item.get("tasks", []):
        for tv in task.get("task_values", []):
            bbox = tv.get("bbox", [])
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            # Normalize to 0-1000
            nx1 = int(round(x1 / w * 1000))
            ny1 = int(round(y1 / h * 1000))
            nx2 = int(round(x2 / w * 1000))
            ny2 = int(round(y2 / h * 1000))
            nx1, ny1 = max(0, min(1000, nx1)), max(0, min(1000, ny1))
            nx2, ny2 = max(0, min(1000, nx2)), max(0, min(1000, ny2))
            if nx1 >= nx2 or ny1 >= ny2:
                continue
            features.append({
                "category": tv["value"]["label"],
                "size": tv["value"].get("size", ""),
                "bbox_2d": [nx1, ny1, nx2, ny2],
            })

    if not features:
        skipped += 1
        continue

    assistant_content = json.dumps(features, ensure_ascii=False, indent=2)
    record = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<image>" + DETECTION_PROMPT},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": [str(img_path)],
        "channel": "10feats_oldprompt",
    }

    # Determine dominant category for stratified split
    cat_counts = defaultdict(int)
    for feat in features:
        cat_counts[feat["category"]] += 1
    dominant = max(cat_counts, key=cat_counts.get)

    records.append({"record": record, "dominant": dominant})

print(f"Valid records: {len(records)}, skipped: {skipped}")

# Stratified 80:20 split
cat_indices = defaultdict(list)
for i, rec in enumerate(records):
    cat_indices[rec["dominant"]].append(i)

train_indices = set()
val_indices = set()
for cat, indices in cat_indices.items():
    random.shuffle(indices)
    n_train = max(1, int(len(indices) * TRAIN_RATIO))
    train_indices.update(indices[:n_train])
    val_indices.update(indices[n_train:])

print(f"\nTrain: {len(train_indices)}, Val: {len(val_indices)}")

# Write train JSONL
train_file = OUTPUT_DIR / "train_10feats_oldprompt.jsonl"
with open(train_file, "w", encoding="utf-8") as f:
    for i in sorted(train_indices):
        f.write(json.dumps(records[i]["record"], ensure_ascii=False) + "\n")
print(f"Wrote {train_file}")

# Write val JSONL
val_file = OUTPUT_DIR / "val_10feats_oldprompt.jsonl"
with open(val_file, "w", encoding="utf-8") as f:
    for i in sorted(val_indices):
        f.write(json.dumps(records[i]["record"], ensure_ascii=False) + "\n")
print(f"Wrote {val_file}")

# Create balanced benchmark (48 samples from val)
val_records = [(i, records[i]) for i in sorted(val_indices)]
val_cat_indices = defaultdict(list)
for idx, (orig_i, rec) in enumerate(val_records):
    val_cat_indices[rec["dominant"]].append(idx)

TARGET_BENCH = 48
bench_selected = set()
for cat, indices in val_cat_indices.items():
    n = max(1, round(len(indices) / len(val_records) * TARGET_BENCH))
    chosen = random.sample(indices, min(n, len(indices)))
    bench_selected.update(chosen)

remaining = TARGET_BENCH - len(bench_selected)
if remaining > 0:
    pool = [i for i in range(len(val_records)) if i not in bench_selected]
    bench_selected.update(random.sample(pool, min(remaining, len(pool))))
if len(bench_selected) > TARGET_BENCH:
    bench_selected = set(random.sample(list(bench_selected), TARGET_BENCH))

bench_file = OUTPUT_DIR / "benchmark_10feats_oldprompt.jsonl"
with open(bench_file, "w", encoding="utf-8") as f:
    for idx in sorted(bench_selected):
        orig_i, rec = val_records[idx]
        f.write(json.dumps(rec["record"], ensure_ascii=False) + "\n")
print(f"Wrote {len(bench_selected)} benchmark samples to {bench_file}")

# Print distribution
print("\nCategory distribution:")
for split_name, idx_set in [("Train", train_indices), ("Val", val_indices)]:
    cats = defaultdict(int)
    for i in idx_set:
        cats[records[i]["dominant"]] += 1
    print(f"  {split_name}:")
    for cat, count in sorted(cats.items()):
        print(f"    {cat}: {count}")
