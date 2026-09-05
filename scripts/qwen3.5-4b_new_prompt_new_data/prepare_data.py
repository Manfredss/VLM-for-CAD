#!/usr/bin/env python3
"""Prepare training data with NEW prompt from 5k_10feats_v2_augmented.json.
80:20 train:val split, balanced benchmark subset.
Uses cached image sizes for speed."""

import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

DATA_DIR = Path("/workspace/finetune/data/data")
JSON_PATH = DATA_DIR / "5k_10feats_v2_augmented.json"
IMAGE_DIR = DATA_DIR / "IM_D03_PT_5k_Augmented"
OUTPUT_DIR = DATA_DIR
CACHE_PATH = DATA_DIR / "image_sizes_cache.json"
TRAIN_RATIO = 0.80

# NEW prompt (from Prompt 10 Features.txt)
SYSTEM_PROMPT = """任务：
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

# Load image sizes cache
print("Loading image sizes cache...")
if CACHE_PATH.exists():
    with open(CACHE_PATH) as f:
        image_sizes = json.load(f)
    print(f"Loaded {len(image_sizes)} cached image sizes")
else:
    print("WARNING: No image sizes cache found. Building from scratch...")
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor
    image_sizes = {}
    all_images = list(IMAGE_DIR.glob("*.*"))
    def get_size(p):
        try:
            with Image.open(p) as img:
                return p.name, img.size
        except:
            return p.name, None
    with ThreadPoolExecutor(max_workers=32) as pool:
        for name, size in pool.map(get_size, all_images):
            if size:
                image_sizes[name] = list(size)
    with open(CACHE_PATH, "w") as f:
        json.dump(image_sizes, f)
    print(f"Built and cached {len(image_sizes)} image sizes")

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

    if img_name not in image_sizes:
        skipped += 1
        continue

    w, h = image_sizes[img_name]

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
                "bbox": [nx1, ny1, nx2, ny2],
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
        "channel": "10feats_newprompt_4b",
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
train_file = OUTPUT_DIR / "train_10feats_newprompt_4b.jsonl"
with open(train_file, "w", encoding="utf-8") as f:
    for i in sorted(train_indices):
        f.write(json.dumps(records[i]["record"], ensure_ascii=False) + "\n")
print(f"Wrote {train_file}")

# Write val JSONL
val_file = OUTPUT_DIR / "val_10feats_newprompt_4b.jsonl"
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

bench_file = OUTPUT_DIR / "benchmark_10feats_newprompt_4b.jsonl"
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