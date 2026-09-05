"""
Convert 5k_10feats_v2_augmented.json to ms-swift JSONL format for Qwen3.5-27B training.

- Uses the new "Prompt 10 Features" as system prompt
- Normalizes bbox to 0-1000 range (matching Qwen VL convention)
- Stratified 85/15 train/val split
- Generates a 50-image benchmark subset from val
"""

import json
import os
import random
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

TRAIN_RATIO = 0.85
SEED = 42
BENCHMARK_SIZE = 50

# The new 10-feature system prompt (bbox adapted to 0-1000 normalized coords)
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

11. 坐标归一化到 0-1000 范围，bbox 必须在该坐标系下返回。


========================
输出要求
========================

仅返回 JSON 结果。
不得附加任何解释、说明或额外文本。"""

USER_PROMPT = "<image>请分析这张工程图纸，识别并提取所有结构特征。"

# Category → channel mapping (for loss_scale compatibility)
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


def normalize_bbox(bbox, img_w, img_h):
    """Normalize pixel coordinates to 0-1000 range."""
    x1, y1, x2, y2 = bbox
    nx1 = int(round(x1 / img_w * 1000))
    ny1 = int(round(y1 / img_h * 1000))
    nx2 = int(round(x2 / img_w * 1000))
    ny2 = int(round(y2 / img_h * 1000))
    nx1 = max(0, min(1000, nx1))
    ny1 = max(0, min(1000, ny1))
    nx2 = max(0, min(1000, nx2))
    ny2 = max(0, min(1000, ny2))
    return [nx1, ny1, nx2, ny2]


def convert_sample(item, image_dir):
    """Convert one raw annotation to swift JSONL format."""
    image_name = item["dataitem_name"]
    image_path = str(image_dir / image_name)

    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
    except Exception as e:
        print(f"WARNING: Cannot open {image_path}: {e}, skipping normalization")
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
    else:
        channel = "default"

    answer = json.dumps(features, ensure_ascii=False, indent=2)

    conversation = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
            {"role": "assistant", "content": answer},
        ],
        "images": [image_path],
        "channel": channel,
    }
    return conversation


def get_dominant_category(item):
    """Get the dominant (most frequent) category in a sample."""
    cats = Counter()
    for task in item.get("tasks", []):
        for tv in task.get("task_values", []):
            cats[tv["value"]["label"]] += 1
    return cats.most_common(1)[0][0] if cats else "unknown"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--benchmark_size", type=int, default=BENCHMARK_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    json_path = Path(args.json_path)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    random.seed(args.seed)

    with open(json_path, "r") as f:
        data = json.load(f)

    # Only keep samples with local images
    available_images = set(f for f in os.listdir(image_dir) if f.endswith(".png"))
    valid_data = [item for item in data if item["dataitem_name"] in available_images]
    print(f"Total annotated samples: {len(data)}")
    print(f"Samples with local images: {len(valid_data)}")

    # Stratified sampling by dominant category
    category_indices = defaultdict(list)
    for i, item in enumerate(valid_data):
        cat = get_dominant_category(item)
        category_indices[cat].append(i)

    train_indices = []
    val_indices = []

    for cat, indices in category_indices.items():
        random.shuffle(indices)
        split = int(len(indices) * args.train_ratio)
        train_indices.extend(indices[:split])
        val_indices.extend(indices[split:])

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    print(f"\nTrain: {len(train_indices)} samples")
    print(f"Val:   {len(val_indices)} samples")

    # Create benchmark subset from val (stratified, up to benchmark_size)
    benchmark_indices = []
    val_category_indices = defaultdict(list)
    for i in val_indices:
        cat = get_dominant_category(valid_data[i])
        val_category_indices[cat].append(i)

    total_val = len(val_indices)
    for cat, indices in val_category_indices.items():
        quota = max(1, int(len(indices) / total_val * args.benchmark_size))
        benchmark_indices.extend(indices[:quota])

    benchmark_indices = benchmark_indices[:args.benchmark_size]
    random.shuffle(benchmark_indices)
    print(f"Benchmark: {len(benchmark_indices)} samples")

    # Convert and write
    os.makedirs(output_dir, exist_ok=True)

    def write_jsonl(indices, filename):
        count = 0
        with open(output_dir / filename, "w", encoding="utf-8") as f:
            for i in indices:
                sample = convert_sample(valid_data[i], image_dir)
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1
        return count

    train_count = write_jsonl(train_indices, "train_10feats.jsonl")
    val_count = write_jsonl(val_indices, "val_10feats.jsonl")
    benchmark_count = write_jsonl(benchmark_indices, "benchmark_10feats.jsonl")

    print(f"\nSaved to {output_dir}")
    print(f"  train_10feats.jsonl:     {train_count} samples")
    print(f"  val_10feats.jsonl:       {val_count} samples")
    print(f"  benchmark_10feats.jsonl: {benchmark_count} samples")

    # Label distribution
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
