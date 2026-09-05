"""
Step 1 (Swift): Convert simens_7feats_v1.json + images into ms-swift JSONL train/val/test.

7 categories (Siemens silver-plate/bending dataset):
  - Round Hole
  - Rectangular Hole
  - Threaded Hole
  - Slotted Hole
  - Fillet
  - Bending
  - Silver Plating

Output format (per line):
  {
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "<image>..."},
      {"role": "assistant", "content": "<json string>"}
    ],
    "images": ["<abs path>"]
  }

bbox is normalized to integer 0-1000 range.

Stratified splitting ensures rare labels (Threaded Hole, Silver Plating)
are represented in every split.
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

from PIL import Image

# ============ Defaults ============
SCRIPT_DIR = Path(__file__).parent
DEFAULT_JSON_PATH = SCRIPT_DIR / "simens_7feats_v1.json"
DEFAULT_IMAGE_DIR = SCRIPT_DIR / "simens_7feats"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "dataset"
DEFAULT_IMAGE_DEPLOY_DIR = "/workspace/data/simens_7feats"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10  # remainder => test
SEED = 42

# Rare labels that need stratified sampling
RARE_LABELS = {"Threaded Hole", "Silver Plating"}

ALLOWED_LABELS = {
    "Round Hole",
    "Rectangular Hole",
    "Threaded Hole",
    "Slotted Hole",
    "Fillet",
    "Bending",
    "Silver Plating",
}

SYSTEM_PROMPT = """任务：
在输入的工程图纸中定位、分类并提取以下结构特征的实例及其组，并提取对应尺寸参数，输出 JSON 列表。

需要识别三大类：
1. 孔类结构
2. 圆角结构
3. 折弯、镀银

========================
1. 类别定义
========================

1.1 孔类结构（4 类）
  - 圆孔 (Round Hole)
  - 矩形孔 (Rectangular Hole)
  - 螺纹孔 (Threaded Hole)
  - 腰孔 (Slotted Hole)

1.2 圆角结构（1 类）
  - 圆角 (Fillet)

1.3 工艺特征（2 类）
  - 折弯 (Bending)
  - 镀银 (Silver Plating)

========================
2. 几何定义与尺寸参数提取规则
========================

2.1 圆孔
- 轮廓：闭合圆
- 尺寸参数 Ø：直径
- 备注：未标注深度默认通孔，盲孔必须标注 DP
  例：Ø18 → "Ø18"；Ø18 DP20 → "Ø18 DP20"

2.2 矩形孔
- 轮廓：长方形（含正方形）
- 正方形：A → "□A" 或 "AxA"；长方形：L×W → "LxW"
  例：□18；12x12；20x14

2.3 螺纹孔
- 轮廓：内同心圆粗实线闭合圆，外同心圆细实线 3/4 弧
- 尺寸参数：公称直径 M、螺距 P、深度 DP、左旋 LH
  例：M8；M8P1；M8P1LH DP10

2.4 腰孔
- 轮廓：两端半圆弧 + 两条切线平行直线段
- 尺寸参数：Ø 半圆弧直径 + 长度 L；或 2xR 半径 + 长度；或 L×W 矩形尺寸
  例：Ø10 20；2xR5 20；10x20

2.5 圆角
- 几何定义：圆弧连接两条相交直线（可 1/4 或 1/2 圆角）
- 半圆圆角必须整体提取，不得拆分
  例：R3；Ø6

2.6 折弯
- 相邻板材/管材形成 V 形角度
  例：90° R3；45° Rmin；90°
- bbox 应覆盖折弯及相邻的两段板材或管材

2.7 镀银
- 虚线框标出的板材区域
  例：105 +5/0；90
- bbox 应覆盖镀银区域

========================
3. 输出格式
========================

严格返回 JSON 列表：

[
  {
    "category": "<英文类别>",
    "bbox": [x_min, y_min, x_max, y_max],
    "size": "<尺寸参数字符串>"
  }
]

允许的 category：
Round Hole
Rectangular Hole
Threaded Hole
Slotted Hole
Fillet
Bending
Silver Plating

bbox：
- 归一化坐标（0-1000 范围）
- 整数
- [x_min, y_min, x_max, y_max]

========================
4. 补充规则
========================

1. 所有单实例必须可见且闭合。
2. 组内单实例必须单独输出；组整体也必须输出（组 size 含数量）。
3. 尺寸提取优先从特征附近标注获取；缺失则继承所属组的单件尺寸。
4. 螺纹孔识别优先：尺寸含 M 归为螺纹孔，而非圆孔。
5. 同一层级不得重复框。
6. 必须检测所有孔、圆角、折弯、镀银实例。

========================
输出要求
========================

仅返回 JSON 结果。
不得附加任何解释、说明或额外文本。"""

USER_PROMPT = "请分析这张工程图纸，识别并提取所有孔类、圆角、折弯以及镀银特征，返回 JSON 列表。"


def normalize_bbox(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    nx1 = int(round(x1 / img_w * 1000))
    ny1 = int(round(y1 / img_h * 1000))
    nx2 = int(round(x2 / img_w * 1000))
    ny2 = int(round(y2 / img_h * 1000))
    nx1 = max(0, min(1000, nx1))
    ny1 = max(0, min(1000, ny1))
    nx2 = max(0, min(1000, nx2))
    ny2 = max(0, min(1000, ny2))
    if nx2 <= nx1:
        nx2 = min(1000, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(1000, ny1 + 1)
    return [nx1, ny1, nx2, ny2]


def convert_sample(item, local_image_dir: Path, deploy_image_dir: str):
    image_name = item["dataitem_name"]
    local_path = local_image_dir / image_name

    try:
        with Image.open(local_path) as img:
            img_w, img_h = img.size
    except Exception as e:
        print(f"WARNING: Cannot open {local_path}: {e}, skipping")
        return None

    features = []
    for task in item.get("tasks", []):
        for tv in task.get("task_values", []):
            val = tv["value"]
            label = val["label"]
            if label not in ALLOWED_LABELS:
                continue
            bbox = tv.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            norm_bbox = normalize_bbox(bbox, img_w, img_h)
            features.append({
                "category": label,
                "size": val.get("size", ""),
                "bbox": norm_bbox,
            })

    answer = json.dumps(features, ensure_ascii=False, indent=2)

    # Use the deploy path in the JSONL (the training pod's mounted path)
    image_out = f"{deploy_image_dir.rstrip('/')}/{image_name}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>{USER_PROMPT}"},
            {"role": "assistant", "content": answer},
        ],
        "images": [image_out],
    }


def stratified_split(valid_data, train_ratio, val_ratio, seed):
    rng = random.Random(seed)
    rare_idx = []
    common_idx = []
    for i, item in enumerate(valid_data):
        labels = {
            tv["value"]["label"]
            for t in item.get("tasks", [])
            for tv in t.get("task_values", [])
        }
        if labels & RARE_LABELS:
            rare_idx.append(i)
        else:
            common_idx.append(i)

    rng.shuffle(rare_idx)
    rng.shuffle(common_idx)

    def split3(lst):
        n = len(lst)
        ntr = int(n * train_ratio)
        nval = int(n * val_ratio)
        return lst[:ntr], lst[ntr:ntr + nval], lst[ntr + nval:]

    tr_r, val_r, te_r = split3(rare_idx)
    tr_c, val_c, te_c = split3(common_idx)

    train = tr_r + tr_c
    val = val_r + val_c
    test = te_r + te_c

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def parse_args():
    parser = argparse.ArgumentParser(description="Build ms-swift JSONL for Siemens 7-feature dataset")
    parser.add_argument("--json_path", type=str, default=str(DEFAULT_JSON_PATH))
    parser.add_argument("--image_dir", type=str, default=str(DEFAULT_IMAGE_DIR),
                        help="Local directory holding the PNG images (used only for size lookup)")
    parser.add_argument("--deploy_image_dir", type=str, default=DEFAULT_IMAGE_DEPLOY_DIR,
                        help="Image directory path to embed in JSONL (where images will live on the training pod)")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--prefix", type=str, default="7feats",
                        help="Output filename prefix: train_{prefix}.jsonl, val_{prefix}.jsonl, test_{prefix}.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    json_path = Path(args.json_path)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        raw = json.load(f)

    available = {f for f in os.listdir(image_dir) if f.lower().endswith(".png")}
    valid = [it for it in raw if it["dataitem_name"] in available]
    print(f"Total annotated items: {len(raw)}")
    print(f"With local images:    {len(valid)}")

    train_idx, val_idx, test_idx = stratified_split(
        valid, args.train_ratio, args.val_ratio, args.seed
    )
    print(f"\nSplit (seed={args.seed}):")
    print(f"  train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}")

    def dump(indices, fname):
        path = output_dir / fname
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for i in indices:
                sample = convert_sample(valid[i], image_dir, args.deploy_image_dir)
                if sample is None:
                    continue
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n += 1
        print(f"  Wrote {n} lines -> {path}")
        return n

    print("\nWriting JSONL files:")
    dump(train_idx, f"train_{args.prefix}.jsonl")
    dump(val_idx, f"val_{args.prefix}.jsonl")
    dump(test_idx, f"test_{args.prefix}.jsonl")

    # Per-split label distribution for sanity
    def dist(indices):
        c = Counter()
        for i in indices:
            for t in valid[i].get("tasks", []):
                for tv in t.get("task_values", []):
                    c[tv["value"]["label"]] += 1
        return c

    print("\n=== Label distribution ===")
    header = f"{'label':<22s} {'train':>8s} {'val':>8s} {'test':>8s}"
    print(header)
    d_tr, d_val, d_te = dist(train_idx), dist(val_idx), dist(test_idx)
    all_labels = sorted(set(d_tr) | set(d_val) | set(d_te))
    for lbl in all_labels:
        print(f"{lbl:<22s} {d_tr.get(lbl,0):>8d} {d_val.get(lbl,0):>8d} {d_te.get(lbl,0):>8d}")


if __name__ == "__main__":
    main()
