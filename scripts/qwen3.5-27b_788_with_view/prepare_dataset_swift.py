"""
Convert 788_11Feats_View.json -> multi-turn ms-swift JSONL.

Two-step conversation:
  Turn 1 (Views & Layout):    13 categories, no size
  Turn 2 (Structural Features): 7 categories from silver_plate_bend, with size

Output JSONL line schema:
  {
    "messages": [
      {"role": "system",    "content": <system prompt>},
      {"role": "user",      "content": "<image>" + STEP1_USER_PROMPT},
      {"role": "assistant", "content": <step 1 JSON>},
      {"role": "user",      "content": STEP2_USER_PROMPT},
      {"role": "assistant", "content": <step 2 JSON>},
    ],
    "images": ["<deploy path>"]
  }

bbox is normalized to integer 0-1000 range. Stratified split keeps rare
labels represented in train/val/test.
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
DEFAULT_JSON_PATH = SCRIPT_DIR / "788_11Feats_View.json"
# Images live in the silver_plate_bend dir (this dataset is a subset).
DEFAULT_IMAGE_DIR = SCRIPT_DIR.parent / "qwn3.5-27b_788_silver_plate_bend" / "simens_7feats"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "dataset"
DEFAULT_IMAGE_DEPLOY_DIR = "/workspace/data/simens_7feats"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10  # remainder => test
SEED = 42

# ============ Categories ============

VIEW_CATEGORIES = {
    "Title Block",
    "Notes",
    "Orthographic Projection - Front View",
    "Orthographic Projection - Top View",
    "Orthographic Projection - Left View",
    "Orthographic Projection - Right View",
    "Orthographic Projection - Bottom View",
    "Orthographic Projection - Rear View",
    "Isometric View",
    "Flat Pattern View",
    "Section View",
    "Detail View",
    "Auxiliary View",
}

FEATURE_CATEGORIES = {
    "Round Hole",
    "Rectangular Hole",
    "Threaded Hole",
    "Slotted Hole",
    "Fillet",
    "Bending",
    "Silver Plating",
}

# Stratified-split protection: any image touching one of these is split rare.
RARE_LABELS = {
    "Threaded Hole",
    "Section View",
    "Detail View",
    "Auxiliary View",
    "Orthographic Projection - Rear View",
}

# ============ Prompts ============

SYSTEM_PROMPT = """你是一个专业的工业图纸分析助手。请按两步分析工程图纸：
第一步：识别图纸中所有视图区域和文档元素；
第二步：检测图纸中的所有结构特征（孔、圆角、折弯、镀银），提取类别、尺寸和位置。
每一步仅返回 JSON 数组，不附加任何解释。"""

STEP1_USER_PROMPT = """任务：分析工程图纸的整体布局和视图结构。

========================
1. 文档元素（2 类）
========================

1.1 标题栏 (Title Block)
- 通常位于图纸右下角
- 包含零件号、名称、材料、比例、日期等元数据
- 有固定的表格格式和边框

1.2 注释 (Notes)
- 通常位于图纸左下角或标题栏附近
- 包含加工要求、表面处理、公差标准等文字说明
- 可能以编号列表形式呈现

========================
2. 视图区域（11 类）
========================

2.1 正交投影视图（6 类）

工程图纸使用投影法将三维物体表示为多个二维视图。
图纸可能采用第一角或第三角投影法。各视图相对位置在两种投影法下相反。

正视图 (Orthographic Projection - Front View)
- 主要视图，通常最能反映零件形状，常是最大的视图
- 一般位于图纸中央或中央偏左
- 其他视图的位置相对于正视图确定

俯视图 (Orthographic Projection - Top View)
- 第三角投影法：位于正视图的正上方
- 第一角投影法：位于正视图的正下方
- 显示物体从上方观察的形状

仰视图 (Orthographic Projection - Bottom View)
- 第三角投影法：位于正视图的正下方
- 第一角投影法：位于正视图的正上方
- 显示物体从下方观察的形状

左视图 (Orthographic Projection - Left View)
- 第三角投影法：位于正视图的左侧
- 第一角投影法：位于正视图的右侧
- 显示物体从左侧观察的形状

右视图 (Orthographic Projection - Right View)
- 第三角投影法：位于正视图的右侧
- 第一角投影法：位于正视图的左侧
- 显示物体从右侧观察的形状

后视图 (Orthographic Projection - Rear View)
- 通常位于左视图或右视图的旁边
- 显示物体从后方观察的形状（绕铅直轴旋转 180° 后的投影）

2.2 其他视图（5 类）

等轴测图 (Isometric View)
- 三维透视图，同时显示三个面（顶面、正面、侧面）
- 通常位于图纸右上角或空白区域
- 用于直观展示零件整体形状
- 不用于尺寸标注

展开视图 (Flat Pattern View)
- 钣金件展开为平面后的二维视图
- 通常带折弯线、折弯标识或加工标识
- 用于钣金加工与下料

剖视图 (Section View)
- 假想用剖切面剖开物体后的投影视图
- 特征：剖面区域有阴影线（剖面线/截面线）
- 通常标有剖切线位置和方向（如 A-A）
- 用于展示内部结构

详图 (Detail View)
- 对原视图局部区域的放大视图
- 通常用圆圈或方框在原视图中标记，并以引出线连接到放大区
- 用于清晰展示小特征

辅助视图 (Auxiliary View)
- 沿某个倾斜面的法线方向投影
- 用于展示倾斜面的真实形状
- 通常标有视图方向箭头和标识字母

========================
3. 输出格式
========================

严格返回 JSON 列表：

[
  {"category": "<类别英文名>", "bbox": [x_min, y_min, x_max, y_max]}
]

允许的 category（共 13 类）：
Title Block
Notes
Orthographic Projection - Front View
Orthographic Projection - Top View
Orthographic Projection - Bottom View
Orthographic Projection - Left View
Orthographic Projection - Right View
Orthographic Projection - Rear View
Isometric View
Flat Pattern View
Section View
Detail View
Auxiliary View

bbox：归一化坐标（0-1000 范围），整数，[x_min, y_min, x_max, y_max]。
bbox 应覆盖该视图/元素的完整区域。

========================
4. 补充规则
========================

1. 每张图纸通常包含 1 个标题栏、1-3 个注释区域、2-4 个投影视图。
2. 钣金件通常包含 1 个展开视图。
3. 视图分类需综合考虑位置关系和内容特征。
4. 若图纸右下角有投影符号（⊕），可据此判断第一角或第三角投影法；无符号时根据视图相对位置推断。

仅返回 JSON 结果。不得附加任何解释、说明或额外文本。"""

STEP2_USER_PROMPT = """基于上述布局分析，现在检测图纸中的所有结构特征。

========================
1. 类别定义（7 类）
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
2. 尺寸提取优先从特征附近标注获取；缺失则继承所属组的单件尺寸。
3. 螺纹孔识别优先：尺寸含 M 归为螺纹孔，而非圆孔。
4. 同一层级不得重复框。
5. 必须检测所有孔、圆角、折弯、镀银实例。

6. 跨视图判断（重要）：对轮廓相似的特征，需综合多视图与尺寸标注判断类别：
   - 螺纹孔 vs 圆孔：俯视图轮廓相同（闭合圆）；螺纹孔在正视图/剖视图中可见螺纹线，或尺寸以 M 开头；圆孔无螺纹标记，尺寸为 Ø 或纯数字。
   - 腰孔：俯视图最易辨识（两端半圆 + 平行直线段）；正视图通常显示为矩形开口，需结合俯视图避免误判为矩形孔。
   - 折弯：俯视图常显示折弯线；折弯角度与半径多在正视图或剖视图中标注；展开视图（Flat Pattern View）反映展开尺寸。bbox 应在最能反映折弯几何的视图中给出，并覆盖折弯及相邻板材。
   - 镀银：虚线框区域可跨多个视图；在每个视图中独立标注其所在的镀银区域。
   - 同一物理特征若在多个视图中分别出现，每个视图实例都应独立标注（视图内不重复）。

仅返回 JSON 结果。不得附加任何解释、说明或额外文本。"""


# ============ Helpers ============

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

    view_items = []
    feature_items = []

    for task in item.get("tasks", []):
        for tv in task.get("task_values", []):
            val = tv["value"]
            label = val["label"]
            bbox = tv.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            norm_bbox = normalize_bbox(bbox, img_w, img_h)

            if label in VIEW_CATEGORIES:
                view_items.append({
                    "category": label,
                    "bbox": norm_bbox,
                })
            elif label in FEATURE_CATEGORIES:
                feature_items.append({
                    "category": label,
                    "size": val.get("size", ""),
                    "bbox": norm_bbox,
                })

    step1_answer = json.dumps(view_items, ensure_ascii=False, indent=2)
    step2_answer = json.dumps(feature_items, ensure_ascii=False, indent=2)

    image_out = f"{deploy_image_dir.rstrip('/')}/{image_name}"

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"<image>{STEP1_USER_PROMPT}"},
            {"role": "assistant", "content": step1_answer},
            {"role": "user",      "content": STEP2_USER_PROMPT},
            {"role": "assistant", "content": step2_answer},
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


# ============ Main ============

def parse_args():
    p = argparse.ArgumentParser(description="Build multi-turn ms-swift JSONL for views + 7 features")
    p.add_argument("--json_path", type=str, default=str(DEFAULT_JSON_PATH))
    p.add_argument("--image_dir", type=str, default=str(DEFAULT_IMAGE_DIR),
                   help="Local directory holding the PNGs (used only for size lookup)")
    p.add_argument("--deploy_image_dir", type=str, default=DEFAULT_IMAGE_DEPLOY_DIR,
                   help="Path embedded in JSONL (where images live on the training pod)")
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    p.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--prefix", type=str, default="view_7feats",
                   help="Output filename prefix: train_{prefix}.jsonl, val_{prefix}.jsonl, test_{prefix}.jsonl")
    return p.parse_args()


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
    if len(valid) < len(raw):
        missing = [it["dataitem_name"] for it in raw if it["dataitem_name"] not in available]
        print(f"  Missing locally ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")

    train_idx, val_idx, test_idx = stratified_split(
        valid, args.train_ratio, args.val_ratio, args.seed
    )
    print(f"\nSplit (seed={args.seed}):")
    print(f"  train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}")

    step1_lens = []
    step2_lens = []

    def dump(indices, fname):
        path = output_dir / fname
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for i in indices:
                sample = convert_sample(valid[i], image_dir, args.deploy_image_dir)
                if sample is None:
                    continue
                step1_lens.append(len(sample["messages"][2]["content"]))
                step2_lens.append(len(sample["messages"][4]["content"]))
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n += 1
        print(f"  Wrote {n} lines -> {path}")
        return n

    print("\nWriting JSONL files:")
    dump(train_idx, f"train_{args.prefix}.jsonl")
    dump(val_idx, f"val_{args.prefix}.jsonl")
    dump(test_idx, f"test_{args.prefix}.jsonl")

    # Per-split label distribution
    def dist(indices):
        c = Counter()
        for i in indices:
            for t in valid[i].get("tasks", []):
                for tv in t.get("task_values", []):
                    label = tv["value"]["label"]
                    if label in VIEW_CATEGORIES or label in FEATURE_CATEGORIES:
                        c[label] += 1
        return c

    print("\n=== Label distribution (allowed only) ===")
    print(f"{'label':<42s} {'train':>7s} {'val':>5s} {'test':>5s}")
    d_tr, d_val, d_te = dist(train_idx), dist(val_idx), dist(test_idx)
    all_labels = sorted(set(d_tr) | set(d_val) | set(d_te),
                        key=lambda x: -(d_tr.get(x, 0) + d_val.get(x, 0) + d_te.get(x, 0)))
    for lbl in all_labels:
        kind = "view" if lbl in VIEW_CATEGORIES else "feat"
        print(f"  [{kind}] {lbl:<36s} {d_tr.get(lbl,0):>7d} {d_val.get(lbl,0):>5d} {d_te.get(lbl,0):>5d}")

    # Response length stats
    if step1_lens:
        step1_lens.sort()
        step2_lens.sort()
        print(f"\nStep 1 (views) response char lengths:    "
              f"median={step1_lens[len(step1_lens)//2]}  "
              f"p95={step1_lens[int(len(step1_lens)*0.95)]}  "
              f"max={step1_lens[-1]}")
        print(f"Step 2 (features) response char lengths: "
              f"median={step2_lens[len(step2_lens)//2]}  "
              f"p95={step2_lens[int(len(step2_lens)*0.95)]}  "
              f"max={step2_lens[-1]}")


if __name__ == "__main__":
    main()
