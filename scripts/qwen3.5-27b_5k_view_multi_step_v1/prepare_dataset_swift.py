"""
Convert 5k_15feats_with_view_v2.json to multi-turn ms-swift JSONL.

Multi-step conversation design:
  Turn 1 (Layout & View Analysis):
    - User: <image> + layout detection prompt
    - Assistant: JSON list of view/layout elements (11 categories, no size)

  Turn 2 (Structural Feature Detection):
    - User: feature detection prompt (references layout context)
    - Assistant: JSON list of structural features (14 categories, with size)

This follows the industry pipeline:
  1. Layout segmentation + view classification (ISO 128 / ASME Y14)
  2. Feature detection + dimension extraction

Benefits:
  - Each step focuses on a distinct cognitive task
  - Step 2 benefits from layout context established in Step 1
  - View detection (currently weakest at 38.1%) gets dedicated attention
  - Reduced output length per turn → fewer truncation issues
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

# ============ Category Grouping ============

VIEW_CATEGORIES = {
    "Title Block", "Notes", "Revision Table", "Bill of Materials",
    "Orthographic Projection - Front View",
    "Orthographic Projection - Top View",
    "Orthographic Projection - Left View",
    "Orthographic Projection - Right View",
    "Isometric View", "Auxiliary View", "Section View",
}

FEATURE_CATEGORIES = {
    "Threaded Hole", "Threaded Hole Group",
    "Round Hole", "Round Hole Group",
    "Pin Hole", "Pin Hole Group",
    "Counterbore Hole", "Counterbore Hole Group",
    "Rectangular Hole",
    "Fillet", "Fillet Group",
    "Chamfer", "Chamfer Group",
    "Threaded Shaft",
}

# Rare categories for stratified sampling
RARE_LABELS = {
    "Threaded Shaft", "Section View", "Auxiliary View",
    "Revision Table", "Bill of Materials",
    "Counterbore Hole", "Counterbore Hole Group",
    "Chamfer", "Chamfer Group",
    "Rectangular Hole",
    "Orthographic Projection - Right View",
    "Isometric View",
}

# ============ Prompts (must match inference_swift.py exactly) ============

SYSTEM_PROMPT = """你是一个专业的工业图纸分析助手。你将按步骤分析工程图纸：
第一步：识别图纸的整体布局，包括各视图区域和文档元素；
第二步：检测图纸中的所有结构特征，提取类型、尺寸参数和位置坐标。
请根据图纸内容给出准确、完整的分析结果，每步仅返回JSON数组。"""

STEP1_USER_PROMPT = """任务：分析工程图纸的整体布局和视图结构。

========================
1. 文档元素（4 类）
========================

1.1 标题栏 (Title Block)
- 通常位于图纸右下角
- 包含零件号、名称、材料、比例、日期等元数据
- 有固定的表格格式和边框

1.2 注释 (Notes)
- 通常位于图纸左下角或标题栏附近
- 包含加工要求、表面处理、公差标准等文字说明
- 可能以编号列表形式呈现

1.3 修改表 (Revision Table)
- 通常位于图纸右上角或标题栏上方
- 记录图纸修改历史（修改编号、日期、描述）
- 表格形式，包含 REV、DATE、DESCRIPTION 等列

1.4 材料清单 (Bill of Materials)
- 零件清单表格，通常位于标题栏上方
- 列出组件编号、名称、数量、材料等信息
- 在装配图中出现，零件图中极少出现

========================
2. 视图区域（7 类）
========================

2.1 正交投影视图（4 类）
工程图纸使用投影法将三维物体表示为多个二维视图。

正视图 (Orthographic Projection - Front View)
- 主要视图，通常是最大或最能反映零件形状的视图
- 一般位于图纸中央偏左位置
- 其他视图的位置相对于正视图确定

俯视图 (Orthographic Projection - Top View)
- 第三角投影法：位于正视图的正上方
- 第一角投影法：位于正视图的正下方
- 显示物体从上方观察的形状

左视图 (Orthographic Projection - Left View)
- 第三角投影法：位于正视图的左侧
- 第一角投影法：位于正视图的右侧
- 显示物体从左侧观察的形状

右视图 (Orthographic Projection - Right View)
- 第三角投影法：位于正视图的右侧
- 第一角投影法：位于正视图的左侧
- 显示物体从右侧观察的形状

2.2 其他视图（3 类）

等轴测图 (Isometric View)
- 三维透视图，同时显示三个面（顶面、正面、侧面）
- 通常位于图纸右上角或空白区域
- 用于直观展示零件整体形状
- 不用于尺寸标注

辅助视图 (Auxiliary View)
- 沿某个倾斜面的法线方向投影
- 用于展示倾斜面的真实形状
- 通常标有视图方向箭头和标识字母

剖视图 (Section View)
- 假想用剖切面剖开物体后的投影视图
- 特征：剖面区域有阴影线（剖面线/截面线）
- 通常标有剖切线位置和方向（如 A-A）
- 用于展示内部结构

========================
3. 输出格式
========================

严格返回 JSON 列表：

[
  {"category": "<类别英文名>", "bbox": [x_min, y_min, x_max, y_max]}
]

允许的 category：
Title Block, Notes, Revision Table, Bill of Materials,
Orthographic Projection - Front View, Orthographic Projection - Top View,
Orthographic Projection - Left View, Orthographic Projection - Right View,
Isometric View, Auxiliary View, Section View

bbox：归一化坐标（0-1000 范围），整数，[x_min, y_min, x_max, y_max]。
bbox 应覆盖该视图/元素的完整区域。

========================
4. 补充规则
========================

1. 每张图纸通常包含：1个标题栏、1-3个注释区域、2-4个投影视图。
2. 修改表和材料清单不一定存在。
3. 视图分类需综合考虑位置关系和内容特征。
4. 若图纸右下角有投影符号（⊕），可据此判断第一角或第三角投影法。

仅返回 JSON 结果。不得附加任何解释、说明或额外文本。"""

STEP2_USER_PROMPT = """基于上述布局分析，现在检测图纸中的所有结构特征。

========================
1. 类别定义
========================

1.1 孔类结构（10 类）
- 圆孔 (Round Hole)
- 螺纹孔 (Threaded Hole)
- 销孔 (Pin Hole)
- 沉头孔 (Counterbore Hole)
- 矩形孔 (Rectangular Hole)
- 圆孔组 (Round Hole Group)
- 螺纹孔组 (Threaded Hole Group)
- 销孔组 (Pin Hole Group)
- 沉头孔组 (Counterbore Hole Group)

1.2 圆角与倒角结构（4 类）
- 圆角 (Fillet)
- 圆角组 (Fillet Group)
- 倒角 (Chamfer)
- 倒角组 (Chamfer Group)

1.3 轴类结构（1 类）
- 螺纹轴 (Threaded Shaft)

========================
2. 几何定义与尺寸参数提取规则
========================

--------------------------------
2.1 圆孔 (Round Hole)
--------------------------------
- 轮廓：闭合圆
- 无螺纹、无公差标注的普通通孔或盲孔
- 尺寸参数：直径值
  例：18 → "18"，Ø12 → "Ø12"，Ø24 DP15 → "Ø24 DP15"

--------------------------------
2.2 螺纹孔 (Threaded Hole)
--------------------------------
- 轮廓：闭合圆，带螺纹线标记
- 识别特征：尺寸包含字母 M（公制螺纹标识）
- 尺寸参数：M + 公称直径
  例：M8 → "M8"，M6 DP20 → "M6 DP20"

--------------------------------
2.3 销孔 (Pin Hole)
--------------------------------
- 轮廓：闭合圆，通常较小，用于定位对齐
- 识别特征：尺寸包含公差代号（如 H7、H6、G6）
- 尺寸参数：直径 + 公差
  例：Ø8H7 → "Ø8H7"，Ø6H7 DP20 → "Ø6H7 DP20"

--------------------------------
2.4 沉头孔 (Counterbore Hole)
--------------------------------
- 特征：阶梯孔，表面有较大直径的沉孔
- 识别特征：标注中包含两个不同直径
- 尺寸参数：外径 深度 内径
  例：Ø14 DP10 Ø9 → "Ø14 DP10 Ø9"，Ø21 Ø14.5 → "Ø21 Ø14.5"

--------------------------------
2.5 矩形孔 (Rectangular Hole)
--------------------------------
- 轮廓：四边形开口（含正方形）
- 尺寸参数：长×宽
  例：20x123.5 → "20x123.5"，50x160 → "50x160"

--------------------------------
2.6 圆角 (Fillet)
--------------------------------
- 一段圆弧，将两条相交直线/面光滑连接
- 尺寸参数：半径 R
  例：R3 → "R3"，R5 → "R5"

--------------------------------
2.7 倒角 (Chamfer)
--------------------------------
- 将直角边缘切除形成斜面
- 尺寸参数格式多样：
  - DxA 形式：D5xA45 → "D5xA45"，D1.2xA45 → "D1.2xA45"
  - C 形式：C1 → "C1"，C2 → "C2"
  - 角度形式：1x45° → "1x45°"

--------------------------------
2.8 螺纹轴 (Threaded Shaft)
--------------------------------
- 外螺纹圆柱特征
- 尺寸参数：M + 公称直径
  例：M10 → "M10"，M14 → "M14"

--------------------------------
2.9 组 (Group) 通用规则
--------------------------------
- 定义：多个相同尺寸特征由统一标注描述
- 标注形式：N x 尺寸 或 N - 尺寸
- bbox 应覆盖该组内所有实例
- 示例：
  4x18 → "4x18"（圆孔组）
  2xM8 → "2xM8"（螺纹孔组）
  2-Ø8H7 → "2-Ø8H7"（销孔组）
  2xR3 → "2xR3"（圆角组）
  2-C1 → "2-C1"（倒角组）
  2-Ø14 DP10 Ø9 → "2-Ø14 DP10 Ø9"（沉头孔组）

========================
3. 输出格式
========================

严格返回 JSON 列表：

[
  {
    "category": "<类别英文名>",
    "bbox": [x_min, y_min, x_max, y_max],
    "size": "<尺寸参数字符串>"
  }
]

允许的 category：
Round Hole, Threaded Hole, Pin Hole, Counterbore Hole, Rectangular Hole,
Round Hole Group, Threaded Hole Group, Pin Hole Group, Counterbore Hole Group,
Fillet, Fillet Group, Chamfer, Chamfer Group, Threaded Shaft

bbox：归一化坐标（0-1000 范围），整数，[x_min, y_min, x_max, y_max]

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
   - 单实例 size 不包含数量前缀
   - 组 size 必须包含数量前缀

5. 孔分类优先级：
   - 含 M → 螺纹孔（非圆孔）
   - 含 H7/H6 等公差 → 销孔
   - 双直径标注 → 沉头孔
   - 四边形轮廓 → 矩形孔
   - 其余 → 圆孔

6. 同一层级不得重复框。

仅返回 JSON 结果。不得附加任何解释、说明或额外文本。"""


def normalize_bbox(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    nx1 = max(0, min(1000, int(round(x1 / img_w * 1000))))
    ny1 = max(0, min(1000, int(round(y1 / img_h * 1000))))
    nx2 = max(0, min(1000, int(round(x2 / img_w * 1000))))
    ny2 = max(0, min(1000, int(round(y2 / img_h * 1000))))
    return [nx1, ny1, nx2, ny2]


def convert_sample(item, image_dir):
    """Convert a single data item into a multi-turn conversation."""
    image_name = item["dataitem_name"]
    image_path = str(image_dir / image_name)

    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
    except Exception as e:
        print(f"WARNING: Cannot open {image_path}: {e}")
        img_w, img_h = 1, 1

    # Split annotations into view vs feature
    view_items = []
    feature_items = []

    for task in item.get("tasks", []):
        for tv in task.get("task_values", []):
            val = tv["value"]
            label = val["label"]
            bbox = normalize_bbox(tv["bbox"], img_w, img_h)

            if label in VIEW_CATEGORIES:
                view_items.append({
                    "category": label,
                    "bbox": bbox,
                })
            elif label in FEATURE_CATEGORIES:
                feature_items.append({
                    "category": label,
                    "size": val.get("size", ""),
                    "bbox": bbox,
                })

    # Build multi-turn conversation
    step1_answer = json.dumps(view_items, ensure_ascii=False, indent=2)
    step2_answer = json.dumps(feature_items, ensure_ascii=False, indent=2)

    conversation = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>{STEP1_USER_PROMPT}"},
            {"role": "assistant", "content": step1_answer},
            {"role": "user", "content": STEP2_USER_PROMPT},
            {"role": "assistant", "content": step2_answer},
        ],
        "images": [image_path],
    }
    return conversation


def get_labels_in_item(item):
    labels = set()
    for t in item.get("tasks", []):
        for tv in t.get("task_values", []):
            labels.add(tv["value"]["label"])
    return labels


def parse_args():
    parser = argparse.ArgumentParser(description="Generate multi-step ms-swift JSONL datasets")
    parser.add_argument("--json_path", type=str, default=str(DEFAULT_JSON_PATH))
    parser.add_argument("--image_dir", type=str, default=str(IMAGE_DIR))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
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

    available_images = set(f for f in os.listdir(image_dir) if f.endswith(".png"))
    valid_data = [item for item in data if item["dataitem_name"] in available_images]
    print(f"Total annotated samples: {len(data)}")
    print(f"Samples with local images: {len(valid_data)}")

    # Stratified split
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

    print(f"\nTrain: {len(train_indices)} (rare: {rare_split}, common: {common_split})")
    print(f"Val:   {len(val_indices)} (rare: {len(rare_indices)-rare_split}, common: {len(common_indices)-common_split})")

    os.makedirs(output_dir, exist_ok=True)

    train_name = "train_multistep.jsonl"
    val_name = "val_multistep.jsonl"

    # Track stats
    step1_lens = []
    step2_lens = []

    train_count = 0
    with open(output_dir / train_name, "w", encoding="utf-8") as f:
        for i in train_indices:
            sample = convert_sample(valid_data[i], image_dir)
            step1_len = len(sample["messages"][2]["content"])
            step2_len = len(sample["messages"][4]["content"])
            step1_lens.append(step1_len)
            step2_lens.append(step2_len)
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

    # Response length stats
    step1_lens.sort()
    step2_lens.sort()
    print(f"\nStep 1 (views) response char lengths:")
    print(f"  median: {step1_lens[len(step1_lens)//2]}, "
          f"p95: {step1_lens[int(len(step1_lens)*0.95)]}, "
          f"max: {step1_lens[-1]}")
    print(f"Step 2 (features) response char lengths:")
    print(f"  median: {step2_lens[len(step2_lens)//2]}, "
          f"p95: {step2_lens[int(len(step2_lens)*0.95)]}, "
          f"max: {step2_lens[-1]}")

    # Category distribution
    view_dist = Counter()
    feat_dist = Counter()
    for item in valid_data:
        for task in item.get("tasks", []):
            for tv in task.get("task_values", []):
                label = tv["value"]["label"]
                if label in VIEW_CATEGORIES:
                    view_dist[label] += 1
                elif label in FEATURE_CATEGORIES:
                    feat_dist[label] += 1

    print("\n=== Step 1 categories (views) ===")
    for label, cnt in view_dist.most_common():
        print(f"  {label}: {cnt}")
    print(f"\n=== Step 2 categories (features) ===")
    for label, cnt in feat_dist.most_common():
        print(f"  {label}: {cnt}")


if __name__ == "__main__":
    main()
