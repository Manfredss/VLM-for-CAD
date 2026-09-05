"""
Convert 788_11Feats_View.json -> multi-turn ms-swift JSONL (improved variant).

Improvements over scripts/qwen3.5-27b_788_with_view/prepare_dataset_swift.py:
  - A: OVERSAMPLE_MAP duplicates train samples that touch tail labels
       (Rear/Bottom/Right/Section/Detail/Auxiliary View, Threaded Hole).
  - B: STEP1 prompt sharpens the rules for the weak categories from the
       prior 2-A100 run (Notes bbox boundary, Rear View identification,
       emphasis that rare classes must still be detected).

Two-step conversation:
  Turn 1 (Views & Layout):    15 allowed categories, no size
  Turn 2 (Structural Features): 10 categories, with size
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
DEFAULT_IMAGE_DIR = SCRIPT_DIR.parent / "qwn3.5-27b_788_silver_plate_bend" / "simens_7feats"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "dataset"
DEFAULT_IMAGE_DEPLOY_DIR = "/workspace/data/simens_7feats"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
SEED = 42

# ============ Categories ============

VIEW_CATEGORIES = {
    "Title Block",
    "Notes",
    "Revision Table",
    "Bill of Materials",
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
    "Round Hole Group",
    "Rectangular Hole Group",
    "Slotted Hole Group",
    "Fillet",
    "Bending",
    "Silver Plating",
}

# Stratified-split protection.
RARE_LABELS = {
    "Threaded Hole",
    "Section View",
    "Detail View",
    "Auxiliary View",
    "Orthographic Projection - Rear View",
}

# A: Oversampling multipliers for tail labels.
# Duplicates a TRAIN sample N times, where N = max multiplier across all labels
# in the sample. Targets the categories that scored 0-0.79 F1 on the 2-A100 run.
OVERSAMPLE_MAP = {
    "Section View": 5,
    "Detail View": 5,
    "Auxiliary View": 5,
    "Orthographic Projection - Rear View": 4,
    "Threaded Hole": 3,
    "Orthographic Projection - Bottom View": 2,
    "Orthographic Projection - Right View": 2,
}

# ============ Prompts ============

SYSTEM_PROMPT = """你是一个专业的工业图纸分析助手。请按两步分析工程图纸：
第一步：识别图纸中所有视图区域和文档元素；
第二步：检测图纸中的所有结构特征（孔、圆角、折弯、镀银），提取类别、尺寸和位置。
每一步仅返回 JSON 数组，不附加任何解释。"""

# B: Improved STEP1 prompt
#   - Notes: explicit bbox boundary rule (cover full block, no per-line splits)
#   - Rear View: positive identification rule
#   - Rare views (Section/Detail/Auxiliary): emphasize they must still be detected when present
#   - Projection symbol still leveraged for 1st/3rd angle
STEP1_USER_PROMPT = """任务：分析工程图纸的整体布局和视图结构。

========================
1. 文档元素（4 类）
========================

1.1 标题栏 (Title Block)
- 通常位于图纸右下角
- 包含零件号、名称、材料、比例、日期等元数据
- 有固定的表格格式和边框
- bbox 应覆盖标题栏完整外框

1.2 注释 (Notes)
- 通常位于图纸左下角或标题栏附近
- 包含加工要求、表面处理、公差标准等文字说明
- 通常以编号列表形式出现（如 1. 2. 3. ... 或圆圈编号）
- bbox 必须覆盖整个注释块的完整区域；
  不得按行拆分；包含编号、所有文字行与紧邻的辅助标识

1.3 修改表 (Revision Table)
- 通常位于图纸右上角或标题栏上方
- 记录图纸修改历史（修改编号、日期、描述）
- 表格形式，包含 REV、DATE、DESCRIPTION 等列
- 不一定存在；仅在出现时检测

1.4 材料清单 (Bill of Materials)
- 零件清单表格，通常位于标题栏上方
- 列出组件编号、名称、数量、材料等信息
- 在装配图中出现，零件图中极少出现
- 不一定存在；仅在出现时检测

========================
2. 视图区域（11 类）
========================

2.1 正交投影视图（6 类）

【投影法 — 关键前提】
本数据集所有图纸均采用 ISO 第一角投影法 (ISO 5456-2 / GB/T 17452，
Siemens / 西门子德标制图惯例)。**不要假设第三角**。
第一角投影下，各视图相对正视图的位置如下：
- 俯视图 (Top View)    → 正视图 **正下方**
- 仰视图 (Bottom View) → 正视图 **正上方**
- 左视图 (Left View)   → 正视图 **正右侧**
- 右视图 (Right View)  → 正视图 **正左侧**
- 后视图 (Rear View)   → 通常位于左视图的右侧 或 右视图的左侧（与正视图水平对齐但不相邻）

【识别策略 — 必读】
分类视图必须先锚定正视图，再以其为参考推断其他视图，而非孤立判断每个视图：
1) 先找到尺寸标注最丰富、特征最完整的视图 → 这是正视图（一张图纸至多一个）。
2) 以正视图为锚点，按 ISO 第一角投影法分配其他正交视图（位置见上）。
3) 若仅有 1 个正交视图，标为正视图；不得对没有邻接关系的视图猜测分类。
4) 若图上有等轴测图，用其判断正面/顶面/侧面，以验证正交视图分类。

正视图 (Orthographic Projection - Front View)
- 主要视图，通常最能反映零件形状，常是最大的视图
- 一般位于图纸中央或中央偏左
- 其他视图的位置相对于正视图确定（依 ISO 第一角投影法）
- 钣金件：常显示折弯方向和折弯角度（例：在板的端面显示 V 形）
- 一张图纸至多包含一个正视图

俯视图 (Orthographic Projection - Top View)
- 位于正视图的 **正下方**（ISO 第一角投影法）
- 显示物体从上方观察的形状
- 内容特征：顶面孔位、槽位、折弯线在顶面的投影；通常宽度 > 高度（扁形）

仰视图 (Orthographic Projection - Bottom View)
- 位于正视图的 **正上方**（ISO 第一角投影法）
- 显示物体从下方观察的形状
- 内容特征：底面孔位；与俯视图为水平轴镜像
- 通常较少见，仅在底面有独特特征（不与顶面镜像）时绘制；
  若图纸已存在俯视图，则同一位置不应再出现仰视图

左视图 (Orthographic Projection - Left View)
- 位于正视图的 **正右侧**（ISO 第一角投影法 — 注意是右，不是左）
- 显示物体从左侧观察的形状
- 一般为窄长形（高度 ≈ 正视图，宽度较小）

右视图 (Orthographic Projection - Right View)
- 位于正视图的 **正左侧**（ISO 第一角投影法 — 注意是左，不是右）
- 显示物体从右侧观察的形状
- 一般为窄长形（高度 ≈ 正视图，宽度较小）

后视图 (Orthographic Projection - Rear View)
- 与正视图轮廓相似，但内容上左右镜像（绕铅直轴旋转 180°）
- 通常出现在左视图的右侧 或 右视图的左侧（与正视图水平对齐但不直接相邻）
- 与正视图无直接的上下/左右对应关系——这是与其他正交视图的关键区分点
- 即使罕见，仍必须独立检测

2.2 其他视图（5 类）

等轴测图 (Isometric View)
- 三维透视图，同时显示三个面（顶面、正面、侧面）
- 通常位于图纸右上角或空白区域
- 用于直观展示零件整体形状
- 不用于尺寸标注

展开视图 (Flat Pattern View)
- 钣金件展开为平面后的二维视图
- 关键识别特征（必有其一）：
  · 视图内有清晰的折弯线（虚线/点划线/中心线类型）
  · 折弯角度标注（例 "90° R3"、"45° Rmin"）位于视图内或紧邻
  · 视图旁有 "Flat Pattern"、"展开"、"DEVELOPED"、"BLANK" 等文字标识
- 关键区分（重要）：
  · 即使展开视图位于图纸中央，只要带上述折弯线/折弯标注，也必须识别为展开视图，不得归为正视图
  · 展开后的轮廓与折弯后的零件不同（包含展开余量），通常更长或更扁平
- 用于钣金加工与下料

剖视图 (Section View)
- 假想用剖切面剖开物体后的投影视图
- 特征：剖面区域有阴影线（剖面线/截面线）
- 通常标有剖切线位置和方向（如 A-A）
- 即使罕见，仍必须独立检测

详图 (Detail View)
- 对原视图局部区域的放大视图
- 通常用圆圈或方框在原视图中标记，并以引出线连接到放大区
- 用于清晰展示小特征
- 即使罕见，仍必须独立检测

辅助视图 (Auxiliary View)
- 沿某个倾斜面的法线方向投影
- 用于展示倾斜面的真实形状
- 通常标有视图方向箭头和标识字母
- 即使罕见，仍必须独立检测

========================
3. 输出格式
========================

严格返回 JSON 列表：

[
  {"category": "<类别英文名>", "bbox": [x_min, y_min, x_max, y_max]}
]

允许的 category（共 15 类）：
Title Block
Notes
Revision Table
Bill of Materials
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
2. 修改表和材料清单不一定存在；仅在图纸中实际出现时检测，不得凭空生成。
3. 钣金件通常包含 1 个展开视图。
4. 视图分类需综合考虑位置关系和内容特征。
5. **本数据集图纸全部为 ISO 第一角投影法**；不得按第三角投影解释视图位置。
6. 罕见类别（剖视图、详图、辅助视图、后视图、修改表、材料清单）出现时必须独立检测；不得遗漏。

【一致性约束 — 必须满足】
7. 每张图纸至多 1 个 Front View。若有多个候选，仅最大且尺寸标注最丰富者为 Front View。
8. 投影法一致性（ISO 第一角）：
   · Top View 与正视图必须铅直对齐，且 Top 在正视图 **下方**；
   · Bottom View 与正视图铅直对齐，Bottom 在正视图 **上方**；
   · Left View 与正视图水平对齐，Left 在正视图 **右侧**；
   · Right View 与正视图水平对齐，Right 在正视图 **左侧**。
   若布局不满足上述对齐，请重新核对分类，避免把侧视图误标为俯视图等。
9. 同一组正交视图中，俯视图与仰视图不应同时出现（仅出现表达底面独特特征的一个）。
10. 等轴测图（如有）中可见的"正面"应与正视图轮廓一致；"顶面"应与俯视图轮廓一致。若不一致，请重新核对正视图分类。
11. 仅出现一个正交视图时：若展示折弯线和折弯标注 → Flat Pattern View；其他情况默认为 Front View。

仅返回 JSON 结果。不得附加任何解释、说明或额外文本。"""

STEP2_USER_PROMPT = """基于上述布局分析，现在检测图纸中的所有结构特征。

========================
1. 类别定义（10 类）
========================

1.1 孔类结构（4 单实例 + 3 组 = 7 类）
- 圆孔 (Round Hole)
- 矩形孔 (Rectangular Hole)
- 螺纹孔 (Threaded Hole)
- 腰孔 (Slotted Hole)
- 圆孔组 (Round Hole Group)
- 矩形孔组 (Rectangular Hole Group)
- 腰孔组 (Slotted Hole Group)

1.2 圆角结构（1 类）
- 圆角 (Fillet)

1.3 工艺特征（2 类）
- 折弯 (Bending)
- 镀银 (Silver Plating)

========================
2. 几何定义与尺寸参数提取规则
========================

2.1 圆孔
- 轮廓：闭合圆；无螺纹标记的普通通孔或盲孔
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
- 角度标注规则（重要）：
  · size 字段必须直接复制图纸上标注的角度数值，不得自行计算或转换
  · 不要把内角换算为外角（或反之）。例：标注 "45°" → size = "45°"；标注 "135°" → size = "135°"
  · 同一折弯只输出一次，以图纸标注为准；45° 与 135° 是补角关系，不应同时出现
  · 若标注的是折弯半径或最小半径而无明确角度，按 "90° R3"、"R Rmin" 等原样保留

2.7 镀银
- 虚线框标出的板材区域
  例：105 +5/0；90
- bbox 应覆盖镀银区域

2.8 圆孔组 (Round Hole Group)
- 定义：多个相同尺寸的圆孔由统一标注描述
- 标注形式：N x D 或 N - D
- 示例：
  4x18 → "4x18"
  4-Ø18 → "4-Ø18"
- bbox 应覆盖该组内所有圆孔实例

2.9 矩形孔组 (Rectangular Hole Group)
- 定义：多个相同尺寸的矩形孔由统一标注描述
- 标注形式：N x L×W 或 N - L×W；正方形可写 N x □A 或 N x AxA
- 示例：
  2x□18 → "2x□18"
  2x18x18 → "2x18x18"
  3-20×14 → "3-20x14"
- bbox 应覆盖该组内所有矩形孔实例

2.10 腰孔组 (Slotted Hole Group)
- 定义：多个相同尺寸的腰孔由统一标注描述
- 标注形式：N x (Ø + 长度)、N x (2xR + 长度)、或 N x (LxW)
- 示例：
  2xØ10 20 → "2xØ10 20"
  3-2xR5 20 → "3-2xR5 20"
  4x10x20 → "4x10x20"
- bbox 应覆盖该组内所有腰孔实例

注意：本任务中螺纹孔、圆角、折弯、镀银均无组类别。

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

允许的 category（共 10 类）：
Round Hole
Rectangular Hole
Threaded Hole
Slotted Hole
Round Hole Group
Rectangular Hole Group
Slotted Hole Group
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
3. 孔分类优先级（严格按以下顺序判定）：
   - 尺寸含 M → 螺纹孔（非圆孔）
   - 两端半圆弧 + 平行直线段轮廓 → 腰孔（非矩形孔）
   - 四边形轮廓（含正方形）→ 矩形孔
   - 其余圆形闭合轮廓 → 圆孔
   判定后不得更改类别。
4. 同一层级不得重复框。
5. 必须检测所有孔、圆角、折弯、镀银实例。

5b. 组与单实例（重要）：
   - "组" 是独立检测对象。当多个相同尺寸的同类孔由统一 N x 尺寸 标注描述时：
     · 必须独立检测每个单实例（size 不含数量前缀）
     · 还必须额外检测整个组（size 含数量前缀；bbox 覆盖该组所有实例的外包围框）
     · 单实例与组属不同层级，不算重复框
   - 组识别条件：类型一致、尺寸一致、与组标注一致。
   - 若单实例附近无尺寸标注：在所属组的标注中查找并继承单件尺寸。

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
    nx1 = max(0, min(1000, int(round(x1 / img_w * 1000))))
    ny1 = max(0, min(1000, int(round(y1 / img_h * 1000))))
    nx2 = max(0, min(1000, int(round(x2 / img_w * 1000))))
    ny2 = max(0, min(1000, int(round(y2 / img_h * 1000))))
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
                view_items.append({"category": label, "bbox": norm_bbox})
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


def get_labels_in_item(item):
    return {
        tv["value"]["label"]
        for t in item.get("tasks", [])
        for tv in t.get("task_values", [])
    }


def stratified_split(valid_data, train_ratio, val_ratio, seed):
    rng = random.Random(seed)
    rare_idx, common_idx = [], []
    for i, item in enumerate(valid_data):
        if get_labels_in_item(item) & RARE_LABELS:
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
    train, val, test = tr_r + tr_c, val_r + val_c, te_r + te_c
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


# ============ Main ============

def parse_args():
    p = argparse.ArgumentParser(description="Build improved (oversampled) ms-swift JSONL")
    p.add_argument("--json_path", type=str, default=str(DEFAULT_JSON_PATH))
    p.add_argument("--image_dir", type=str, default=str(DEFAULT_IMAGE_DIR))
    p.add_argument("--deploy_image_dir", type=str, default=DEFAULT_IMAGE_DEPLOY_DIR)
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    p.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--prefix", type=str, default="view_7feats_improve")
    p.add_argument("--no_oversample", action="store_true",
                   help="Disable OVERSAMPLE_MAP (for ablation)")
    return p.parse_args()


def main():
    args = parse_args()
    json_path = Path(args.json_path)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = json.load(open(json_path))
    available = {f for f in os.listdir(image_dir) if f.lower().endswith(".png")}
    valid = [it for it in raw if it["dataitem_name"] in available]
    print(f"Total annotated items: {len(raw)}")
    print(f"With local images:    {len(valid)}")

    train_idx, val_idx, test_idx = stratified_split(
        valid, args.train_ratio, args.val_ratio, args.seed
    )
    print(f"\nSplit (seed={args.seed}):  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    step1_lens, step2_lens = [], []

    def dump(indices, fname, oversample=False):
        path = output_dir / fname
        n = 0
        oversampled = 0
        with open(path, "w", encoding="utf-8") as f:
            for i in indices:
                sample = convert_sample(valid[i], image_dir, args.deploy_image_dir)
                if sample is None:
                    continue
                step1_lens.append(len(sample["messages"][2]["content"]))
                step2_lens.append(len(sample["messages"][4]["content"]))
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n += 1
                if oversample and not args.no_oversample:
                    labels = get_labels_in_item(valid[i])
                    mult = max((OVERSAMPLE_MAP.get(lab, 1) for lab in labels), default=1)
                    for _ in range(mult - 1):
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        n += 1
                        oversampled += 1
        print(f"  Wrote {n} lines -> {path}" +
              (f"  (oversampled +{oversampled})" if oversampled else ""))
        return n

    print("\nWriting JSONL files:")
    dump(train_idx, f"train_{args.prefix}.jsonl", oversample=True)
    dump(val_idx, f"val_{args.prefix}.jsonl", oversample=False)
    dump(test_idx, f"test_{args.prefix}.jsonl", oversample=False)

    def dist(indices):
        c = Counter()
        for i in indices:
            for t in valid[i].get("tasks", []):
                for tv in t.get("task_values", []):
                    label = tv["value"]["label"]
                    if label in VIEW_CATEGORIES or label in FEATURE_CATEGORIES:
                        c[label] += 1
        return c

    print("\n=== Label distribution (allowed only, raw counts pre-oversample) ===")
    print(f"{'label':<42s} {'train':>7s} {'val':>5s} {'test':>5s} {'mult':>5s}")
    d_tr, d_val, d_te = dist(train_idx), dist(val_idx), dist(test_idx)
    all_labels = sorted(set(d_tr) | set(d_val) | set(d_te),
                        key=lambda x: -(d_tr.get(x, 0) + d_val.get(x, 0) + d_te.get(x, 0)))
    for lbl in all_labels:
        kind = "view" if lbl in VIEW_CATEGORIES else "feat"
        mult = OVERSAMPLE_MAP.get(lbl, 1) if not args.no_oversample else 1
        print(f"  [{kind}] {lbl:<36s} {d_tr.get(lbl,0):>7d} {d_val.get(lbl,0):>5d} {d_te.get(lbl,0):>5d} {mult:>5d}x")


if __name__ == "__main__":
    main()
