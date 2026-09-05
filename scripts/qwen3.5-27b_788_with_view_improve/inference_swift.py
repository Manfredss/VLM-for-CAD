"""
inference_swift.py — Multi-turn inference for views + 7 features (improved variant).

Improvements over the base inference_swift.py:
  - B: post-hoc category snapping (--snap_categories, default on). If the model
       emits a category that isn't in the allowed list, snap to the closest valid
       one by string distance. Cheap substitute for token-level constrained decoding.
  - Same prompts as the improved prepare_dataset_swift.py (Notes bbox boundary,
       Rear View hint, rare-class emphasis).

Defaults aligned with the improved training output dir
`/workspace/output/swift_27b_788_view_7feats_improve/`.
"""

import os
import re
import json
import time
import argparse
import logging
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =====================================================================
# Device map resolution (same convention as silver_plate_bend)
# =====================================================================
def _resolve_device_map():
    value = os.environ.get("INFERENCE_DEVICE_MAP", "auto").strip()
    if not value or value.lower() == "auto":
        return "auto"
    if value.lower() == "split":
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            os.environ.get("INFERENCE_MODEL_PATH", "Qwen/Qwen3.5-27B"),
            trust_remote_code=True,
        )
        text_config = getattr(config, "text_config", None)
        num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
        if num_hidden_layers is None:
            num_hidden_layers = getattr(config, "num_hidden_layers", None)
        if not num_hidden_layers:
            raise ValueError("Unable to determine num_hidden_layers for split device map")

        device0 = os.environ.get("INFERENCE_DEVICE0", "cuda:0").strip() or "cuda:0"
        device1 = os.environ.get("INFERENCE_DEVICE1", "cuda:1").strip() or "cuda:1"
        split_index = num_hidden_layers // 2

        device_map = {
            "model.visual": device0,
            "model.language_model.embed_tokens": device0,
            "model.language_model.rotary_emb": device0,
            "model.language_model.norm": device1,
            "lm_head": device1,
        }
        for layer_index in range(num_hidden_layers):
            layer_device = device0 if layer_index < split_index else device1
            device_map[f"model.language_model.layers.{layer_index}"] = layer_device
        return device_map
    if value.lower() in {"none", "manual"}:
        return None
    if value.lower() == "cpu" or value.lower().startswith("cuda:"):
        return {"": value}
    return value


def _resolve_fallback_device() -> str:
    value = os.environ.get("INFERENCE_MODEL_DEVICE", "").strip()
    if value:
        return value
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


# =====================================================================
# Prompts (must match prepare_dataset_swift.py exactly)
# =====================================================================
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
- bbox 应覆盖标题栏完整外框

1.2 注释 (Notes)
- 通常位于图纸左下角或标题栏附近
- 包含加工要求、表面处理、公差标准等文字说明
- 通常以编号列表形式出现（如 1. 2. 3. ... 或圆圈编号）
- bbox 必须覆盖整个注释块的完整区域；
  不得按行拆分；包含编号、所有文字行与紧邻的辅助标识

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
- 与正视图轮廓相似，但内容上左右镜像（绕铅直轴旋转 180°）
- 通常出现在左视图或右视图的旁边
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
- 通常带折弯线、折弯标识或加工标识
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
5. 罕见类别（剖视图、详图、辅助视图、后视图）出现时必须独立检测；不得遗漏。

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
3. 螺纹孔识别优先：尺寸含 M 归为螺纹孔，而非圆孔。
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


# =====================================================================
# Model loading
# =====================================================================
def load_model(model_path: str, adapter_path: str, load_in_4bit: bool = False):
    from transformers import AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel

    try:
        from transformers import AutoModelForImageTextToText as _ModelCls
    except ImportError:
        try:
            from transformers import AutoModelForVision2Seq as _ModelCls
        except ImportError:
            from transformers import AutoModelForCausalLM as _ModelCls

    logger.info(f"Loading processor: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    logger.info(f"Loading base model: {model_path}  (4-bit={load_in_4bit})")

    os.environ.setdefault("INFERENCE_MODEL_PATH", model_path)
    device_map = _resolve_device_map()

    model_kwargs = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if device_map is not None:
        model_kwargs["device_map"] = device_map

    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Flash Attention 2 enabled")
    except ImportError:
        model_kwargs["attn_implementation"] = "sdpa"

    model = _ModelCls.from_pretrained(model_path, **model_kwargs)

    if adapter_path and adapter_path.lower() not in {"none", "null", ""}:
        logger.info(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        logger.info("No LoRA adapter specified — running base model")

    if device_map is None:
        target_device = _resolve_fallback_device()
        logger.info(f"Moving model to device: {target_device}")
        model = model.to(target_device)

    model.eval()

    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        logger.info(f"  GPU {i} VRAM: {alloc:.2f} GB")

    return model, processor


# =====================================================================
# Multi-turn inference
# =====================================================================
def _generate(model, processor, messages, max_new_tokens):
    try:
        from qwen_vl_utils import process_vision_info
        _use_qwen_utils = True
    except ImportError:
        _use_qwen_utils = False

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )

    if _use_qwen_utils:
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs if video_inputs else None,
            return_tensors="pt",
        )
    else:
        from PIL import Image as PILImage
        # Pull image paths out of the messages
        image_paths = []
        for m in messages:
            if isinstance(m.get("content"), list):
                for c in m["content"]:
                    if c.get("type") == "image":
                        path = c.get("image", "").replace("file://", "")
                        if path:
                            image_paths.append(path)
        imgs = [PILImage.open(p).convert("RGB") for p in image_paths]
        inputs = processor(text=[text], images=imgs if imgs else None, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_ids, skip_special_tokens=True)


def run_inference_two_turn(model, processor, image_path, max_new_tokens):
    image_path = Path(image_path)

    # Turn 1: views
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{str(image_path)}"},
                {"type": "text", "text": STEP1_USER_PROMPT},
            ],
        },
    ]
    raw_views = _generate(model, processor, messages, max_new_tokens)

    # Turn 2: features (continues the conversation)
    messages.append({"role": "assistant", "content": raw_views})
    messages.append({"role": "user", "content": STEP2_USER_PROMPT})
    raw_features = _generate(model, processor, messages, max_new_tokens)

    return raw_views, raw_features


# =====================================================================
# Output parsing
# =====================================================================
def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def parse_output(raw_text: str):
    text = _strip_think_tags(raw_text).strip()

    code_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if code_match:
        try:
            obj = json.loads(code_match.group(1))
            if isinstance(obj, list):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            features = obj.get("features", obj.get("value", []))
            if isinstance(features, list):
                converted = []
                for f in features:
                    item = dict(f)
                    if "label" in item and "category" not in item:
                        item["category"] = item.pop("label")
                    converted.append(item)
                return converted
    except (json.JSONDecodeError, ValueError):
        pass

    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        try:
            obj = json.loads(arr_match.group())
            if isinstance(obj, list):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    last_brace = text.rfind("}")
    if last_brace >= 0:
        candidate = text[:last_brace + 1]
        first_bracket = candidate.find("[")
        if first_bracket >= 0:
            candidate = candidate[first_bracket:]
            if not candidate.rstrip().endswith("]"):
                candidate = candidate.rstrip().rstrip(",") + "]"
            try:
                obj = json.loads(candidate)
                if isinstance(obj, list):
                    logger.info(f"Recovered {len(obj)} items from truncated JSON")
                    return obj
            except (json.JSONDecodeError, ValueError):
                pass

    logger.warning(f"JSON parse failed. raw output: {text[:200]!r}")
    return []


def _deduplicate(detections: list) -> list:
    seen = set()
    result = []
    for item in detections:
        key = (
            item.get("category"),
            item.get("size", ""),
            tuple(item.get("bbox", item.get("bbox_2d", []))),
        )
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


# =====================================================================
# Category snapping (B): if model emits an out-of-vocab category, snap to closest valid
# =====================================================================
VIEW_CATEGORIES = [
    "Title Block", "Notes",
    "Orthographic Projection - Front View",
    "Orthographic Projection - Top View",
    "Orthographic Projection - Bottom View",
    "Orthographic Projection - Left View",
    "Orthographic Projection - Right View",
    "Orthographic Projection - Rear View",
    "Isometric View", "Flat Pattern View",
    "Section View", "Detail View", "Auxiliary View",
]
FEATURE_CATEGORIES = [
    "Round Hole", "Rectangular Hole", "Threaded Hole", "Slotted Hole",
    "Round Hole Group", "Rectangular Hole Group", "Slotted Hole Group",
    "Fillet", "Bending", "Silver Plating",
]


def _snap_category(cat: str, valid_cats: list, cutoff: float = 0.6):
    """If `cat` is in `valid_cats` return as-is. Otherwise return the closest match
    by difflib SequenceMatcher ratio if above `cutoff`, else return `cat` unchanged.
    Reports the snap so we can audit how often it fires."""
    if not isinstance(cat, str) or not cat:
        return cat, False
    if cat in valid_cats:
        return cat, False
    from difflib import get_close_matches
    matches = get_close_matches(cat, valid_cats, n=1, cutoff=cutoff)
    if matches:
        return matches[0], True
    return cat, False


def _normalize_items(items, default_size="", snap_to=None):
    """If `snap_to` is a list of valid categories, post-hoc snap any out-of-vocab
    category emissions to the closest valid match. Returns (out, snap_count)."""
    out = []
    snap_count = 0
    for f in items:
        cat = f.get("category", f.get("label", ""))
        if snap_to is not None:
            cat, snapped = _snap_category(cat, snap_to)
            if snapped:
                snap_count += 1
        out.append({
            "category": cat,
            "size":     f.get("size", default_size),
            "bbox":     f.get("bbox", f.get("bbox_2d", [])),
        })
    return out, snap_count


# =====================================================================
# Dataset loading
# =====================================================================
def load_image_paths_from_jsonl(jsonl_path: Path) -> list:
    paths = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            images = rec.get("images", [])
            if isinstance(images, list):
                for img in images:
                    if isinstance(img, str) and img:
                        paths.append(img)
    return paths


def resolve_image_path(stored_path: str, image_dir: Path) -> Path:
    p = Path(stored_path)
    if p.is_absolute() and p.exists():
        return p
    alt = image_dir / p.name
    if alt.exists():
        return alt
    return p


def load_checkpoint(checkpoint_path: Path) -> dict:
    done = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    done[item["dataitem_name"]] = item
                except (json.JSONDecodeError, KeyError):
                    pass
        logger.info(f"Resumed: {len(done)} images already processed from {checkpoint_path}")
    return done


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-27B views + 7-feature multi-turn inference (improved)")
    parser.add_argument("--model_path",    type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--adapter_path",  type=str, default="/workspace/output/swift_27b_788_view_7feats_improve/final_adapter")
    parser.add_argument("--test_jsonl",    type=str, default="/workspace/data/test_view_7feats_improve.jsonl")
    parser.add_argument("--image_dir",     type=str, default="/workspace/data/simens_7feats")
    parser.add_argument("--output_file",   type=str, default="/workspace/output/swift_27b_788_view_7feats_improve_results.json")
    parser.add_argument("--max_new_tokens",type=int, default=3072,
                        help="Per-turn cap. Total budget across both turns is 2x this.")
    parser.add_argument("--save_every",    type=int, default=20)
    parser.add_argument("--load_in_4bit",  action="store_true")
    parser.add_argument("--snap_categories", type=lambda s: s.lower() in {"1","true","yes","on"},
                        default=True,
                        help="Post-hoc snap out-of-vocab categories to closest valid (default: True).")
    args = parser.parse_args()
    view_snap = VIEW_CATEGORIES if args.snap_categories else None
    feat_snap = FEATURE_CATEGORIES if args.snap_categories else None

    test_jsonl = Path(args.test_jsonl)
    image_dir = Path(args.image_dir)
    output_file = Path(args.output_file)
    checkpoint_path = output_file.with_suffix(".checkpoint.jsonl")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    _adapter_p = Path(args.adapter_path) if args.adapter_path else Path("base")
    _LEAF_DIRS = {"final_adapter", "best_adapter", "adapter"}
    if _adapter_p.name in _LEAF_DIRS:
        _run_tag = _adapter_p.parent.name
    else:
        _run_tag = _adapter_p.name
    if not _run_tag or _run_tag in ("output", "workspace"):
        _run_tag = "swift_27b_view_7feats_" + time.strftime("%Y%m%d_%H%M%S")

    _log_dir = Path(f"/workspace/logs/{_run_tag}")
    try:
        _log_dir.mkdir(parents=True, exist_ok=True)
        _fh = logging.FileHandler(_log_dir / "inference.log", mode="a", encoding="utf-8")
        _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(_fh)
    except Exception as e:
        logger.warning(f"File log handler failed: {e}")
    logger.info(f"Run tag: {_run_tag}")

    if not test_jsonl.exists():
        logger.error(f"Test JSONL not found: {test_jsonl}")
        return
    stored_paths = load_image_paths_from_jsonl(test_jsonl)
    logger.info(f"Loaded {len(stored_paths)} images from {test_jsonl}")

    resolved = []
    missing = 0
    for sp in stored_paths:
        p = resolve_image_path(sp, image_dir)
        if p.exists():
            resolved.append(p)
        else:
            missing += 1
    if missing:
        logger.warning(f"{missing} image paths could not be resolved; skipping them")
    if not resolved:
        logger.error("No images resolved; aborting.")
        return

    done_dict = load_checkpoint(checkpoint_path)
    remaining = [p for p in resolved if p.name not in done_dict]
    logger.info(f"Remaining: {len(remaining)} / {len(resolved)} images to process")

    if remaining:
        model, processor = load_model(args.model_path, args.adapter_path, args.load_in_4bit)

        ckpt_f = open(checkpoint_path, "a", encoding="utf-8")
        total = len(remaining)
        t_start = time.time()

        for i, img_path in enumerate(remaining):
            t0 = time.time()
            try:
                raw_views, raw_features = run_inference_two_turn(
                    model, processor, img_path, args.max_new_tokens
                )
                view_items, view_snaps = _normalize_items(
                    _deduplicate(parse_output(raw_views)), default_size="", snap_to=view_snap)
                feat_items, feat_snaps = _normalize_items(
                    _deduplicate(parse_output(raw_features)), snap_to=feat_snap)
                error = None
            except Exception as e:
                import traceback as _tb
                logger.error(f"[{i+1}/{total}] ERROR on {img_path.name}: {e}\n{_tb.format_exc()}")
                raw_views = ""
                raw_features = ""
                view_items = []
                feat_items = []
                view_snaps = 0
                feat_snaps = 0
                error = str(e)

            elapsed = time.time() - t0
            total_done = i + 1
            avg_time = (time.time() - t_start) / total_done
            eta_sec = avg_time * (total - total_done)
            eta_str = f"{int(eta_sec // 3600)}h{int((eta_sec % 3600) // 60)}m"

            n_v = len(view_items)
            n_f = len(feat_items)
            snap_note = f" snap={view_snaps}+{feat_snaps}" if (view_snaps or feat_snaps) else ""
            logger.info(
                f"[{total_done}/{total}] {img_path.name} | "
                f"{n_v} views + {n_f} features{snap_note} | "
                f"{elapsed:.1f}s | ETA {eta_str}"
            )

            combined = view_items + feat_items
            record = {
                "dataitem_name": img_path.name,
                "result": combined,
                "result_views": view_items,
                "result_features": feat_items,
                "raw_views": raw_views,
                "raw_features": raw_features,
            }
            if view_snaps or feat_snaps:
                record["_snapped_categories"] = {"views": view_snaps, "features": feat_snaps}
            if error:
                record["_error"] = error
            if (not view_items and raw_views) or (not feat_items and raw_features):
                record["_parse_warn"] = True

            done_dict[img_path.name] = record

            ckpt_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            ckpt_f.flush()

            if total_done % args.save_every == 0:
                _save_final(done_dict, resolved, output_file)
                logger.info(f"  Progress saved -> {output_file}")

        ckpt_f.close()
        logger.info(f"Inference complete. Processed {total} images.")

    _save_final(done_dict, resolved, output_file)
    logger.info(f"Final output saved -> {output_file}")
    logger.info(f"Total records: {len(done_dict)}")

    all_records = list(done_dict.values())
    total_views = sum(len(r.get("result_views", [])) for r in all_records)
    total_feats = sum(len(r.get("result_features", [])) for r in all_records)
    errors = sum(1 for r in all_records if "_error" in r)
    parse_warns = sum(1 for r in all_records if r.get("_parse_warn"))
    zero_total = sum(
        1 for r in all_records
        if not r.get("result_views") and not r.get("result_features")
        and "_error" not in r
    )

    logger.info(f"  Total view detections:    {total_views}")
    logger.info(f"  Total feature detections: {total_feats}")
    logger.info(f"  Errors:                   {errors}")
    logger.info(f"  Parse warnings:           {parse_warns}")
    logger.info(f"  Empty (both turns):       {zero_total}")


def _save_final(done_dict, all_images, output_file):
    ordered = []
    seen = set()
    for img_path in all_images:
        rec = done_dict.get(img_path.name)
        if rec is not None:
            ordered.append(rec)
            seen.add(img_path.name)
    for name, rec in done_dict.items():
        if name not in seen:
            ordered.append(rec)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
