#!/usr/bin/env python3
"""
Multi-step inference for Qwen3.5-27B engineering drawing analysis.

Pipeline:
  Step 1: Layout & view detection → JSON list of view/layout elements
  Step 2: Structural feature detection → JSON list of features with size

The model runs two inference passes per image:
  1. Generate step 1 output (views/layout)
  2. Append step 2 prompt + generate step 2 output (features)
  3. Merge both outputs into final result
"""

import argparse
import json
import logging
import os
import re
import time
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =====================================================================
# Prompts (must match training prompts exactly)
# =====================================================================
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


# =====================================================================
# Category sets (used by evaluate_checkpoints_benchmark.py)
# =====================================================================
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


# =====================================================================
# Device resolution
# =====================================================================
def _resolve_device_map():
    if torch.cuda.device_count() > 1:
        return "auto"
    if torch.cuda.is_available():
        return {"": 0}
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return {"": "mps"}
    return {"": "cpu"}


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
    device_map = _resolve_device_map()

    model_kwargs = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )

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
    except ImportError:
        model_kwargs["attn_implementation"] = "sdpa"

    model = _ModelCls.from_pretrained(model_path, **model_kwargs)

    logger.info(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, processor


# =====================================================================
# Inference helpers
# =====================================================================
def _generate(model, processor, messages, max_new_tokens):
    """Run a single generation given a message list."""
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
        # Extract image from messages
        img = None
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        img_path = part["image"].replace("file://", "")
                        img = PILImage.open(img_path).convert("RGB")
                        break
        inputs = processor(
            text=[text],
            images=[img] if img else None,
            return_tensors="pt",
        )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.05,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_ids, skip_special_tokens=True)


def run_multistep_inference(model, processor, image_path, max_new_tokens=2048):
    """
    Run 2-step inference on a single image.
    Returns (step1_features, step2_features, step1_raw, step2_raw).
    """
    image_path = Path(image_path)

    # Step 1: Layout & view detection
    messages_step1 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{str(image_path)}"},
                {"type": "text", "text": STEP1_USER_PROMPT},
            ],
        },
    ]

    step1_raw = _generate(model, processor, messages_step1, max_new_tokens)
    step1_features = parse_output(step1_raw)
    step1_features = _deduplicate(step1_features)

    # Step 2: Feature detection (with step 1 context)
    messages_step2 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{str(image_path)}"},
                {"type": "text", "text": STEP1_USER_PROMPT},
            ],
        },
        {"role": "assistant", "content": step1_raw},
        {"role": "user", "content": STEP2_USER_PROMPT},
    ]

    step2_raw = _generate(model, processor, messages_step2, max_new_tokens)
    step2_features = parse_output(step2_raw)
    step2_features = _deduplicate(step2_features)

    return step1_features, step2_features, step1_raw, step2_raw


# =====================================================================
# Output parsing
# =====================================================================
def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def parse_output(raw_text: str):
    text = _strip_think_tags(raw_text).strip()

    # 1. Markdown code block
    code_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if code_match:
        try:
            obj = json.loads(code_match.group(1))
            if isinstance(obj, list):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # 2. Direct JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. Extract [...] segment
    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        try:
            obj = json.loads(arr_match.group())
            if isinstance(obj, list):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # 4. Recover truncated JSON
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


def calculate_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter) if (aa + ab - inter) > 0 else 0.0


# =====================================================================
# Batch inference main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Multi-step batch inference")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default="/workspace/data/5k")
    parser.add_argument("--output_file", type=str, default="/workspace/output/results_multistep.json")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--benchmark_jsonl", type=str, default="",
                        help="If provided, run only on benchmark samples")
    parser.add_argument("--train_dataset", type=str, default="/workspace/data/train_multistep.jsonl")
    parser.add_argument("--val_dataset", type=str, default="/workspace/data/val_multistep.jsonl")
    args = parser.parse_args()

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine image list
    if args.benchmark_jsonl:
        # Run on benchmark samples
        image_paths = []
        with open(args.benchmark_jsonl) as f:
            for line in f:
                rec = json.loads(line.strip())
                image_paths.append(rec["images"][0])
        logger.info(f"Loaded {len(image_paths)} benchmark samples")
    else:
        # Run on all images from train+val datasets
        image_names = set()
        for ds_path in [args.train_dataset, args.val_dataset]:
            if not ds_path or not Path(ds_path).exists():
                continue
            with open(ds_path) as f:
                for line in f:
                    rec = json.loads(line.strip())
                    for img in rec.get("images", []):
                        image_names.add(Path(img).name)

        image_dir = Path(args.image_dir)
        image_paths = sorted([
            str(image_dir / name)
            for name in image_names
            if (image_dir / name).exists()
        ])
        logger.info(f"Found {len(image_paths)} images")

    if not image_paths:
        logger.error("No images found.")
        return

    model, processor = load_model(args.model_path, args.adapter_path, args.load_in_4bit)

    results = []
    t_start = time.time()

    for i, img_path in enumerate(image_paths, 1):
        img_name = Path(img_path).name
        t0 = time.time()

        try:
            step1, step2, step1_raw, step2_raw = run_multistep_inference(
                model, processor, img_path, args.max_new_tokens
            )
        except Exception as e:
            logger.error(f"[{i}/{len(image_paths)}] ERROR on {img_name}: {e}")
            step1, step2 = [], []

        # Normalize step1 output (add size="" for consistency)
        step1_norm = [
            {"category": f.get("category", ""), "size": "", "bbox": f.get("bbox", [])}
            for f in step1
        ]
        # Normalize step2 output
        step2_norm = [
            {
                "category": f.get("category", ""),
                "size": f.get("size", ""),
                "bbox": f.get("bbox", []),
            }
            for f in step2
        ]

        # Merge both steps
        merged = step1_norm + step2_norm

        elapsed = time.time() - t0
        avg = (time.time() - t_start) / i
        eta = avg * (len(image_paths) - i)

        logger.info(
            f"[{i}/{len(image_paths)}] {img_name} | "
            f"views={len(step1)} features={len(step2)} total={len(merged)} | "
            f"{elapsed:.1f}s | ETA {int(eta//60)}m"
        )

        results.append({
            "dataitem_name": img_name,
            "result": merged,
            "step1_count": len(step1),
            "step2_count": len(step2),
        })

        # Save periodically
        if i % 10 == 0 or i == len(image_paths):
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Done. {len(results)} images processed. Saved to {output_file}")


if __name__ == "__main__":
    main()
