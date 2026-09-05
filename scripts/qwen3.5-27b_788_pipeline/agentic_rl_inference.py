"""
Agentic RL inference — multi-pass self-refinement for drawing feature detection.

Pipeline:
  Pass 1 (standard):  Turn 1 → detect views, Turn 2 → detect features
  Pass 2 (critique):  Model reviews its own output, flags issues
  Pass 3 (refine):    Model corrects flagged issues, outputs final result

The self-refinement loop enables the model to catch and fix its own mistakes,
which is the core mechanism for agentic RL training.

Usage:
  python agentic_rl_inference.py \
      --adapter_path /workspace/output/swift_27b_788_pipeline/v0-*/swa_adapter \
      --image_path /workspace/data/simens_7feats/001_example.png

  # Batch mode for generating training data:
  python agentic_rl_inference.py \
      --adapter_path <path> \
      --test_jsonl /workspace/data/test_view_7feats_pipeline.jsonl \
      --image_dir /workspace/data/simens_7feats \
      --output_file results_agentic.json
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
# Prompts (from prepare_dataset_swift.py — improved variant)
# =====================================================================
SYSTEM_PROMPT = """你是一个专业的工业图纸分析助手。请按两步分析工程图纸：
第一步：识别图纸中所有视图区域和文档元素；
第二步：检测图纸中的所有结构特征（孔、圆角、折弯、镀银），提取类别、尺寸和位置。
每一步仅返回 JSON 数组，不附加任何解释。"""

# Standard prompts (imported from inference_swift.py in practice; inlined for standalone use)
# In production, these are loaded from the inference module to stay in sync.
# For the agentic RL script, we only need the critique and refinement prompts.

CRITIQUE_PROMPT = """请仔细审查你上一步输出的特征检测结果。

对照原始图纸，逐项检查以下问题：

1. **遗漏检测**：图纸中是否存在你未检测到的特征？
   - 检查所有视图区域，特别是侧视图、剖视图、展开视图
   - 小尺寸特征（如小圆角、小孔）是否被遗漏？

2. **错误分类**：是否有特征被错误分类？
   - 螺纹孔 vs 圆孔：尺寸以 M 开头的是螺纹孔
   - 腰孔 vs 矩形孔：两端半圆弧的是腰孔
   - 圆角 vs 孔：圆弧连接两条直线的是圆角

3. **定位偏差**：bbox 是否准确覆盖特征？
   - bbox 应完整包含特征，不应截断
   - 特征应在其所在视图的 bbox 内

4. **尺寸错误**：size 字段是否正确？
   - 尺寸应直接复制图纸标注
   - 缺失的尺寸应从附近标注或组标注继承

5. **组与单实例**：是否同时检测了单实例和组？
   - 当多个相同尺寸的同类孔有统一 N x 尺寸标注时，应同时输出单实例和组

请以 JSON 格式返回审查结果：
{
  "issues": [
    {"type": "missing|misclassified|bbox_shift|size_error|missing_group", "detail": "<描述>", "severity": "high|medium|low"}
  ],
  "overall_assessment": "<一句话总结>"
}

仅返回 JSON，不附加任何解释。"""

REFINE_PROMPT = """基于你的审查结果，请修正并输出最终的特征检测列表。

修正规则：
1. 补充所有遗漏的特征
2. 修正分类错误的特征
3. 调整定位有偏差的 bbox
4. 修正尺寸错误
5. 补充缺失的组标注
6. 审查中未发现问题的特征保持原样

严格返回 JSON 列表（与初始检测格式相同）：

[
  {
    "category": "<英文类别>",
    "bbox": [x_min, y_min, x_max, y_max],
    "size": "<尺寸参数字符串>"
  }
]

仅返回 JSON 结果。不得附加任何解释、说明或额外文本。"""


# =====================================================================
# Device map
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

    model_kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True)
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
        model = model.to(_resolve_fallback_device())

    model.eval()
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        logger.info(f"  GPU {i} VRAM: {alloc:.2f} GB")

    return model, processor


# =====================================================================
# Generation helper
# =====================================================================
def _generate(model, processor, messages, max_new_tokens, temperature=0.0, top_p=None):
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
            text=[text], images=image_inputs,
            videos=video_inputs if video_inputs else None,
            return_tensors="pt",
        )
    else:
        from PIL import Image as PILImage
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

    do_sample = temperature > 0
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=1.05,
        )
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_ids, skip_special_tokens=True)


# =====================================================================
# Output parsing
# =====================================================================
def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def parse_json_output(raw_text: str):
    text = _strip_think_tags(raw_text).strip()

    code_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if code_match:
        try:
            obj = json.loads(code_match.group(1))
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return obj
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
                    return obj
            except (json.JSONDecodeError, ValueError):
                pass

    logger.warning(f"JSON parse failed. raw output: {text[:200]!r}")
    return []


# =====================================================================
# Load prompts from inference_swift (keep in sync)
# =====================================================================
def _load_step_prompts():
    """Load STEP1 and STEP2 prompts from the pipeline's inference script."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from inference_swift import STEP1_USER_PROMPT, STEP2_USER_PROMPT
        return STEP1_USER_PROMPT, STEP2_USER_PROMPT
    except ImportError:
        logger.warning("Could not import prompts from inference_swift; using defaults")
        return "检测所有视图区域。", "检测所有结构特征。"


# =====================================================================
# Agentic multi-pass inference
# =====================================================================
def run_agentic_inference(model, processor, image_path, max_new_tokens,
                          temperature=0.0, enable_critique=True):
    """Run the full agentic loop: detect → critique → refine.

    Returns:
        dict with initial_views, initial_features, critique, refined_features,
        and raw text for each pass.
    """
    image_path = Path(image_path)
    STEP1_USER_PROMPT, STEP2_USER_PROMPT = _load_step_prompts()

    # ---- Pass 1: Standard two-turn detection ----
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
    raw_views = _generate(model, processor, messages, max_new_tokens, temperature)

    messages.append({"role": "assistant", "content": raw_views})
    messages.append({"role": "user", "content": STEP2_USER_PROMPT})
    raw_features = _generate(model, processor, messages, max_new_tokens, temperature)

    initial_views = parse_json_output(raw_views)
    initial_features = parse_json_output(raw_features)

    result = {
        "initial_views": initial_views,
        "initial_features": initial_features,
        "raw_views": raw_views,
        "raw_features": raw_features,
        "critique": None,
        "refined_features": initial_features,
        "raw_critique": "",
        "raw_refined": raw_features,
        "refinement_applied": False,
    }

    if not enable_critique:
        return result

    # ---- Pass 2: Self-critique ----
    messages.append({"role": "assistant", "content": raw_features})
    messages.append({"role": "user", "content": CRITIQUE_PROMPT})
    raw_critique = _generate(model, processor, messages, max_new_tokens // 2, temperature)
    critique_data = parse_json_output(raw_critique)
    result["critique"] = critique_data if isinstance(critique_data, dict) else {}
    result["raw_critique"] = raw_critique

    # ---- Pass 3: Refinement ----
    messages.append({"role": "assistant", "content": raw_critique})
    messages.append({"role": "user", "content": REFINE_PROMPT})
    raw_refined = _generate(model, processor, messages, max_new_tokens, temperature)
    refined_features = parse_json_output(raw_refined)
    result["refined_features"] = refined_features
    result["raw_refined"] = raw_refined

    if refined_features and refined_features != initial_features:
        result["refinement_applied"] = True

    return result


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


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Agentic RL multi-pass inference for drawing feature detection")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--adapter_path", type=str,
                        default="/workspace/output/swift_27b_788_pipeline/final_adapter")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Single image for ad-hoc inference")
    parser.add_argument("--test_jsonl", type=str, default=None,
                        help="JSONL with image list for batch inference")
    parser.add_argument("--image_dir", type=str, default="/workspace/data/simens_7feats")
    parser.add_argument("--output_file", type=str,
                        default="/workspace/output/agentic_rl_results.json")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--no_critique", action="store_true",
                        help="Disable critique/refine (run standard inference only)")
    args = parser.parse_args()

    # Single image mode
    if args.image_path:
        model, processor = load_model(args.model_path, args.adapter_path, args.load_in_4bit)
        result = run_agentic_inference(
            model, processor, args.image_path, args.max_new_tokens,
            args.temperature, not args.no_critique
        )
        print(json.dumps({
            "image": Path(args.image_path).name,
            "n_initial_features": len(result["initial_features"]),
            "n_refined_features": len(result.get("refined_features", [])),
            "refinement_applied": result.get("refinement_applied", False),
            "critique_issues": len(result.get("critique", {}).get("issues", [])),
            "raw_views": result["raw_views"][:200] + "...",
            "raw_features": result["raw_features"][:200] + "...",
            "raw_critique": result.get("raw_critique", "")[:200] + "...",
            "raw_refined": result.get("raw_refined", "")[:200] + "...",
        }, ensure_ascii=False, indent=2))
        return

    # Batch mode
    if not args.test_jsonl:
        logger.error("Either --image_path or --test_jsonl required")
        return

    test_jsonl = Path(args.test_jsonl)
    image_dir = Path(args.image_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    stored_paths = load_image_paths_from_jsonl(test_jsonl)
    logger.info(f"Loaded {len(stored_paths)} images from {test_jsonl}")

    resolved = []
    for sp in stored_paths:
        p = resolve_image_path(sp, image_dir)
        if p.exists():
            resolved.append(p)

    if not resolved:
        logger.error("No images resolved")
        return

    model, processor = load_model(args.model_path, args.adapter_path, args.load_in_4bit)

    results = []
    t_start = time.time()
    enable_critique = not args.no_critique

    for i, img_path in enumerate(resolved):
        t0 = time.time()
        try:
            result = run_agentic_inference(
                model, processor, img_path, args.max_new_tokens,
                args.temperature, enable_critique
            )
        except Exception as e:
            logger.error(f"[{i+1}/{len(resolved)}] ERROR on {img_path.name}: {e}")
            result = {
                "initial_views": [], "initial_features": [],
                "refined_features": [], "refinement_applied": False,
                "_error": str(e),
            }

        elapsed = time.time() - t0
        total_done = i + 1
        avg_time = (time.time() - t_start) / total_done
        eta_sec = avg_time * (len(resolved) - total_done)

        record = {
            "dataitem_name": img_path.name,
            "n_initial": len(result["initial_features"]),
            "n_refined": len(result.get("refined_features", [])),
            "refinement_applied": result.get("refinement_applied", False),
            "initial_features": result["initial_features"],
            "refined_features": result["refined_features"],
            "raw_views": result.get("raw_views", ""),
            "raw_features": result.get("raw_features", ""),
            "raw_critique": result.get("raw_critique", ""),
            "raw_refined": result.get("raw_refined", ""),
        }
        if "_error" in result:
            record["_error"] = result["_error"]

        results.append(record)

        logger.info(
            f"[{total_done}/{len(resolved)}] {img_path.name} | "
            f"init={len(result['initial_features'])} → "
            f"refined={len(result.get('refined_features', []))} | "
            f"delta={len(result.get('refined_features', [])) - len(result['initial_features'])} | "
            f"{elapsed:.1f}s | ETA {int(eta_sec//60)}m{int(eta_sec%60)}s"
        )

        if total_done % args.save_every == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary
    total_initial = sum(r["n_initial"] for r in results)
    total_refined = sum(r["n_refined"] for r in results)
    n_refined = sum(1 for r in results if r.get("refinement_applied"))
    logger.info(f"Done. {len(results)} images → {output_file}")
    logger.info(f"  Total initial features:  {total_initial}")
    logger.info(f"  Total refined features:  {total_refined}")
    logger.info(f"  Images with refinement:  {n_refined}/{len(results)}")


if __name__ == "__main__":
    main()
