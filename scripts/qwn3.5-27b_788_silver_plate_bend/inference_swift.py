"""
inference_swift.py — Qwen3.5-27B inference for Siemens 7-feature detection
(silver plating / bending, revised prompt v3).

7 categories: Round Hole, Rectangular Hole, Threaded Hole, Slotted Hole,
              Fillet, Bending, Silver Plating.

Loads base model + LoRA adapter from training, runs on the images listed
in a JSONL split file (default: test_7feats.jsonl) and writes one JSON
record per image. Writes a resumable checkpoint file alongside the output.
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
# Device map resolution (copied from reference inference_swift.py)
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
# Prompts (match prepare_dataset_swift.py)
# =====================================================================
DEFAULT_SYSTEM_PROMPT = """任务：
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

DETECTION_PROMPT = "请分析这张工程图纸，识别并提取所有孔类、圆角、折弯以及镀银特征，返回 JSON 列表。"


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
# Single image inference
# =====================================================================
def run_inference_single(model, processor, image_path, max_new_tokens: int) -> str:
    try:
        from qwen_vl_utils import process_vision_info
        _use_qwen_utils = True
    except ImportError:
        _use_qwen_utils = False

    image_path = Path(image_path)
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{str(image_path)}"},
                {"type": "text", "text": DETECTION_PROMPT},
            ],
        },
    ]

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
        img = PILImage.open(image_path).convert("RGB")
        inputs = processor(text=[text], images=[img], return_tensors="pt")

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
# Dataset loading
# =====================================================================
def load_image_paths_from_jsonl(jsonl_path: Path) -> list:
    """Load image paths (absolute) from a swift JSONL file."""
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
    """Prefer the path stored in JSONL; fall back to looking the basename up in image_dir."""
    p = Path(stored_path)
    if p.is_absolute() and p.exists():
        return p
    alt = image_dir / p.name
    if alt.exists():
        return alt
    return p  # may not exist; error handled upstream


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
# Batch inference main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-27B Siemens 7-feature inference")
    parser.add_argument("--model_path",    type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--adapter_path",  type=str, default="/workspace/output/swift_27b_7feats_silver_bend/final_adapter")
    parser.add_argument("--test_jsonl",    type=str, default="/workspace/data/test_7feats.jsonl",
                        help="JSONL produced by prepare_dataset_swift.py, whose 'images' entries drive inference.")
    parser.add_argument("--image_dir",     type=str, default="/workspace/data/simens_7feats",
                        help="Fallback image directory (used only if JSONL path doesn't resolve).")
    parser.add_argument("--output_file",   type=str, default="/workspace/output/swift_27b_7feats_silver_bend_results.json")
    parser.add_argument("--max_new_tokens",type=int, default=4096)
    parser.add_argument("--save_every",    type=int, default=20)
    parser.add_argument("--load_in_4bit",  action="store_true")
    args = parser.parse_args()

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
        _run_tag = "swift_27b_7feats_" + time.strftime("%Y%m%d_%H%M%S")

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
                raw_text = run_inference_single(model, processor, img_path, args.max_new_tokens)
                parsed = _deduplicate(parse_output(raw_text))
                error = None
            except Exception as e:
                logger.error(f"[{i+1}/{total}] ERROR on {img_path.name}: {e}")
                raw_text = ""
                parsed = []
                error = str(e)

            elapsed = time.time() - t0
            total_done = i + 1
            avg_time = (time.time() - t_start) / total_done
            eta_sec = avg_time * (total - total_done)
            eta_str = f"{int(eta_sec // 3600)}h{int((eta_sec % 3600) // 60)}m"

            logger.info(
                f"[{total_done}/{total}] {img_path.name} | "
                f"{len(parsed) if isinstance(parsed, list) else 'err'} detections | "
                f"{elapsed:.1f}s | ETA {eta_str}"
            )

            normalized = [
                {
                    "category": f.get("category", f.get("label", "")),
                    "size":     f.get("size", ""),
                    "bbox":     f.get("bbox", f.get("bbox_2d", [])),
                }
                for f in parsed
            ]
            result_str = json.dumps(normalized, ensure_ascii=False, indent=2)

            record = {
                "dataitem_name": img_path.name,
                "result": [result_str],
            }
            if error:
                record["_error"] = error
            if not parsed and raw_text:
                record["_raw"] = raw_text

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

    def _count_detections(r):
        try:
            return len(json.loads(r["result"][0]))
        except Exception:
            return 0

    total_detections = sum(_count_detections(r) for r in all_records)
    errors = sum(1 for r in all_records if "_error" in r)
    parse_fails = sum(1 for r in all_records if "_raw" in r)
    zero_det = sum(1 for r in all_records if _count_detections(r) == 0 and "_raw" not in r and "_error" not in r)

    logger.info(f"  Total detections:         {total_detections}")
    logger.info(f"  Errors:                   {errors}")
    logger.info(f"  Parse failures:           {parse_fails}")
    logger.info(f"  Images with 0 detections: {zero_det}")


def _save_final(done_dict, all_images, output_file):
    ordered = []
    for img_path in all_images:
        if img_path.name in done_dict:
            ordered.append(done_dict[img_path.name])
    known = {img.name for img in all_images}
    for name, rec in done_dict.items():
        if name not in known:
            ordered.append(rec)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
