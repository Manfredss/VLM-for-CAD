"""
inference_swift.py — Qwen3.5-27B ms-swift 微调模型批量推理脚本

与 nautilus_scripts_30b_swift/inference_swift.py 的区别：
  - 模型改为 Qwen/Qwen3.5-27B（原生多模态，无 -VL 后缀）
  - enable_thinking=False（Qwen3.5 默认开启 thinking，推理时需关闭）
  - 自动剥离 <think>...</think> 标签（防御性处理）
  - 需要 transformers>=5.2.0

特性：
  - 断点续传（自动跳过已处理的图片）
    - 仅对 train/val 数据集中的图片执行推理（自动过滤未标注图片）
  - 每 50 张自动保存进度
  - 解析失败时保留原始文本，不中断流程
  - 进度条 + 速度估算

用法：
  python3 /workspace/scripts_27b_swift/inference_swift.py \
      --model_path   Qwen/Qwen3.5-27B \
      --adapter_path /workspace/output/swift_27b/final_adapter \
      --image_dir    /workspace/data/IM_D03_PT_5K \
      --output_file  /workspace/output/swift_27b_inference_results.json \
      --train_dataset /workspace/data/train_augmented.jsonl \
      --val_dataset   /workspace/data/val_augmented.jsonl \
      [--max_new_tokens 4096] \
      [--load_in_4bit]
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
# Prompts — 与 prepare_dataset_swift.py 训练时完全一致（bbox 版本）
# =====================================================================
DEFAULT_SYSTEM_PROMPT = (
    "你是一个专业的工业图纸分析助手。你的任务是识别和分析工程图纸中的工件特征，"
    "包括但不限于：螺纹孔、圆角、矩形孔、圆孔、长圆孔、倒角、槽等特征。"
    "对于每种特征，请输出其类型、规格尺寸（如M8、R3）以及在图纸中的位置坐标。"
    "请根据图纸内容给出准确、完整的分析结果。"
)

# 文字在前（以"图纸："结尾），图片在后 — 与训练时格式完全一致
DETECTION_PROMPT = (
    "你是一个专门用于分析工程图纸的AI视觉系统。\n\n"
    "任务：\n对提供的工程图纸图像进行目标检测。\n\n"
    "对每个检测到的对象，提取：\n"
    "- category（类别）\n"
    "- size（规格尺寸，如 M8、R3、ø10）\n"
    "- bbox（归一化坐标边界框，0-1000 范围）\n\n"
    "需要提取的类别有：\n"
    "- Threaded Hole, Threaded Hole Group\n"
    "- Fillet, Fillet Group\n"
    "- Round Hole, Round Hole Group\n"
    "- Slotted Hole, Slotted Hole Group\n"
    "- Rectangular Hole, Rectangular Hole Group\n"
    "其中，Group 仅为描述性特征，Group 内包含的特征仍需单独列举\n\n"
    "边界框格式（整数，归一化到 0-1000）：\n[x_min, y_min, x_max, y_max]\n\n"
    "规则：\n"
    "- 检测所有可见的符号、标注、尺寸、表格、标题栏以及几何元素。\n"
    "- 使用归一化坐标（0-1000 范围）。\n"
    "- bbox 必须由四个整数值组成，且满足 x_min < x_max 且 y_min < y_max。\n"
    "- 不要输出解释。\n"
    "- 仅输出严格 json。\n"
    "- 不要添加任何额外字段。\n"
    "- 若未检测到对象，返回空数组。\n"
    "- 输出必须严格遵守以下格式。\n\n"
    "输出示例（禁止修改）\n"
    "```json\n"
    "[\n"
    "  {\n"
    '    "category": "Round Hole",\n'
    '    "size": "ø10",\n'
    '    "bbox": [x_min, y_min, x_max, y_max]\n'
    "  },\n"
    "  ...\n"
    "]\n"
    "```\n"
    "图纸："
)


# =====================================================================
# 模型加载
# =====================================================================
def load_model(model_path: str, adapter_path: str, load_in_4bit: bool):
    from transformers import AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel

    # 选择模型类
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

    model_kwargs = dict(
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
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

    # 检测 Flash Attention
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Flash Attention 2 enabled")
    except ImportError:
        model_kwargs["attn_implementation"] = "sdpa"

    model = _ModelCls.from_pretrained(model_path, **model_kwargs)

    logger.info(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        logger.info(f"  GPU {i} VRAM: {alloc:.2f} GB")

    return model, processor


# =====================================================================
# 单张图片推理
# =====================================================================
def run_inference_single(model, processor, image_path: Path, max_new_tokens: int) -> str:
    """对单张图片执行推理，返回模型原始输出文本。"""
    try:
        from qwen_vl_utils import process_vision_info
        _use_qwen_utils = True
    except ImportError:
        _use_qwen_utils = False

    messages = [
        {
            "role": "system",
            "content": DEFAULT_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": DETECTION_PROMPT},
                {"type": "image", "image": f"file://{str(image_path)}"},
            ],
        },
    ]

    # Qwen3.5 默认开启 thinking mode，推理时需关闭
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
# 解析模型输出
# =====================================================================
def _strip_think_tags(text: str) -> str:
    """剥离 Qwen3.5 的 <think>...</think> 标签（防御性处理）"""
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def parse_output(raw_text: str):
    """
    解析模型输出为 Python list。
    支持以下格式：
      1. markdown json 代码块: ```json [...] ```
      2. 直接 JSON array
      3. 嵌套 JSON
    """
    # 先剥离 think 标签
    text = _strip_think_tags(raw_text).strip()

    # 1. 提取 markdown 代码块中的 JSON（swift 训练格式输出）
    code_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if code_match:
        try:
            obj = json.loads(code_match.group(1))
            if isinstance(obj, list):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # 2. 直接尝试 JSON 解析
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

    # 3. 提取第一个 [...] 片段
    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        try:
            obj = json.loads(arr_match.group())
            if isinstance(obj, list):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # 4. 向后扫描恢复（截断的 JSON）
    last_brace = text.rfind("}")
    if last_brace >= 0:
        candidate = text[:last_brace + 1]
        # 找到第一个 [
        first_bracket = candidate.find("[")
        if first_bracket >= 0:
            candidate = candidate[first_bracket:]
            # 尝试补全截断的数组
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
    """去除完全重复的检测结果（category + bbox 完全相同）。"""
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
# 断点续传：加载已有进度
# =====================================================================
def load_checkpoint(checkpoint_path: Path) -> dict:
    """读取 JSONL checkpoint，返回 {dataitem_name: result} dict。"""
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


def load_dataset_image_names(*dataset_paths: str) -> set:
    """
    从一个或多个数据集文件（JSON array 或 JSONL）提取图片文件名（basename）。
    兼容两种来源：
      1) 顶层 images 字段（swift 数据常见）
      2) messages 中 user.content(list) 的 image 字段
    """
    names = set()
    for path_str in dataset_paths:
        if not path_str:
            continue

        dataset_path = Path(path_str)
        if not dataset_path.exists():
            logger.warning(f"Dataset file not found, skipping: {dataset_path}")
            continue

        items = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            if dataset_path.suffix == ".jsonl":
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            else:
                try:
                    items = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse dataset file: {dataset_path}")
                    continue

        before = len(names)
        for item in items:
            images = item.get("images")
            if isinstance(images, list):
                for img in images:
                    if isinstance(img, str) and img:
                        names.add(Path(img).name)

            for msg in item.get("messages", []):
                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        img_path = part.get("image", "")
                        if img_path:
                            names.add(Path(img_path).name)

        logger.info(
            f"Loaded {len(names) - before} image names from {dataset_path} "
            f"(total unique: {len(names)})"
        )

    return names


# =====================================================================
# 主流程
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-27B swift-trained model batch inference")
    parser.add_argument("--model_path",    type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--adapter_path",  type=str, default="/workspace/output/swift_27b/final_adapter")
    parser.add_argument("--image_dir",     type=str, default="/workspace/data/IM_D03_PT_5K")
    parser.add_argument("--output_file",   type=str, default="/workspace/output/swift_27b_inference_results.json")
    parser.add_argument("--max_new_tokens",type=int, default=4096,
                        help="最大生成 token 数")
    parser.add_argument("--save_every",    type=int, default=50,
                        help="每隔多少张保存一次进度")
    parser.add_argument("--batch_size",    type=int, default=700,
                        help="每处理多少张额外生成一个独立批次 JSON")
    parser.add_argument("--load_in_4bit",  action="store_true",
                        help="用 4-bit NF4 量化加载")
    parser.add_argument("--image_suffix",  type=str, default=".png")
    parser.add_argument("--train_dataset", type=str, default="/workspace/data/train_augmented.jsonl",
                        help="train JSON/JSONL 路径，仅对其中出现的图片做推理")
    parser.add_argument("--val_dataset",   type=str, default="/workspace/data/val_augmented.jsonl",
                        help="val JSON/JSONL 路径，仅对其中出现的图片做推理")
    args = parser.parse_args()

    image_dir   = Path(args.image_dir)
    output_file = Path(args.output_file)
    checkpoint_path = output_file.with_suffix(".checkpoint.jsonl")

    # ---- 文件日志 ----
    _adapter_p = Path(args.adapter_path)
    _LEAF_DIRS = {"final_adapter", "best_adapter", "adapter"}
    if _adapter_p.name in _LEAF_DIRS:
        _run_tag = _adapter_p.parent.name
    else:
        _run_tag = _adapter_p.name
    if not _run_tag or _run_tag in ("output", "workspace"):
        _run_tag = "swift_27b_inference_" + time.strftime("%Y%m%d_%H%M%S")

    _log_dir = Path(f"/workspace/logs/{_run_tag}")
    _log_dir.mkdir(parents=True, exist_ok=True)
    _fh = logging.FileHandler(_log_dir / "inference.log", mode="a", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(_fh)
    logger.info(f"Run tag:  {_run_tag}")
    logger.info(f"Log file: {_log_dir / 'inference.log'}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 收集所有图片
    all_images = sorted(image_dir.glob(f"*{args.image_suffix}"))
    logger.info(f"Found {len(all_images)} images in {image_dir}")

    # 数据集白名单过滤（强制）：仅推理 train/val 中出现过的图片，跳过未标注图片
    whitelist = load_dataset_image_names(args.train_dataset, args.val_dataset)
    if not whitelist:
        logger.error("Dataset whitelist is empty. Please check --train_dataset/--val_dataset paths.")
        return

    before_filter = len(all_images)
    all_images = [p for p in all_images if p.name in whitelist]
    logger.info(f"Dataset whitelist: {len(whitelist)} images")
    logger.info(
        f"After filtering: {len(all_images)} images to process "
        f"(skipped {before_filter - len(all_images)} non-train/val images)"
    )

    all_images_index = {p.name: idx for idx, p in enumerate(all_images)}

    if not all_images:
        logger.error("No images found. Check --image_dir path.")
        return

    # 加载已有进度
    done_dict = load_checkpoint(checkpoint_path)
    remaining = [p for p in all_images if p.name not in done_dict]
    logger.info(f"Remaining: {len(remaining)} images to process")

    if not remaining:
        logger.info("All images already processed. Merging to final output...")
    else:
        model, processor = load_model(args.model_path, args.adapter_path, args.load_in_4bit)

        ckpt_f = open(checkpoint_path, "a", encoding="utf-8")

        total    = len(remaining)
        t_start  = time.time()

        for i, img_path in enumerate(remaining):
            t0 = time.time()

            try:
                raw_text = run_inference_single(model, processor, img_path, args.max_new_tokens)
                parsed   = _deduplicate(parse_output(raw_text))
                error    = None
            except Exception as e:
                logger.error(f"[{i+1}/{total}] ERROR on {img_path.name}: {e}")
                raw_text = ""
                parsed   = []
                error    = str(e)

            elapsed    = time.time() - t0
            total_done = i + 1
            avg_time   = (time.time() - t_start) / total_done
            eta_sec    = avg_time * (total - total_done)
            eta_str    = f"{int(eta_sec // 3600)}h{int((eta_sec % 3600) // 60)}m"

            logger.info(
                f"[{total_done}/{total}] {img_path.name} | "
                f"{len(parsed) if isinstance(parsed, list) else 'err'} detections | "
                f"{elapsed:.1f}s | ETA {eta_str}"
            )

            # 统一字段顺序：category → size → bbox
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
                _save_final(done_dict, all_images, output_file)
                logger.info(f"  Progress saved → {output_file}")

            global_idx = all_images_index[img_path.name]
            if (global_idx + 1) % args.batch_size == 0:
                batch_num   = (global_idx + 1) // args.batch_size
                batch_start = global_idx - args.batch_size + 1
                batch_end   = global_idx + 1
                _save_batch(done_dict, all_images, batch_start, batch_end,
                            output_file, batch_num)

        ckpt_f.close()
        logger.info(f"\nInference complete. Processed {total} images.")

    # 补存批次
    n_full_batches = len(all_images) // args.batch_size
    for b in range(1, n_full_batches + 1):
        batch_start = (b - 1) * args.batch_size
        batch_end   = b * args.batch_size
        if all(img.name in done_dict for img in all_images[batch_start:batch_end]):
            _save_batch(done_dict, all_images, batch_start, batch_end, output_file, b)
    tail_start = n_full_batches * args.batch_size
    if tail_start < len(all_images) and len(done_dict) >= len(all_images):
        _save_batch(done_dict, all_images, tail_start, len(all_images),
                    output_file, n_full_batches + 1)

    # 合并最终输出
    _save_final(done_dict, all_images, output_file)
    logger.info(f"Final output saved → {output_file}")
    logger.info(f"Total records: {len(done_dict)}")

    # 统计
    all_records = list(done_dict.values())

    def _count_detections(r):
        try:
            return len(json.loads(r["result"][0]))
        except Exception:
            return 0

    total_detections = sum(_count_detections(r) for r in all_records)
    errors     = sum(1 for r in all_records if "_error" in r)
    parse_fails = sum(1 for r in all_records if "_raw" in r)
    zero_det   = sum(1 for r in all_records if _count_detections(r) == 0 and "_raw" not in r and "_error" not in r)

    logger.info(f"  Total detections:         {total_detections}")
    logger.info(f"  Errors:                   {errors}")
    logger.info(f"  Parse failures:           {parse_fails}")
    logger.info(f"  Images with 0 detections: {zero_det}")


def _save_batch(done_dict: dict, all_images: list,
                start_idx: int, end_idx: int,
                output_file: Path, batch_num: int):
    batch_file = output_file.parent / f"{output_file.stem}_batch_{batch_num:03d}.json"
    if batch_file.exists():
        return
    batch_records = [
        done_dict[img.name]
        for img in all_images[start_idx:end_idx]
        if img.name in done_dict
    ]
    if not batch_records:
        return
    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(batch_records, f, ensure_ascii=False, indent=2)
    logger.info(
        f"  >>> Batch {batch_num:03d} saved "
        f"(#{start_idx + 1}~#{end_idx}, {len(batch_records)} images) "
        f"→ {batch_file.name}"
    )


def _save_final(done_dict: dict, all_images: list, output_file: Path):
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
