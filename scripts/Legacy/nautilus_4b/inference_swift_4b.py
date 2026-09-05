"""
inference_swift_4b.py — Qwen3.5-4B ms-swift 微调模型批量推理脚本

直接从 test_v2.jsonl 读取原始 messages，保证 prompt 格式与训练完全一致。
- 断点续传（自动跳过已处理的图片）
- 每 10 张自动保存进度
- 实时输出进度
- enable_thinking=False（匹配 add_non_thinking_prefix=True 训练设置）

基于 nautilus_scripts_27b_swift/inference_swift.py 改造。
"""

import os
import re
import json
import time
import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# 训练时使用的 system prompt（来自 args.json）
DEFAULT_SYSTEM_PROMPT = (
    "你是一个专业的工业图纸分析助手。你的任务是识别和分析工程图纸中的工件特征，"
    "包括但不限于：螺纹孔、圆角、矩形孔、圆孔、长圆孔、倒角、槽等特征。"
    "对于每种特征，请输出其类型、规格尺寸（如M8、R3）以及在图纸中的位置坐标。"
    "请根据图纸内容给出准确、完整的分析结果。"
)


def load_model(model_path: str, adapter_path: str):
    from transformers import AutoProcessor
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

    logger.info(f"Loading base model: {model_path}")
    model_kwargs = dict(
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )

    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Flash Attention 2 enabled")
    except ImportError:
        model_kwargs["attn_implementation"] = "sdpa"
        logger.info("Using SDPA attention (flash_attn not available)")

    model = _ModelCls.from_pretrained(model_path, **model_kwargs)

    logger.info(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        logger.info(f"  GPU {i} VRAM: {alloc:.2f} GB")

    return model, processor


def run_inference_from_messages(model, processor, messages: list, max_new_tokens: int) -> str:
    """
    直接用 JSONL 中的 messages 做推理，保证格式与训练完全一致。
    messages 应该是 [{role: user, content: [...]}] 格式。
    """
    try:
        from qwen_vl_utils import process_vision_info
        _use_qwen_utils = True
    except ImportError:
        _use_qwen_utils = False

    # 加入 system prompt（训练时 swift 在模板层面注入的）
    full_messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
    ] + [msg for msg in messages if msg["role"] == "user"]

    # enable_thinking=False 匹配训练时的 add_non_thinking_prefix=True
    text = processor.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )

    if _use_qwen_utils:
        image_inputs, video_inputs = process_vision_info(full_messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs if video_inputs else None,
            return_tensors="pt",
        )
    else:
        from PIL import Image as PILImage
        # 从 messages 中提取图片路径
        images = []
        for msg in full_messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        img_path = part.get("image", "")
                        if img_path.startswith("file://"):
                            img_path = img_path[7:]
                        images.append(PILImage.open(img_path).convert("RGB"))
        inputs = processor(text=[text], images=images if images else None, return_tensors="pt")

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


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def parse_output(raw_text: str):
    text = _strip_think_tags(raw_text).strip()

    # 1. markdown json block
    code_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if code_match:
        try:
            obj = json.loads(code_match.group(1))
            if isinstance(obj, list):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # 2. direct JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. extract [...] fragment
    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        try:
            obj = json.loads(arr_match.group())
            if isinstance(obj, list):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # 4. recover truncated JSON
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
        logger.info(f"Resumed: {len(done)} images already processed")
    return done


def load_test_data(jsonl_path: str, image_dir: str = None) -> list:
    """
    从 JSONL 加载测试数据。
    返回 list of {messages, dataitem_name}。
    自动修复图片路径（相对路径 → 绝对路径）。
    """
    items = []
    aug_pat = re.compile(r'_(rot|flip|aug|crop|bright|contrast|noise|blur|scale|shift)\d*', re.IGNORECASE)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 跳过增强数据
            name = obj.get("dataitem_name", "")
            if aug_pat.search(name):
                continue

            # 修复 messages 中的图片路径
            messages = obj.get("messages", [])
            for msg in messages:
                content = msg.get("content", [])
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "image":
                            img_path = part["image"]
                            # 替换增强目录名
                            img_path = img_path.replace("IM_D03_PT_5k_Augmented", "IM_D03_PT_5K")
                            # 转绝对路径
                            if not os.path.isabs(img_path) and not img_path.startswith("file://"):
                                if image_dir:
                                    # 只取文件名，拼上 image_dir
                                    basename = os.path.basename(img_path)
                                    img_path = os.path.join(image_dir, basename)
                                else:
                                    img_path = os.path.abspath(img_path)
                            part["image"] = f"file://{img_path}"

            items.append({
                "messages": messages,
                "dataitem_name": name,
            })

    logger.info(f"Loaded {len(items)} test samples from {jsonl_path}")
    return items


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-4B inference from JSONL")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True,
                        help="test_v2.jsonl path")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Image directory (absolute path)")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()

    output_file = Path(args.output_file)
    checkpoint_path = output_file.with_suffix(".checkpoint.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 加载测试数据
    test_items = load_test_data(args.test_data, args.image_dir)
    if not test_items:
        logger.error("No test data loaded!")
        return

    # 断点续传
    done_dict = load_checkpoint(checkpoint_path)
    remaining = [item for item in test_items if item["dataitem_name"] not in done_dict]
    logger.info(f"Total: {len(test_items)}, Done: {len(done_dict)}, Remaining: {len(remaining)}")

    if not remaining:
        logger.info("All samples already processed.")
    else:
        # 加载模型
        model, processor = load_model(args.model_path, args.adapter_path)

        ckpt_f = open(checkpoint_path, "a", encoding="utf-8")
        total = len(remaining)
        t_start = time.time()

        for i, item in enumerate(remaining):
            t0 = time.time()
            name = item["dataitem_name"]

            try:
                raw_text = run_inference_from_messages(
                    model, processor, item["messages"], args.max_new_tokens
                )
                parsed = parse_output(raw_text)
                error = None
            except Exception as e:
                logger.error(f"[{i+1}/{total}] ERROR on {name}: {e}")
                raw_text = ""
                parsed = []
                error = str(e)

            elapsed = time.time() - t0
            total_done = i + 1
            avg_time = (time.time() - t_start) / total_done
            eta_sec = avg_time * (total - total_done)
            eta_str = f"{int(eta_sec // 3600)}h{int((eta_sec % 3600) // 60):02d}m"

            status = f"{len(parsed)} det" if parsed else ("parse_fail" if raw_text else "error")
            logger.info(f"[{total_done}/{total}] {name} | {status} | {elapsed:.1f}s | ETA {eta_str}")
            sys.stdout.flush()

            record = {
                "dataitem_name": name,
                "response": raw_text,
                "parsed_count": len(parsed),
            }
            if error:
                record["_error"] = error

            done_dict[name] = record
            ckpt_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            ckpt_f.flush()

            if total_done % args.save_every == 0:
                _save_final(done_dict, test_items, output_file)
                logger.info(f"  Progress saved ({total_done}/{total})")

        ckpt_f.close()
        logger.info(f"Inference complete. Processed {total} samples.")

    # 保存最终结果
    _save_final(done_dict, test_items, output_file)
    logger.info(f"Final output saved → {output_file}")

    # 统计
    total_det = sum(v.get("parsed_count", 0) for v in done_dict.values())
    errors = sum(1 for v in done_dict.values() if "_error" in v)
    logger.info(f"  Total: {len(done_dict)} samples, {total_det} detections, {errors} errors")


def _save_final(done_dict: dict, test_items: list, output_file: Path):
    ordered = []
    for item in test_items:
        name = item["dataitem_name"]
        if name in done_dict:
            ordered.append(done_dict[name])
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
