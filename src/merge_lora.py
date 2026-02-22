"""
Merge LoRA adapter weights into the base model for easier deployment.

Usage:
    python src/merge_lora.py \
        --base_model Qwen/Qwen3-VL-3B \
        --adapter_path outputs/qwen3-vl-3b-lora/final \
        --output_path outputs/qwen3-vl-3b-merged
"""

import argparse
import logging
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", type=str, required=True, help="Base model path")
    parser.add_argument("--adapter_path", type=str, required=True, help="LoRA adapter path")
    parser.add_argument("--output_path", type=str, required=True, help="Output merged model path")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    logger.info(f"Loading base model: {args.base_model}")
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model, torch_dtype=torch_dtype, trust_remote_code=True
    )

    logger.info(f"Loading LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    logger.info("Merging LoRA weights...")
    model = model.merge_and_unload()

    # After merge_and_unload() the model is a plain base model, but newer
    # transformers versions keep _hf_peft_config_loaded=True on the instance,
    # which routes save_pretrained through a PEFT code path that has an
    # UnboundLocalError bug in active_adapters(). Clear the flag so
    # save_pretrained takes the normal (non-PEFT) path.
    if getattr(model, "_hf_peft_config_loaded", False):
        model._hf_peft_config_loaded = False

    logger.info(f"Saving merged model to: {args.output_path}")
    model.save_pretrained(args.output_path, safe_serialization=True)

    # Save processor
    try:
        processor = AutoProcessor.from_pretrained(args.adapter_path, trust_remote_code=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    processor.save_pretrained(args.output_path)

    logger.info("Done! Merged model saved.")


if __name__ == "__main__":
    main()
