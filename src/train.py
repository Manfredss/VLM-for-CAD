"""
Training script for fine-tuning Qwen3-VL on industrial drawing feature recognition.

Usage:
    python src/train.py --config configs/train_config.yaml
    
    # Override config values via CLI:
    python src/train.py --config configs/train_config.yaml \
        --training.num_train_epochs 5 \
        --training.learning_rate 5e-5

    # Multi-GPU with DeepSpeed:
    torchrun --nproc_per_node=4 src/train.py \
        --config configs/train_config.yaml \
        --training.deepspeed configs/ds_config.json
"""

import os
import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from dataset import DrawingFeatureDataset, DataCollatorForVLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str, overrides: dict = None) -> dict:
    """Load YAML config and apply CLI overrides."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if overrides:
        for key, value in overrides.items():
            parts = key.split(".")
            d = config
            for part in parts[:-1]:
                d = d[part]
            # Try to cast to appropriate type
            old_val = d.get(parts[-1])
            if old_val is not None:
                if isinstance(old_val, bool):
                    value = value.lower() in ("true", "1", "yes")
                elif isinstance(old_val, int):
                    value = int(value)
                elif isinstance(old_val, float):
                    value = float(value)
            d[parts[-1]] = value

    return config


def download_model(config: dict) -> str:
    """Download model from specified source and return local path."""
    model_cfg = config["model"]

    if model_cfg.get("local_path") and Path(model_cfg["local_path"]).exists():
        logger.info(f"Using local model: {model_cfg['local_path']}")
        return model_cfg["local_path"]

    source = model_cfg.get("source", "modelscope")

    if source == "modelscope":
        try:
            from modelscope import snapshot_download
            model_name = model_cfg["name_ms"]
            logger.info(f"Downloading model from ModelScope: {model_name}")
            local_path = snapshot_download(model_name)
            return local_path
        except ImportError:
            logger.warning("modelscope not installed, falling back to huggingface")
            source = "huggingface"

    if source == "huggingface":
        model_name = model_cfg["name_hf"]
        logger.info(f"Using HuggingFace model: {model_name}")
        return model_name

    raise ValueError(f"Unknown model source: {source}")


def setup_model_and_processor(config: dict, model_path: str):
    """Initialize model and processor."""
    model_cfg = config["model"]

    # Determine dtype
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map.get(model_cfg.get("dtype", "bf16"), torch.bfloat16)

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # Load model
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }

    if model_cfg.get("use_flash_attn", True):
        model_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        # Use PyTorch's built-in SDPA (fused kernel, nearly as fast as flash-attn)
        model_kwargs["attn_implementation"] = "sdpa"

    logger.info(f"Loading model from {model_path} with dtype={dtype}")
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        **model_kwargs,
    )

    # Apply LoRA if enabled
    lora_cfg = config.get("lora", {})
    if lora_cfg.get("enabled", True):
        logger.info("Applying LoRA configuration")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg.get("r", 64),
            lora_alpha=lora_cfg.get("lora_alpha", 128),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
            modules_to_save=lora_cfg.get("modules_to_save", None),
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Enable gradient checkpointing
    if config["training"].get("gradient_checkpointing", True):
        model.enable_input_require_grads()

    return model, processor


def build_training_args(config: dict, has_eval: bool = False) -> TrainingArguments:
    """Build HuggingFace TrainingArguments from config.

    Args:
        config: Full training config dict.
        has_eval: Whether a validation dataset is available. When False,
                  eval_strategy is forced to "no" and load_best_model_at_end
                  is disabled (HuggingFace Trainer errors if both are set
                  inconsistently).
    """
    train_cfg = config["training"]

    args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        logging_steps=train_cfg.get("logging_steps", 10),
        logging_dir=train_cfg.get("logging_dir", "outputs/logs"),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        eval_strategy=train_cfg.get("eval_strategy", "steps") if has_eval else "no",
        eval_steps=train_cfg.get("eval_steps", 200) if has_eval else None,
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        # use_reentrant=False avoids deprecation warnings and works correctly
        # with PEFT/LoRA in PyTorch >= 2.1.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=train_cfg.get("deepspeed", None),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        dataloader_pin_memory=train_cfg.get("dataloader_pin_memory", True),
        seed=train_cfg.get("seed", 42),
        report_to=["tensorboard"],
        remove_unused_columns=False,
        # Only load the best checkpoint if we actually ran evaluation.
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
    )

    return args


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL for drawing feature recognition")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to config file")
    
    # Allow arbitrary config overrides via --key.subkey value
    args, unknown = parser.parse_known_args()

    # Parse overrides
    overrides = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                overrides[key] = unknown[i + 1]
                i += 2
            else:
                overrides[key] = "true"
                i += 1
        else:
            i += 1

    # Load config
    config = load_config(args.config, overrides if overrides else None)

    # Set seed
    set_seed(config["training"].get("seed", 42))

    # Download / locate model
    model_path = download_model(config)

    # Setup model and processor
    model, processor = setup_model_and_processor(config, model_path)

    # Build datasets
    dataset_cfg = config["dataset"]
    logger.info("Loading training dataset...")
    train_dataset = DrawingFeatureDataset(
        json_path=dataset_cfg["train_json"],
        image_root=dataset_cfg["image_root"],
        processor=processor,
        system_prompt=dataset_cfg.get("system_prompt", ""),
        max_seq_length=config["training"].get("max_seq_length", 4096),
        max_samples=dataset_cfg.get("max_samples", None),
        min_pixels=dataset_cfg.get("min_pixels", 256),
        max_pixels=dataset_cfg.get("max_pixels", 1280),
    )

    val_dataset = None
    if dataset_cfg.get("val_json") and Path(dataset_cfg["val_json"]).exists():
        logger.info("Loading validation dataset...")
        val_dataset = DrawingFeatureDataset(
            json_path=dataset_cfg["val_json"],
            image_root=dataset_cfg["image_root"],
            processor=processor,
            system_prompt=dataset_cfg.get("system_prompt", ""),
            max_seq_length=config["training"].get("max_seq_length", 4096),
            min_pixels=dataset_cfg.get("min_pixels", 256),
            max_pixels=dataset_cfg.get("max_pixels", 1280),
        )
    else:
        logger.warning("No validation dataset found – evaluation will be skipped.")

    has_eval = val_dataset is not None

    # Data collator
    data_collator = DataCollatorForVLM(
        processor=processor,
        max_seq_length=config["training"].get("max_seq_length", 4096),
    )

    # Build training arguments
    training_args = build_training_args(config, has_eval=has_eval)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Resume or start training
    resume_path = config["training"].get("resume_from_checkpoint", None)
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info(f"  Total samples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"  Val samples: {len(val_dataset)}")
    logger.info(f"  Epochs: {config['training']['num_train_epochs']}")
    logger.info(f"  Batch size: {config['training']['per_device_train_batch_size']}")
    logger.info(f"  Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    logger.info(f"  Learning rate: {config['training']['learning_rate']}")
    logger.info(f"  Output dir: {config['training']['output_dir']}")
    logger.info("=" * 60)

    trainer.train(resume_from_checkpoint=resume_path)

    # Save the final model
    final_output_dir = os.path.join(config["training"]["output_dir"], "final")
    logger.info(f"Saving final model to {final_output_dir}")

    if config.get("lora", {}).get("enabled", True):
        # Save LoRA adapter weights
        model.save_pretrained(final_output_dir)
        processor.save_pretrained(final_output_dir)
        logger.info("LoRA adapter saved. To merge with base model, use merge_lora.py")
    else:
        # Save full model
        trainer.save_model(final_output_dir)
        processor.save_pretrained(final_output_dir)

    # Persist the exact config used for this run alongside the model weights
    config_save_path = os.path.join(final_output_dir, "train_config_used.yaml")
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    logger.info(f"Training config saved to {config_save_path}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
