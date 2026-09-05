#!/bin/bash
# ================================================================
# Qwen3.5-4B ms-swift LoRA Post-Training — Rare Features
#
# Continue fine-tuning from the original-dataset checkpoint (ckpt-1200)
# on samples enriched with rare features:
#   - Rectangular Hole / Rectangular Hole Group (absent/0.3% in original)
#   - Slotted Hole / Slotted Hole Group (absent/1.8% in original)
#   - Fillet Group (4.6% in original)
#
# Key design choices for preserving existing ability:
#   - Load existing LoRA adapter weights (--adapters) with fresh optimizer
#   - Low learning rate (5e-6 LLM, 1e-5 ViT/aligner) — ~4x lower than initial
#   - Short training (2 epochs on rare-enriched set)
#   - 15% replay samples with common features to prevent forgetting
#   - Lower neftune noise (1.5 vs 3) for stability
#   - Cosine LR with higher min_lr floor to avoid collapse
#   - Frequent eval (every 100 steps) for early stopping via detection_f1
#
# Usage (called by run_pipeline_4b_posttrain.sh):
#   bash train_swift_4b_posttrain.sh
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${DATA_DIR:-/workspace/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/swift_4b_posttrain}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/workspace/finetune/output/swift_4b/v5-20260312-054110/checkpoint-1200}"

# Verify checkpoint exists
if [ ! -d "$RESUME_CHECKPOINT" ]; then
    echo "ERROR: Resume checkpoint not found: $RESUME_CHECKPOINT"
    echo "Available checkpoints:"
    ls -d /workspace/finetune/output/swift_4b/*/checkpoint-* 2>/dev/null || echo "  (none)"
    exit 1
fi
echo ">>> Loading adapter from: $RESUME_CHECKPOINT"

# Verify post-training data exists
TRAIN_DATA="$DATA_DIR/posttrain_rare.jsonl"
VAL_DATA="$DATA_DIR/posttrain_rare_val.jsonl"
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Post-training data not found: $TRAIN_DATA"
    exit 1
fi

# Flash Attention detection
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
if ! python3 -c "import flash_attn" >/dev/null 2>&1; then
    ATTN_IMPL="sdpa"
elif ! python3 -c "import torch; assert torch.cuda.get_device_capability(0)[0] >= 8" >/dev/null 2>&1; then
    ATTN_IMPL="sdpa"
fi
echo ">>> Attention backend: ${ATTN_IMPL}"
echo ">>> Train data: $TRAIN_DATA ($(wc -l < "$TRAIN_DATA") samples)"
echo ">>> Val data:   $VAL_DATA ($(wc -l < "$VAL_DATA") samples)"
echo ">>> Output dir: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 \
USE_HF=1 \
QWENVL_BBOX_FORMAT='new' \
IMAGE_MAX_TOKEN_NUM=1280 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
METRIC_MODEL='Qwen/Qwen3.5-4B' \
METRIC_IOU_THRESHOLD='0.4' \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift sft \
    --model Qwen/Qwen3.5-4B \
    --adapters "$RESUME_CHECKPOINT" \
    --dataset "$TRAIN_DATA" \
    --val_dataset "$VAL_DATA" \
    --system '你是一个专业的工业图纸分析助手。你的任务是识别和分析工程图纸中的工件特征，包括但不限于：螺纹孔、圆角、矩形孔、圆孔、长圆孔、倒角、槽等特征。对于每种特征，请输出其类型、规格尺寸（如M8、R3）以及在图纸中的位置坐标。请根据图纸内容给出准确、完整的分析结果。' \
    --external_plugins "$SCRIPT_DIR/optimizer.py" "$SCRIPT_DIR/metric.py" \
    --optimizer IMoptimizer \
    --eval_metric IMMetric \
    --metric_for_best_model detection_f1 \
    --greater_is_better true \
    --load_best_model_at_end true \
    --strict true \
    --load_from_cache_file true \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --attn_impl "$ATTN_IMPL" \
    --packing false \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --freeze_vit false \
    --freeze_aligner false \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing true \
    --truncation_strategy left \
    --gradient_accumulation_steps 4 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --save_only_model true \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.1 \
    --neftune_noise_alpha 1.5 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 2e-6}' \
    --vit_lr 1e-5 \
    --aligner_lr 1e-5 \
    --learning_rate 5e-6 \
