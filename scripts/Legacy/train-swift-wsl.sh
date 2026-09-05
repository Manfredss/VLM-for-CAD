#!/bin/bash
# WSL2 single-GPU training script for Qwen3-VL-4B (RTX 5070 Ti 16 GB)
# Run from the project root:
#   cd /mnt/d/Xue/ML/Retrain/retrain\ model
#   bash scripts/train-swift-wsl.sh

set -e

# Resolve project root relative to this script
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
echo "Working directory: $PROJECT_ROOT"

# Verify required data files
if [ ! -f "data/train.jsonl" ]; then
    echo "ERROR: data/train.jsonl not found. Generate it first:"
    echo "  python src/prepare_data_swift.py"
    exit 1
fi

# ── flash_attn detection ──────────────────────────────────────────────────────
# flash_attn 2.7+ supports Blackwell (sm_100). Enable if installed.
if python -c "import flash_attn" 2>/dev/null; then
    ATTN_IMPL="flash_attn"
    PACKING="--packing true --padding_free true"
    echo "flash_attn detected — using flash_attn + packing"
else
    ATTN_IMPL="sdpa"
    PACKING=""
    echo "flash_attn not found — using sdpa (install flash_attn for ~20% speedup)"
fi

# ── Training ──────────────────────────────────────────────────────────────────
OUTPUT_DIR="output/v$(date +%Y%m%d-%H%M%S)"

# --greater_is_better true \

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
QWENVL_BBOX_FORMAT='new' \
IMAGE_MAX_TOKEN_NUM=1280 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
swift sft \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --dataset 'data/train.jsonl' \
    --val_dataset 'data/val.jsonl' \
    --custom_register_path template.py \
    --template IMTemplate \
    --enable_channel_loss true \
    --strict true \
    --load_from_cache_file true \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 6 \
    --per_device_train_batch_size 1 \
    --attn_impl "$ATTN_IMPL" \
    $PACKING \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --freeze_vit false \
    --freeze_aligner false \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing true \
    --truncation_strategy left \
    --gradient_accumulation_steps 4 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 5 \
    --save_only_model true \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.05 \
    --deepspeed no \
    --neftune_noise_alpha 5 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
    --vit_lr 5e-5 \
    --aligner_lr 5e-5 \
    --learning_rate 2e-5 \
    --report_to tensorboard

    # OOM options (apply in order):
    #   1. Reduce IMAGE_MAX_TOKEN_NUM=768
    #   2. Reduce --gradient_accumulation_steps 2
    #   3. Reduce --lora_rank 16
