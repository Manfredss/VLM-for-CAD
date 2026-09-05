#!/bin/bash
# WSL2 single-GPU training script for Qwen3.5-4B (RTX 5070 Ti 16 GB)  v2
#
# Key differences from train-swift-wsl.sh (v1):
#   - Uses v2 dataset (bbox_2d normalized 0-1000, native multimodal format)
#   - No custom template (native format handles image embedding)
#   - IMoptimizer for layered LR (ViT / aligner / LLM)
#   - IMMetric for IoU-based bbox+label accuracy during eval
#   - explicit metric_for_best_model + greater_is_better=false
#   - packing=false (simpler; no flash_attn required)
#
# Prerequisites:
#   python src/prepare_data_swift_v2.py   (generates data/train_v2.jsonl etc.)
#
# Run from the project root:
#   cd /mnt/d/Xue/ML/Retrain/retrain\ model
#   bash scripts/train-swift-wsl-v2.sh

set -e

# Resolve project root relative to this script
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
echo "Working directory: $PROJECT_ROOT"

# Verify required data files
if [ ! -f "data/train_v2.jsonl" ]; then
    echo "ERROR: data/train_v2.jsonl not found. Generate it first:"
    echo "  python src/prepare_data_swift_v2.py"
    exit 1
fi

# ── flash_attn detection ──────────────────────────────────────────────────────
if python -c "import flash_attn" 2>/dev/null; then
    ATTN_IMPL="flash_attn"
    echo "flash_attn detected — using flash_attn"
else
    ATTN_IMPL="sdpa"
    echo "flash_attn not found — using sdpa"
fi

# ── Training ──────────────────────────────────────────────────────────────────
OUTPUT_DIR="output/v$(date +%Y%m%d-%H%M%S)"
PLUGIN_DIR="$PROJECT_ROOT/scripts/nautilus_scripts_27b_swift"

USE_HF=1 \
CUDA_VISIBLE_DEVICES=0 \
QWENVL_BBOX_FORMAT='new' \
IMAGE_MAX_TOKEN_NUM=1280 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
METRIC_MODEL='Qwen/Qwen3.5-4B' \
swift sft \
    --model Qwen/Qwen3.5-4B \
    --dataset 'data/train_v2.jsonl' \
    --val_dataset 'data/val_v2.jsonl' \
    --system '你是一个专业的工业图纸分析助手。你的任务是识别和分析工程图纸中的工件特征，包括但不限于：螺纹孔、圆角、矩形孔、圆孔、长圆孔、倒角、槽等特征。对于每种特征，请输出其类型、规格尺寸（如M8、R3）以及在图纸中的位置坐标。请根据图纸内容给出准确、完整的分析结果。' \
    --external_plugins "$PLUGIN_DIR/optimizer.py" "$PLUGIN_DIR/metric.py" \
    --optimizer IMoptimizer \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --enable_channel_loss true \
    --strict true \
    --load_from_cache_file true \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_eval_batch_size 1 \
    --attn_impl "$ATTN_IMPL" \
    --packing false \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit false \
    --freeze_aligner false \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing true \
    --truncation_strategy left \
    --gradient_accumulation_steps 2 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 5 \
    --save_only_model true \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.05 \
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
