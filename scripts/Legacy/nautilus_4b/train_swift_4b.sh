#!/bin/bash
# ================================================================
# Qwen3.5-4B ms-swift LoRA 训练 — 1x A40 (48GB) / A100 (40/80GB)
#
# 用法（由 run_pipeline_4b.sh 调用，也可单独运行）:
#   bash /workspace/scripts_4b/train_swift_4b.sh
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${DATA_DIR:-/workspace/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/swift_4b}"

# Flash Attention 检测 (需要 Ampere+ GPU, compute capability >= 8.0)
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
if ! python3 -c "import flash_attn" >/dev/null 2>&1; then
    ATTN_IMPL="sdpa"
elif ! python3 -c "import torch; assert torch.cuda.get_device_capability(0)[0] >= 8" >/dev/null 2>&1; then
    echo ">>> GPU compute capability < 8.0, flash_attn not supported at runtime"
    ATTN_IMPL="sdpa"
fi
echo ">>> Attention backend: ${ATTN_IMPL}"
echo ">>> Data dir:   $DATA_DIR"
echo ">>> Output dir: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 \
USE_HF=1 \
QWENVL_BBOX_FORMAT='new' \
IMAGE_MAX_TOKEN_NUM=1280 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
METRIC_MODEL='Qwen/Qwen3.5-4B' \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift sft \
    --model Qwen/Qwen3.5-4B \
    --dataset "$DATA_DIR/train_v2.jsonl" \
    --val_dataset "$DATA_DIR/val_v2.jsonl" \
    --system '你是一个专业的工业图纸分析助手。你的任务是识别和分析工程图纸中的工件特征，包括但不限于：螺纹孔、圆角、矩形孔、圆孔、长圆孔、倒角、槽等特征。对于每种特征，请输出其类型、规格尺寸（如M8、R3）以及在图纸中的位置坐标。请根据图纸内容给出准确、完整的分析结果。' \
    --external_plugins "$SCRIPT_DIR/optimizer.py" "$SCRIPT_DIR/metric.py" \
    --optimizer IMoptimizer \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --load_best_model_at_end true \
    --strict true \
    --load_from_cache_file true \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
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
    --gradient_accumulation_steps 8 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 5 \
    --save_only_model true \
    --logging_steps 5 \
    --max_length 4096 \
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
    --report_to none
