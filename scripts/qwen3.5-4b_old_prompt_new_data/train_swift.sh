#!/bin/bash
# ================================================================
# Qwen3.5-4B ms-swift LoRA Training — Old Prompt + New Data
# Single GPU, no DeepSpeed.
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_FILE="${TRAIN_FILE:-/workspace/finetune/data/data/train_10feats_oldprompt_4b.jsonl}"
VAL_FILE="${VAL_FILE:-/workspace/finetune/data/data/val_10feats_oldprompt_4b.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/finetune/output/swift_4b_oldprompt}"

ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
if ! python3 -c "import flash_attn" >/dev/null 2>&1; then
    ATTN_IMPL="sdpa"
elif ! python3 -c "import torch; assert torch.cuda.get_device_capability(0)[0] >= 8" >/dev/null 2>&1; then
    ATTN_IMPL="sdpa"
fi

echo ">>> Attention backend: ${ATTN_IMPL}"
echo ">>> Train dataset: $TRAIN_FILE"
echo ">>> Val dataset:   $VAL_FILE"
echo ">>> Output dir:    $OUTPUT_DIR"

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
    --dataset "$TRAIN_FILE" \
    --val_dataset "$VAL_FILE" \
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
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --attn_impl "$ATTN_IMPL" \
    --packing false \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --enable_thinking false \
    --add_non_thinking_prefix true \
    --freeze_vit false \
    --freeze_aligner false \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing true \
    --truncation_strategy left \
    --gradient_accumulation_steps 8 \
    --eval_steps 400 \
    --save_steps 400 \
    --save_total_limit 5 \
    --save_only_model true \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.05 \
    --neftune_noise_alpha 3 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
    --vit_lr 5e-5 \
    --aligner_lr 5e-5 \
    --learning_rate 2e-5
