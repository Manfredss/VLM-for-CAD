#!/bin/bash
# ================================================================
# Qwen3.5-27B ms-swift LoRA training — 15 features + view detection
#
# v2: Continued fine-tuning from checkpoint-1400 with rebalanced data
#
# Changes from v1:
#   - Resume from checkpoint-1400 (adapter weights only, fresh optimizer)
#   - Rebalanced training data (weak categories oversampled)
#   - Lower LR: LLM 1e-5, ViT 5e-6, Aligner 1e-5
#   - 5 epochs on rebalanced data
#   - LoRA rank=32, alpha=64 (kept same)
#
# Usage:
#   TRAIN_CUDA_VISIBLE_DEVICES=0,1 bash /workspace/scripts_27b_swift/train_swift.sh
# ================================================================

ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
TRAIN_DATASET="${TRAIN_DATASET:-/workspace/data/train_15feats_view_rebalanced.jsonl}"
VAL_DATASET="${VAL_DATASET:-/workspace/data/val_15feats_view.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/swift_27b_15feats_view_v2}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-/workspace/output/swift_27b_15feats_view/v0-20260322-085618/checkpoint-1400}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
RESUME_ONLY_MODEL="${RESUME_ONLY_MODEL:-true}"
ADD_VERSION="${ADD_VERSION:-true}"
IGNORE_DATA_SKIP="${IGNORE_DATA_SKIP:-false}"
MASTER_PORT="${MASTER_PORT:-29500}"
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1560}"
MAX_LENGTH="${MAX_LENGTH:-6144}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
VIT_LR="${VIT_LR:-5e-6}"
ALIGNER_LR="${ALIGNER_LR:-1e-5}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.cache/triton}"
TRITON_AUTOTUNE_DIR="${TRITON_AUTOTUNE_DIR:-${HOME:-/home/jovyan}/.triton/autotune}"

_count_devices() {
    python3 - <<'PY'
import os
devices = os.environ.get('TRAIN_CUDA_VISIBLE_DEVICES', '').strip()
if not devices:
    print(0)
else:
    print(len([x for x in devices.split(',') if x.strip() != '']))
PY
}

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
if [ "${NPROC_PER_NODE}" -le 0 ]; then
    NPROC_PER_NODE=1
fi

if [ -z "${GRADIENT_ACCUMULATION_STEPS:-}" ]; then
    GRADIENT_ACCUMULATION_STEPS=2
fi

if command -v swift >/dev/null 2>&1; then
    SWIFT_BIN="$(command -v swift)"
elif [ -x /opt/conda/bin/swift ]; then
    SWIFT_BIN="/opt/conda/bin/swift"
else
    echo ">>> ERROR: ms-swift executable not found"
    exit 127
fi

if ! python3 -c "import flash_attn" >/dev/null 2>&1; then
    ATTN_IMPL="sdpa"
fi

if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
fi

mkdir -p "$TRITON_CACHE_DIR" "$TRITON_AUTOTUNE_DIR"

echo ">>> ===== 15-Feature + View Training (v2: rebalanced continued) ====="
echo ">>> Attention backend: ${ATTN_IMPL}"
echo ">>> Train dataset: ${TRAIN_DATASET}"
echo ">>> Val dataset:   ${VAL_DATASET}"
echo ">>> Output dir:    ${OUTPUT_DIR}"
echo ">>> Train GPUs:    ${TRAIN_CUDA_VISIBLE_DEVICES} (nproc=${NPROC_PER_NODE})"
echo ">>> Train batch:   per_device=${PER_DEVICE_TRAIN_BATCH_SIZE}, grad_acc=${GRADIENT_ACCUMULATION_STEPS}"
echo ">>> LoRA:          rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo ">>> Image budget:  token_num=${IMAGE_MAX_TOKEN_NUM}"
echo ">>> Max length:    ${MAX_LENGTH}"
echo ">>> LRs:           LLM=${LEARNING_RATE}, ViT=${VIT_LR}, Aligner=${ALIGNER_LR}"
echo ">>> Eval/save:     ${EVAL_STEPS}/${SAVE_STEPS}"
echo ">>> Save limit:    ${SAVE_TOTAL_LIMIT}"
echo ">>> Epochs:        ${NUM_TRAIN_EPOCHS}"

EXTRA_ARGS=()
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    echo ">>> Resume ckpt:   ${RESUME_FROM_CHECKPOINT}"
    if [ "${RESUME_ONLY_MODEL}" = "true" ]; then
        echo ">>> Resume mode:   adapters_only=true"
        EXTRA_ARGS+=(--adapters "${RESUME_FROM_CHECKPOINT}" --load_args false)
    else
        echo ">>> Resume mode:   resume_only_model=${RESUME_ONLY_MODEL}, ignore_data_skip=${IGNORE_DATA_SKIP}"
        EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}" --resume_only_model "${RESUME_ONLY_MODEL}")
    fi
fi

NPROC_PER_NODE="${NPROC_PER_NODE}" \
MASTER_PORT="${MASTER_PORT}" \
CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" \
USE_HF=1 \
QWENVL_BBOX_FORMAT='new' \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM}" \
TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
"${SWIFT_BIN}" sft \
    --model Qwen/Qwen3.5-27B \
    --dataset "${TRAIN_DATASET}" \
    --val_dataset "${VAL_DATASET}" \
    --external_plugins /workspace/scripts_27b_swift/optimizer.py /workspace/scripts_27b_swift/metric.py \
    --optimizer IMoptimizer \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --strict true \
    --load_from_cache_file true \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size 1 \
    --attn_impl "$ATTN_IMPL" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --target_modules all-linear \
    --enable_thinking false \
    --add_non_thinking_prefix true \
    --freeze_vit false \
    --freeze_aligner false \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing true \
    --packing false \
    --truncation_strategy left \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --eval_steps "${EVAL_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --logging_steps 5 \
    --max_length "${MAX_LENGTH}" \
    --output_dir "${OUTPUT_DIR}" \
    --add_version "${ADD_VERSION}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --deepspeed zero3 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --lr_scheduler_type cosine \
    --vit_lr "${VIT_LR}" \
    --aligner_lr "${ALIGNER_LR}" \
    --learning_rate "${LEARNING_RATE}" \
    --report_to none \
    --ignore_data_skip "${IGNORE_DATA_SKIP}" \
    "${EXTRA_ARGS[@]}"
