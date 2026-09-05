#!/bin/bash
# ================================================================
# Qwen3.5-27B ms-swift LoRA training — Multi-step conversation
#
# Multi-step design:
#   Turn 1: Layout & view detection (11 view/layout categories)
#   Turn 2: Structural feature detection (14 feature categories + size)
#
# Key settings (v2: higher resolution + capacity):
#   - max_length=8192, IMAGE_MAX_TOKEN_NUM=2560 (1.6x resolution)
#   - LoRA rank=64 all-linear (2x capacity)
#   - batch_size=1, grad_acc=4 (same effective batch)
#   - DeepSpeed Zero-3 on 2xA100
#   - Hierarchical LR: ViT 1e-5, Aligner 2e-5, LLM 2e-5
#   - 7 epochs with oversampled weak categories
#   - Unlocked ViT + Aligner
#
# Usage:
#   TRAIN_CUDA_VISIBLE_DEVICES=0,1 bash /workspace/scripts_27b_swift/train_swift_multistep.sh
# ================================================================

ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
TRAIN_DATASET="${TRAIN_DATASET:-/workspace/data/train_multistep.jsonl}"
VAL_DATASET="${VAL_DATASET:-/workspace/data/val_multistep.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/swift_27b_multistep_v2}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-7}"
RESUME_ONLY_MODEL="${RESUME_ONLY_MODEL:-false}"
ADD_VERSION="${ADD_VERSION:-true}"
IGNORE_DATA_SKIP="${IGNORE_DATA_SKIP:-false}"
MASTER_PORT="${MASTER_PORT:-29500}"
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-2560}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
VIT_LR="${VIT_LR:-1e-5}"
ALIGNER_LR="${ALIGNER_LR:-2e-5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.cache/triton}"
TRITON_AUTOTUNE_DIR="${TRITON_AUTOTUNE_DIR:-${HOME:-/home/jovyan}/.triton/autotune}"

if [ "${NPROC_PER_NODE}" -le 0 ]; then
    NPROC_PER_NODE=1
fi

if [ -z "${GRADIENT_ACCUMULATION_STEPS:-}" ]; then
    GRADIENT_ACCUMULATION_STEPS=4
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

echo ">>> ===== Multi-Step Training ====="
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
    --external_plugins /workspace/scripts_27b_swift_multistep/optimizer.py /workspace/scripts_27b_swift_multistep/metric.py \
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
