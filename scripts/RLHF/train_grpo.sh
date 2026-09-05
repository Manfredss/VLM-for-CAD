#!/bin/bash
# ================================================================
# GRPO/DPO training for Qwen3.5-27B engineering drawing analysis
#
# Prerequisites:
#   1. Best SFT checkpoint available
#   2. Training data prepared via prepare_preference_data.py
#
# Usage:
#   # GRPO mode (default)
#   bash train_grpo.sh
#
#   # DPO mode
#   RLHF_TYPE=dpo TRAIN_DATASET=/workspace/data/rlhf_dpo.jsonl bash train_grpo.sh
#
#   # Custom checkpoint
#   SFT_CHECKPOINT=/workspace/output/swift_27b_multistep_v2/checkpoint-1800 bash train_grpo.sh
# ================================================================

set -euo pipefail

# --- Config (all overridable via env) ---
RLHF_TYPE="${RLHF_TYPE:-grpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-27B}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-}"
TRAIN_DATASET="${TRAIN_DATASET:-/workspace/data/rlhf_grpo.jsonl}"
VAL_DATASET="${VAL_DATASET:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/rlhf_${RLHF_TYPE}}"

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29501}"

LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
VIT_LR="${VIT_LR:-2e-6}"
ALIGNER_LR="${ALIGNER_LR:-5e-6}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-2560}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"

# GRPO-specific
NUM_SAMPLES="${NUM_SAMPLES:-4}"
TEMPERATURE="${TEMPERATURE:-0.7}"
KL_COEFF="${KL_COEFF:-0.05}"

REWARD_PLUGIN="${REWARD_PLUGIN:-$(dirname "$0")/reward.py}"
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"

# --- Validation ---
if [ -z "${SFT_CHECKPOINT}" ]; then
    echo ">>> ERROR: SFT_CHECKPOINT is required."
    echo ">>> Usage: SFT_CHECKPOINT=/path/to/checkpoint bash train_grpo.sh"
    exit 1
fi

if [ ! -f "${TRAIN_DATASET}" ]; then
    echo ">>> ERROR: Training dataset not found: ${TRAIN_DATASET}"
    echo ">>> Run prepare_preference_data.py first."
    exit 1
fi

# --- Find swift binary ---
if command -v swift >/dev/null 2>&1; then
    SWIFT_BIN="$(command -v swift)"
elif [ -x /opt/conda/bin/swift ]; then
    SWIFT_BIN="/opt/conda/bin/swift"
else
    echo ">>> ERROR: ms-swift executable not found"
    exit 127
fi

# --- Check flash attention ---
if ! python3 -c "import flash_attn" >/dev/null 2>&1; then
    ATTN_IMPL="sdpa"
fi

# --- Print config ---
echo ">>> ===== RLHF Training (${RLHF_TYPE}) ====="
echo ">>> Base model:    ${BASE_MODEL}"
echo ">>> SFT checkpoint: ${SFT_CHECKPOINT}"
echo ">>> Train dataset: ${TRAIN_DATASET}"
echo ">>> Output dir:    ${OUTPUT_DIR}"
echo ">>> GPUs:          ${CUDA_DEVICES} (nproc=${NPROC_PER_NODE})"
echo ">>> LoRA:          rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo ">>> LR:            ${LEARNING_RATE} (ViT=${VIT_LR}, Aligner=${ALIGNER_LR})"
echo ">>> Epochs:        ${NUM_TRAIN_EPOCHS}"
echo ">>> Attention:     ${ATTN_IMPL}"
if [ "${RLHF_TYPE}" = "grpo" ]; then
    echo ">>> GRPO samples:  ${NUM_SAMPLES}"
    echo ">>> Temperature:   ${TEMPERATURE}"
    echo ">>> KL coeff:      ${KL_COEFF}"
fi
echo ">>> Reward plugin: ${REWARD_PLUGIN}"

# --- Build command ---
EXTRA_ARGS=()

# Load SFT adapter
EXTRA_ARGS+=(--adapters "${SFT_CHECKPOINT}")

# GRPO-specific args
if [ "${RLHF_TYPE}" = "grpo" ]; then
    EXTRA_ARGS+=(
        --num_samples_per_prompt "${NUM_SAMPLES}"
        --temperature "${TEMPERATURE}"
    )
fi

# Val dataset if provided
if [ -n "${VAL_DATASET}" ] && [ -f "${VAL_DATASET}" ]; then
    EXTRA_ARGS+=(--val_dataset "${VAL_DATASET}")
fi

# --- Run ---
NPROC_PER_NODE="${NPROC_PER_NODE}" \
MASTER_PORT="${MASTER_PORT}" \
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
USE_HF=1 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM}" \
"${SWIFT_BIN}" rlhf \
    --rlhf_type "${RLHF_TYPE}" \
    --model "${BASE_MODEL}" \
    --dataset "${TRAIN_DATASET}" \
    --external_plugins "${REWARD_PLUGIN}" \
    --torch_dtype bfloat16 \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size 1 \
    --attn_impl "${ATTN_IMPL}" \
    --tuner_type lora \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --target_modules all-linear \
    --enable_thinking false \
    --freeze_vit false \
    --freeze_aligner false \
    --gradient_checkpointing true \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --eval_steps "${EVAL_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --logging_steps 5 \
    --max_length "${MAX_LENGTH}" \
    --output_dir "${OUTPUT_DIR}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --deepspeed zero3 \
    --learning_rate "${LEARNING_RATE}" \
    --vit_lr "${VIT_LR}" \
    --aligner_lr "${ALIGNER_LR}" \
    --lr_scheduler_type cosine \
    --report_to none \
    "${EXTRA_ARGS[@]}"
