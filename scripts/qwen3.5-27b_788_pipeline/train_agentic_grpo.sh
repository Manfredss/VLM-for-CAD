#!/bin/bash
# ================================================================
# Agentic GRPO training — Siemens 788 pipeline
#
# Extends standard GRPO with multi-turn self-refinement.
# The model is trained to:
#   1. Produce good initial detections (standard reward)
#   2. Critique its own output accurately
#   3. Refine/correct mistakes (improvement bonus)
#
# The training data includes the full multi-turn conversation:
#   user → views → features → critique → refined features
#
# Reward = base_reward(refined_output) + 0.5 * max(-0.2, min(0.2, delta))
# where delta = score(refined) - score(initial)
#
# Prerequisites:
#   1. SFT checkpoint from train_sft.sh
#   2. Agentic training data (run agentic_rl_inference.py first to generate)
#
# Usage:
#   SFT_CHECKPOINT=/workspace/output/swift_27b_788_pipeline/v0-*/swa_adapter \
#   TRAIN_DATASET=/workspace/data/agentic_train.jsonl \
#   bash train_agentic_grpo.sh
# ================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d /workspace/venv/bin ]; then
    export PATH="/workspace/venv/bin:${PATH}"
fi

# --- Config ---
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-27B}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-}"
TRAIN_DATASET="${TRAIN_DATASET:-/workspace/data/agentic_train.jsonl}"
VAL_DATASET="${VAL_DATASET:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/swift_27b_788_pipeline_agentic_grpo}"

TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
export TRAIN_CUDA_VISIBLE_DEVICES
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29502}"

LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"    # lower than standard GRPO
VIT_LR="${VIT_LR:-1e-6}"
ALIGNER_LR="${ALIGNER_LR:-2e-6}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"  # agentic RL uses fewer epochs
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MAX_LENGTH="${MAX_LENGTH:-16384}"           # longer for multi-turn
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-4096}"
SAVE_STEPS="${SAVE_STEPS:-50}"
EVAL_STEPS="${EVAL_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

# Agentic GRPO-specific
NUM_SAMPLES="${NUM_SAMPLES:-4}"
TEMPERATURE="${TEMPERATURE:-0.7}"
KL_COEFF="${KL_COEFF:-0.02}"               # tighter KL to stay near SFT
BETA="${BETA:-0.04}"
IMPROVEMENT_BONUS_WEIGHT="${IMPROVEMENT_BONUS_WEIGHT:-0.5}"

REWARD_PLUGIN="${REWARD_PLUGIN:-${SCRIPT_DIR}/reward.py}"
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero3}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.cache/triton}"

# --- Validation ---
if [ -z "${SFT_CHECKPOINT}" ]; then
    echo ">>> ERROR: SFT_CHECKPOINT is required."
    echo ">>> Usage: SFT_CHECKPOINT=/path/to/swa_adapter bash train_agentic_grpo.sh"
    exit 1
fi

if [ ! -f "${TRAIN_DATASET}" ]; then
    echo ">>> ERROR: Training dataset not found: ${TRAIN_DATASET}"
    echo ">>> Generate agentic training data with agentic_rl_inference.py first."
    exit 1
fi

# --- Find swift ---
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

mkdir -p "$TRITON_CACHE_DIR"

# --- Print config ---
echo ">>> ===== Agentic GRPO Training — Siemens 788 Pipeline ====="
echo ">>> Base model:        ${BASE_MODEL}"
echo ">>> SFT checkpoint:    ${SFT_CHECKPOINT}"
echo ">>> Train dataset:     ${TRAIN_DATASET}"
echo ">>> Output dir:        ${OUTPUT_DIR}"
echo ">>> GPUs:              ${TRAIN_CUDA_VISIBLE_DEVICES} (nproc=${NPROC_PER_NODE})"
echo ">>> LoRA:              rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo ">>> LR:                ${LEARNING_RATE} (ViT=${VIT_LR}, Aligner=${ALIGNER_LR})"
echo ">>> Epochs:            ${NUM_TRAIN_EPOCHS}"
echo ">>> GRPO samples:      ${NUM_SAMPLES}"
echo ">>> Temperature:       ${TEMPERATURE}"
echo ">>> KL coeff:          ${KL_COEFF}"
echo ">>> Beta:              ${BETA}"
echo ">>> Improvement bonus: ${IMPROVEMENT_BONUS_WEIGHT}"
echo ">>> Reward plugin:     ${REWARD_PLUGIN}"
echo ">>> DeepSpeed:         ${DEEPSPEED_CONFIG}"
echo ">>> "
echo ">>> Agentic mode: multi-turn self-refinement"
echo ">>> (detect → critique → refine)"
echo ">>> Reward = base_score(refined) + ${IMPROVEMENT_BONUS_WEIGHT} * delta"

# --- Build extra args ---
EXTRA_ARGS=()
EXTRA_ARGS+=(--adapters "${SFT_CHECKPOINT}")
EXTRA_ARGS+=(--num_samples_per_prompt "${NUM_SAMPLES}")
EXTRA_ARGS+=(--temperature "${TEMPERATURE}")
EXTRA_ARGS+=(--beta "${BETA}")

if [ -n "${VAL_DATASET}" ] && [ -f "${VAL_DATASET}" ]; then
    EXTRA_ARGS+=(--val_dataset "${VAL_DATASET}")
fi

# --- Run ---
NPROC_PER_NODE="${NPROC_PER_NODE}" \
MASTER_PORT="${MASTER_PORT}" \
CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" \
USE_HF=1 \
QWENVL_BBOX_FORMAT='new' \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM}" \
TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
"${SWIFT_BIN}" rlhf \
    --rlhf_type grpo \
    --model "${BASE_MODEL}" \
    --dataset "${TRAIN_DATASET}" \
    --external_plugins "${REWARD_PLUGIN}" \
    --reward_funcs compute_reward \
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
    --vit_gradient_checkpointing true \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --eval_steps "${EVAL_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --logging_steps 5 \
    --max_length "${MAX_LENGTH}" \
    --output_dir "${OUTPUT_DIR}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --learning_rate "${LEARNING_RATE}" \
    --vit_lr "${VIT_LR}" \
    --aligner_lr "${ALIGNER_LR}" \
    --lr_scheduler_type cosine \
    --report_to none \
    --log_completions true \
    "${EXTRA_ARGS[@]}"
