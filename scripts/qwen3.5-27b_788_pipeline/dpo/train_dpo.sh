#!/bin/bash
# ================================================================
# DPO fine-tune on top of the SFT model.
#
# Inputs:
#   - SFT adapter (from train_swift.sh, ideally swa_adapter)
#   - DPO JSONL produced by make_dpo_jsonl.py
#
# Output:
#   - DPO-tuned adapter (replaces / extends the SFT adapter)
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d /workspace/venv/bin ]; then
    export PATH="/workspace/venv/bin:${PATH}"
fi

# --- inputs ---
ADAPTER="${ADAPTER:-/workspace/output/swift_27b_788_pipeline/v0-*/swa_adapter}"
DPO_DATASET="${DPO_DATASET:-/workspace/data/dpo_view_7feats_pipeline.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/swift_27b_788_pipeline_dpo}"

# --- DPO hyperparameters ---
DPO_BETA="${DPO_BETA:-0.1}"        # standard
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-7}"   # DPO uses much smaller LR than SFT
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
MAX_LENGTH="${MAX_LENGTH:-10240}"
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-2560}"

# --- runtime ---
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
export TRAIN_CUDA_VISIBLE_DEVICES

_count_devices() {
    python3 - <<'PY'
import os
d = os.environ.get('TRAIN_CUDA_VISIBLE_DEVICES','').strip()
print(len([x for x in d.split(',') if x.strip()]) if d else 0)
PY
}
NPROC_PER_NODE="${NPROC_PER_NODE:-$(_count_devices)}"
[ "${NPROC_PER_NODE}" -le 0 ] && NPROC_PER_NODE=1

if command -v swift >/dev/null 2>&1; then
    SWIFT_BIN="$(command -v swift)"
else
    echo ">>> ms-swift not on PATH" >&2
    exit 127
fi

ATTN_IMPL="sdpa"
python3 -c "import flash_attn" 2>/dev/null && ATTN_IMPL="flash_attn"

echo ">>> ===== DPO fine-tune ====="
echo ">>> Adapter (SFT init):  ${ADAPTER}"
echo ">>> DPO dataset:         ${DPO_DATASET}"
echo ">>> Output:              ${OUTPUT_DIR}"
echo ">>> beta=${DPO_BETA} lr=${LEARNING_RATE} epochs=${NUM_TRAIN_EPOCHS}"
echo ">>> GPUs: ${TRAIN_CUDA_VISIBLE_DEVICES} (nproc=${NPROC_PER_NODE})"

NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" \
USE_HF=1 \
QWENVL_BBOX_FORMAT='new' \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM}" \
"${SWIFT_BIN}" rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen3.5-27B \
    --adapters ${ADAPTER} \
    --dataset "${DPO_DATASET}" \
    --torch_dtype bfloat16 \
    --beta "${DPO_BETA}" \
    --learning_rate "${LEARNING_RATE}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing true \
    --tuner_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --attn_impl "${ATTN_IMPL}" \
    --max_length "${MAX_LENGTH}" \
    --output_dir "${OUTPUT_DIR}" \
    --add_version true \
    --warmup_ratio 0.1 \
    --deepspeed zero3 \
    --logging_steps 5 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 5 \
    --report_to none
