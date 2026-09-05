#!/bin/bash
# ================================================================
# Qwen3.5-27B ms-swift LoRA training — IMPROVED variant
# (oversampling + sharpened prompts; SFT only — DPO is a separate step in dpo/)
#
# Improvements over scripts_27b_788_with_view (the base 0.93-F1 pipeline):
#   - A: train JSONL has tail-class oversampling (Section/Detail/Aux/Rear/Threaded etc.)
#   - B: STEP1 prompt has sharper rules for Notes bbox + Rear View identification
#
# Same training hyperparameters as the base 3A100 retry:
#   - max_length=10240, image_max_token_num=2560
#   - LoRA rank=64, alpha=128
#   - DeepSpeed Zero-3
#   - Hierarchical LR: ViT 3e-5, Aligner 2e-5, LLM 2e-5
#   - 6 epochs, eval/save every 50 steps
#   - per_device_batch=1, grad_acc=2
#
# Usage (on training pod):
#   TRAIN_CUDA_VISIBLE_DEVICES=0,1,2 bash /workspace/scripts_27b_788_with_view_improve/train_swift.sh
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Put the workspace venv first on PATH so `swift`, `python3`, etc. resolve to it.
if [ -d /workspace/venv/bin ]; then
    export PATH="/workspace/venv/bin:${PATH}"
fi

ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
TRAIN_DATASET="${TRAIN_DATASET:-/workspace/data/train_view_7feats_improve.jsonl}"
VAL_DATASET="${VAL_DATASET:-/workspace/data/val_view_7feats_improve.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/swift_27b_788_view_7feats_improve}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2}"
export TRAIN_CUDA_VISIBLE_DEVICES
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-6}"
RESUME_ONLY_MODEL="${RESUME_ONLY_MODEL:-false}"
ADD_VERSION="${ADD_VERSION:-true}"
IGNORE_DATA_SKIP="${IGNORE_DATA_SKIP:-false}"
MASTER_PORT="${MASTER_PORT:-29500}"
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-4096}"
MAX_LENGTH="${MAX_LENGTH:-12288}"
EVAL_STEPS="${EVAL_STEPS:-50}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-999}"
KEEP_BEST_N="${KEEP_BEST_N:-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
VIT_LR="${VIT_LR:-3e-5}"
ALIGNER_LR="${ALIGNER_LR:-2e-5}"
EXTERNAL_PLUGINS="${EXTERNAL_PLUGINS:-${SCRIPT_DIR}/optimizer.py ${SCRIPT_DIR}/metric.py}"
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

NPROC_PER_NODE="${NPROC_PER_NODE:-$(_count_devices)}"
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

JANITOR_LOG="${JANITOR_LOG:-/workspace/logs/keep_best_n_ckpts_view_7feats_improve.log}"
JANITOR_PID=""
_stop_janitor() {
    if [ -n "$JANITOR_PID" ] && kill -0 "$JANITOR_PID" 2>/dev/null; then
        kill -TERM "$JANITOR_PID" 2>/dev/null || true
    fi
}
trap _stop_janitor EXIT INT TERM

mkdir -p "$(dirname "$JANITOR_LOG")"
nohup python3 "${SCRIPT_DIR}/keep_best_n_ckpts.py" \
    --output-dir "${OUTPUT_DIR}" \
    --keep-n "${KEEP_BEST_N}" \
    --interval 60 \
    >> "$JANITOR_LOG" 2>&1 &
JANITOR_PID=$!
echo ">>> Top-N janitor started: PID ${JANITOR_PID}, keep_n=${KEEP_BEST_N}, log=${JANITOR_LOG}"

echo ">>> ===== Siemens 7 Features + 13 Views (multi-turn) ====="
echo ">>> Attention backend: ${ATTN_IMPL}"
echo ">>> Train dataset:   ${TRAIN_DATASET}"
echo ">>> Val dataset:     ${VAL_DATASET}"
echo ">>> Output dir:      ${OUTPUT_DIR}"
echo ">>> Train GPUs:      ${TRAIN_CUDA_VISIBLE_DEVICES} (nproc=${NPROC_PER_NODE})"
echo ">>> Train batch:     per_device=${PER_DEVICE_TRAIN_BATCH_SIZE}, grad_acc=${GRADIENT_ACCUMULATION_STEPS}"
echo ">>> LoRA:            rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo ">>> Image budget:    token_num=${IMAGE_MAX_TOKEN_NUM}"
echo ">>> Max length:      ${MAX_LENGTH}"
echo ">>> LRs:             LLM=${LEARNING_RATE}, ViT=${VIT_LR}, Aligner=${ALIGNER_LR}"
echo ">>> Eval/save steps: ${EVAL_STEPS}/${SAVE_STEPS}"
echo ">>> Save limit:      ${SAVE_TOTAL_LIMIT} (HF rotation effectively disabled; janitor keeps best ${KEEP_BEST_N})"
echo ">>> Epochs:          ${NUM_TRAIN_EPOCHS}"
echo ">>> Plugins:         ${EXTERNAL_PLUGINS}"

EXTRA_ARGS=()
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    echo ">>> Resume ckpt:     ${RESUME_FROM_CHECKPOINT}"
    if [ "${RESUME_ONLY_MODEL}" = "true" ]; then
        echo ">>> Resume mode:     adapters_only=true"
        EXTRA_ARGS+=(--adapters "${RESUME_FROM_CHECKPOINT}" --load_args false)
    else
        echo ">>> Resume mode:     resume_only_model=${RESUME_ONLY_MODEL}, ignore_data_skip=${IGNORE_DATA_SKIP}"
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
    --external_plugins ${EXTERNAL_PLUGINS} \
    --optimizer IMoptimizer \
    --eval_metric IMMetric \
    --metric_for_best_model loss \
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
    --load_best_model_at_end true \
    --logging_steps 5 \
    --max_length "${MAX_LENGTH}" \
    --output_dir "${OUTPUT_DIR}" \
    --add_version "${ADD_VERSION}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --deepspeed "${DEEPSPEED_CONFIG:-zero3}" \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --lr_scheduler_type cosine \
    --vit_lr "${VIT_LR}" \
    --aligner_lr "${ALIGNER_LR}" \
    --learning_rate "${LEARNING_RATE}" \
    --report_to none \
    --ignore_data_skip "${IGNORE_DATA_SKIP}" \
    --use_logits_to_keep true \
    "${EXTRA_ARGS[@]}"
