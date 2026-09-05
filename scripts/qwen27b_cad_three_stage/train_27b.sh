#!/usr/bin/env bash
# Two-phase LoRA SFT for the three-stage 2D-CAD dataset.
#
# Phase A keeps the vision tower frozen while the language/aligner LoRA learns
# the new JSON protocol.  Phase B continues from Phase A and opens the visual
# LoRA at a much smaller LR.  This script deliberately does not run SWA, DPO,
# GRPO, or select a checkpoint on the test set.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage:
  bash train_27b.sh preflight
  bash train_27b.sh phase_a
  PHASE_A_ADAPTER=/path/to/checkpoint bash train_27b.sh phase_b
  bash train_27b.sh both

Important environment variables:
  MODEL                    default: Qwen/Qwen3.6-27B
  TRAIN_FULL_DATASET       default: ./dataset/train_three_stage.jsonl
  TRAIN_CROP_DATASET       default: ./dataset/train_view_crops.jsonl
  VAL_DATASET              default: ./dataset/val_three_stage.jsonl
  PHASE_A_USE_CROP_DATA    default: false
  PHASE_B_USE_CROP_DATA    default: true
  PHASE_B_FREEZE_VIT       default: false; set true for protocol-only adaptation
  PHASE_B_FREEZE_ALIGNER   default: false; set true to preserve visual grounding
  USE_CROP_DATA            optional umbrella override for both phases
  OUTPUT_ROOT              default: ./outputs/qwen3.6-27b-cad-three-stage
  ALLOW_NONEMPTY_OUTPUT    default: false (resume explicitly instead)
  CUDA_VISIBLE_DEVICES     default: 0,1,2
  DEEPSPEED_CONFIG         default: zero3
  SKIP_SWIFT_HELP          default: false; true uses proven --tuner_type
  USE_LOGITS_TO_KEEP       default: auto; set true to reduce long-sequence logits

Qwen3.5 fallback:
  MODEL=Qwen/Qwen3.5-27B bash train_27b.sh both
EOF
}

ACTION="${1:-both}"
case "${ACTION}" in
    a|A|phase-a) ACTION="phase_a" ;;
    b|B|phase-b) ACTION="phase_b" ;;
    both|phase_a|phase_b|preflight) ;;
    -h|--help|help) usage; exit 0 ;;
    *) echo "ERROR: unknown action: ${ACTION}" >&2; usage >&2; exit 2 ;;
esac

# ---------------------------------------------------------------------------
# Configuration (all values remain explicitly overridable on a training pod)
# ---------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/dataset}"
TRAIN_FULL_DATASET="${TRAIN_FULL_DATASET:-${DATA_DIR}/train_three_stage.jsonl}"
TRAIN_CROP_DATASET="${TRAIN_CROP_DATASET:-${DATA_DIR}/train_view_crops.jsonl}"
VAL_DATASET="${VAL_DATASET:-${DATA_DIR}/val_three_stage.jsonl}"
# Curriculum defaults: Phase A learns the three-stage full-image protocol;
# Phase B adds high-resolution view crops. USE_CROP_DATA remains an optional
# umbrella override for older launch commands.
USE_CROP_DATA="${USE_CROP_DATA:-}"
PHASE_A_USE_CROP_DATA="${PHASE_A_USE_CROP_DATA:-${USE_CROP_DATA:-false}}"
PHASE_B_USE_CROP_DATA="${PHASE_B_USE_CROP_DATA:-${USE_CROP_DATA:-true}}"
PHASE_A_FREEZE_ALIGNER="${PHASE_A_FREEZE_ALIGNER:-false}"
PHASE_B_FREEZE_VIT="${PHASE_B_FREEZE_VIT:-false}"
PHASE_B_FREEZE_ALIGNER="${PHASE_B_FREEZE_ALIGNER:-false}"

MODEL_SLUG="${MODEL##*/}"
MODEL_SLUG="${MODEL_SLUG//[^[:alnum:]._-]/_}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/${MODEL_SLUG}-cad-three-stage}"
PHASE_A_OUTPUT="${PHASE_A_OUTPUT:-${OUTPUT_ROOT}/phase_a}"
PHASE_B_OUTPUT="${PHASE_B_OUTPUT:-${OUTPUT_ROOT}/phase_b}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export CUDA_VISIBLE_DEVICES
IFS=',' read -r -a _CUDA_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#_CUDA_DEVICES[@]}}"
MASTER_PORT="${MASTER_PORT:-29500}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-4096}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"
IFS=',' read -r -a TARGET_MODULE_ARRAY <<< "${TARGET_MODULES}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
EVAL_STEPS="${EVAL_STEPS:-50}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SEED="${SEED:-42}"
LOAD_BEST_MODEL_AT_END="${LOAD_BEST_MODEL_AT_END:-true}"

PHASE_A_EPOCHS="${PHASE_A_EPOCHS:-1.0}"
PHASE_A_LLM_LR="${PHASE_A_LLM_LR:-1e-5}"
PHASE_A_ALIGNER_LR="${PHASE_A_ALIGNER_LR:-5e-6}"
PHASE_A_VIT_LR="${PHASE_A_VIT_LR:-1e-6}"  # frozen; logged for reproducibility

PHASE_B_EPOCHS="${PHASE_B_EPOCHS:-1.5}"
PHASE_B_LLM_LR="${PHASE_B_LLM_LR:-6e-6}"
PHASE_B_ALIGNER_LR="${PHASE_B_ALIGNER_LR:-4e-6}"
PHASE_B_VIT_LR="${PHASE_B_VIT_LR:-1.5e-6}"

ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero3}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
ALLOW_CPU_TRAINING="${ALLOW_CPU_TRAINING:-false}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-true}"
RESUME_FROM_CHECKPOINT_A="${RESUME_FROM_CHECKPOINT_A:-}"
RESUME_FROM_CHECKPOINT_B="${RESUME_FROM_CHECKPOINT_B:-}"
PHASE_A_ADAPTER="${PHASE_A_ADAPTER:-}"
ALLOW_NONEMPTY_OUTPUT="${ALLOW_NONEMPTY_OUTPUT:-false}"
SKIP_SWIFT_HELP="${SKIP_SWIFT_HELP:-false}"
USE_LOGITS_TO_KEEP="${USE_LOGITS_TO_KEEP:-auto}"

if [[ -n "${SWIFT_BIN:-}" && -x "${SWIFT_BIN}" ]]; then
    SWIFT_BIN="${SWIFT_BIN}"
elif command -v swift >/dev/null 2>&1; then
    SWIFT_BIN="${SWIFT_BIN:-$(command -v swift)}"
elif [[ -x /opt/conda/bin/swift ]]; then
    SWIFT_BIN="${SWIFT_BIN:-/opt/conda/bin/swift}"
else
    echo "ERROR: ms-swift executable not found in PATH." >&2
    exit 127
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

preflight() {
    local paths=("${TRAIN_FULL_DATASET}" "${VAL_DATASET}")
    local need_crop="false"
    if [[ "${ACTION}" == "phase_a" && "${PHASE_A_USE_CROP_DATA}" == "true" ]]; then
        need_crop="true"
    elif [[ "${ACTION}" == "phase_b" && "${PHASE_B_USE_CROP_DATA}" == "true" ]]; then
        need_crop="true"
    elif [[ "${ACTION}" == "both" && ( "${PHASE_A_USE_CROP_DATA}" == "true" || "${PHASE_B_USE_CROP_DATA}" == "true" ) ]]; then
        need_crop="true"
    fi
    if [[ "${need_crop}" == "true" ]]; then
        paths+=("${TRAIN_CROP_DATASET}")
    fi
    local path
    for path in "${paths[@]}"; do
        if [[ ! -s "${path}" ]]; then
            echo "ERROR: required non-empty dataset is missing: ${path}" >&2
            return 2
        fi
    done

    MODEL_TO_CHECK="${MODEL}" \
    EXPECTED_GPU_COUNT="${NPROC_PER_NODE}" \
    TRAIN_DTYPE="${TORCH_DTYPE}" \
    ALLOW_CPU="${ALLOW_CPU_TRAINING}" \
    python3 - <<'PY'
import importlib
import importlib.metadata
import os
import sys

from packaging.version import Version


def package_version(distribution, module=None):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        if module:
            loaded = importlib.import_module(module)
            return getattr(loaded, "__version__", "0")
        raise


required_modules = ("torch", "transformers", "peft", "swift", "qwen_vl_utils")
failures = []
for name in required_modules:
    try:
        importlib.import_module(name)
    except Exception as exc:  # import-time CUDA/library errors matter here
        failures.append(f"cannot import {name}: {exc}")

model = os.environ["MODEL_TO_CHECK"]
if "Qwen3.6" in model:
    try:
        transformers_v = Version(package_version("transformers", "transformers"))
        if transformers_v < Version("5.0.0.dev0"):
            failures.append(
                f"Qwen3.6 requires transformers>=5.0.0.dev0, found {transformers_v}"
            )
    except Exception as exc:
        failures.append(f"cannot determine transformers version: {exc}")
    try:
        swift_v = Version(package_version("ms-swift", "swift"))
        if swift_v < Version("4.1.3"):
            failures.append(f"Qwen3.6 requires ms-swift>=4.1.3, found {swift_v}")
    except Exception as exc:
        failures.append(f"cannot determine ms-swift version: {exc}")
    try:
        qwen_utils_v = Version(package_version("qwen-vl-utils", "qwen_vl_utils"))
        if qwen_utils_v < Version("0.0.14"):
            failures.append(
                f"Qwen3.6 requires qwen-vl-utils>=0.0.14, found {qwen_utils_v}"
            )
    except Exception as exc:
        failures.append(f"cannot determine qwen-vl-utils version: {exc}")
    try:
        importlib.import_module("decord")
    except Exception as exc:
        failures.append(f"Qwen3.6 requires decord: {exc}")
    try:
        from transformers import AutoModelForMultimodalLM  # noqa: F401
    except Exception as exc:
        failures.append(f"AutoModelForMultimodalLM is unavailable: {exc}")

if not failures:
    import torch

    allow_cpu = os.environ.get("ALLOW_CPU", "false").lower() == "true"
    expected = int(os.environ.get("EXPECTED_GPU_COUNT", "1"))
    if not torch.cuda.is_available() and not allow_cpu:
        failures.append("CUDA is unavailable (set ALLOW_CPU_TRAINING=true only for a smoke test)")
    elif torch.cuda.is_available() and torch.cuda.device_count() < expected:
        failures.append(
            f"visible CUDA devices={torch.cuda.device_count()}, requested processes={expected}"
        )
    if (
        torch.cuda.is_available()
        and os.environ.get("TRAIN_DTYPE") == "bfloat16"
        and hasattr(torch.cuda, "is_bf16_supported")
        and not torch.cuda.is_bf16_supported()
    ):
        failures.append("selected GPUs do not support bfloat16")

if failures:
    print("Preflight FAILED:", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    raise SystemExit(2)

print("Preflight OK")
print(f"  model={model}")
for distribution, module in (
    ("ms-swift", "swift"),
    ("transformers", "transformers"),
    ("peft", "peft"),
    ("qwen-vl-utils", "qwen_vl_utils"),
):
    print(f"  {distribution}={package_version(distribution, module)}")
PY

    if ! python3 -c 'import flash_attn' >/dev/null 2>&1; then
        if [[ "${ATTN_IMPL}" == "flash_attn" || "${ATTN_IMPL}" == "flash_attention_2" ]]; then
            echo "WARN: flash_attn is unavailable; falling back to sdpa." >&2
            ATTN_IMPL="sdpa"
        fi
    fi
}

latest_adapter() {
    local root="$1"
    local adapter_config latest=""
    while IFS= read -r adapter_config; do
        latest="$(dirname "${adapter_config}")"
    done < <(find "${root}" -type f -name adapter_config.json -print 2>/dev/null | sort -V)
    printf '%s' "${latest}"
}

# Pick the modern flag when available, while keeping the script useful with an
# older ms-swift image used for Qwen3.5 regression experiments.
if [[ "${SKIP_SWIFT_HELP}" == "true" ]]; then
    # ms-swift 4.1.3 on the Nautilus image has a very expensive cold `--help`
    # import over CephFS.  --tuner_type is proven by the launch preflight/run.
    SWIFT_HELP=""
    TRAIN_TYPE_FLAG="--tuner_type"
else
    SWIFT_HELP="$(${SWIFT_BIN} sft --help 2>&1 || true)"
fi
if [[ "${SKIP_SWIFT_HELP}" != "true" ]] && grep -q -- '--train_type' <<< "${SWIFT_HELP}"; then
    TRAIN_TYPE_FLAG="--train_type"
elif [[ "${SKIP_SWIFT_HELP}" != "true" ]]; then
    TRAIN_TYPE_FLAG="--tuner_type"
fi

# Keep one universally supported entry so Bash 3.2 + `set -u` can safely expand
# the array even when none of the version-gated flags exists.
OPTIONAL_SWIFT_ARGS=(--report_to none)
if grep -q -- '--enable_thinking' <<< "${SWIFT_HELP}"; then
    OPTIONAL_SWIFT_ARGS+=(--enable_thinking false)
fi
if grep -q -- '--add_non_thinking_prefix' <<< "${SWIFT_HELP}"; then
    OPTIONAL_SWIFT_ARGS+=(--add_non_thinking_prefix true)
fi
if [[ "${USE_LOGITS_TO_KEEP}" != "auto" ]]; then
    OPTIONAL_SWIFT_ARGS+=(--use_logits_to_keep "${USE_LOGITS_TO_KEEP}")
elif grep -q -- '--use_logits_to_keep' <<< "${SWIFT_HELP}"; then
    OPTIONAL_SWIFT_ARGS+=(--use_logits_to_keep true)
fi
if grep -q -- '--strict' <<< "${SWIFT_HELP}"; then
    OPTIONAL_SWIFT_ARGS+=(--strict true)
fi

run_phase() {
    local phase="$1"
    local freeze_vit="$2"
    local epochs="$3"
    local llm_lr="$4"
    local vit_lr="$5"
    local aligner_lr="$6"
    local output_dir="$7"
    local init_adapter="$8"
    local resume_checkpoint="$9"
    local use_crop_data="${10}"
    local freeze_aligner="${11}"

    local train_datasets=("${TRAIN_FULL_DATASET}")
    if [[ "${use_crop_data}" == "true" ]]; then
        train_datasets+=("${TRAIN_CROP_DATASET}")
    fi

    local vit_gc="true"
    if [[ "${freeze_vit}" == "true" ]]; then
        vit_gc="false"
    fi

    local command=(
        "${SWIFT_BIN}" sft
        --model "${MODEL}"
        --dataset "${train_datasets[@]}"
        --val_dataset "${VAL_DATASET}"
        --external_plugins "${SCRIPT_DIR}/optimizer.py"
        --optimizer CADLayerwiseOptimizer
        "${TRAIN_TYPE_FLAG}" lora
        --target_modules "${TARGET_MODULE_ARRAY[@]}"
        --lora_rank "${LORA_RANK}"
        --lora_alpha "${LORA_ALPHA}"
        --lora_dropout "${LORA_DROPOUT}"
        --freeze_vit "${freeze_vit}"
        --freeze_aligner "${freeze_aligner}"
        --learning_rate "${llm_lr}"
        --vit_lr "${vit_lr}"
        --aligner_lr "${aligner_lr}"
        --weight_decay "${WEIGHT_DECAY}"
        --num_train_epochs "${epochs}"
        --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
        --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
        --gradient_checkpointing true
        --vit_gradient_checkpointing "${vit_gc}"
        --torch_dtype "${TORCH_DTYPE}"
        --attn_impl "${ATTN_IMPL}"
        --max_length "${MAX_LENGTH}"
        --packing false
        --truncation_strategy left
        --warmup_ratio "${WARMUP_RATIO}"
        --lr_scheduler_type cosine
        --eval_steps "${EVAL_STEPS}"
        --save_steps "${SAVE_STEPS}"
        --logging_steps "${LOGGING_STEPS}"
        --save_total_limit "${SAVE_TOTAL_LIMIT}"
        --load_best_model_at_end "${LOAD_BEST_MODEL_AT_END}"
        --metric_for_best_model loss
        --greater_is_better false
        --dataset_num_proc "${DATASET_NUM_PROC}"
        --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
        --deepspeed "${DEEPSPEED_CONFIG}"
        --output_dir "${output_dir}"
        --add_version false
        --seed "${SEED}"
        "${OPTIONAL_SWIFT_ARGS[@]}"
    )

    if [[ -n "${resume_checkpoint}" ]]; then
        command+=(--resume_from_checkpoint "${resume_checkpoint}")
    elif [[ -n "${init_adapter}" ]]; then
        command+=(--adapters "${init_adapter}" --load_args false)
    fi

    if [[ -d "${output_dir}" && -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" && -z "${resume_checkpoint}" && "${ALLOW_NONEMPTY_OUTPUT}" != "true" ]]; then
        echo "ERROR: output directory is non-empty: ${output_dir}" >&2
        echo "Use the matching RESUME_FROM_CHECKPOINT_A/B variable or choose a new output path; set ALLOW_NONEMPTY_OUTPUT=true only deliberately." >&2
        return 2
    fi
    mkdir -p "${output_dir}"
    local stamp log_file
    stamp="$(date '+%Y%m%d_%H%M%S')"
    log_file="${LOG_DIR}/${phase}_${stamp}.log"

    echo ">>> phase=${phase} model=${MODEL}"
    echo ">>> train=${train_datasets[*]}"
    echo ">>> val=${VAL_DATASET}"
    echo ">>> output=${output_dir} log=${log_file}"
    echo ">>> freeze_vit=${freeze_vit} freeze_aligner=${freeze_aligner} use_crop_data=${use_crop_data} epochs=${epochs} LRs(llm/vit/aligner)=${llm_lr}/${vit_lr}/${aligner_lr}"
    echo ">>> LoRA targets=${TARGET_MODULE_ARRAY[*]} r=${LORA_RANK} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT}"
    printf '>>> command:'
    printf ' %q' "${command[@]}"
    printf '\n'

    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    MASTER_PORT="${MASTER_PORT}" \
    USE_HF=1 \
    QWENVL_BBOX_FORMAT=new \
    IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "${command[@]}" 2>&1 | tee "${log_file}"
}

if [[ "${RUN_PREFLIGHT}" == "true" || "${ACTION}" == "preflight" ]]; then
    preflight
fi
if [[ "${ACTION}" == "preflight" ]]; then
    exit 0
fi

if [[ "${ACTION}" == "phase_a" || "${ACTION}" == "both" ]]; then
    run_phase \
        phase_a true "${PHASE_A_EPOCHS}" \
        "${PHASE_A_LLM_LR}" "${PHASE_A_VIT_LR}" "${PHASE_A_ALIGNER_LR}" \
        "${PHASE_A_OUTPUT}" "" "${RESUME_FROM_CHECKPOINT_A}" "${PHASE_A_USE_CROP_DATA}" \
        "${PHASE_A_FREEZE_ALIGNER}"
fi

if [[ "${ACTION}" == "phase_b" || "${ACTION}" == "both" ]]; then
    if [[ -z "${PHASE_A_ADAPTER}" && -z "${RESUME_FROM_CHECKPOINT_B}" ]]; then
        PHASE_A_ADAPTER="$(latest_adapter "${PHASE_A_OUTPUT}")"
    fi
    if [[ -z "${PHASE_A_ADAPTER}" && -z "${RESUME_FROM_CHECKPOINT_B}" ]]; then
        echo "ERROR: Phase B needs PHASE_A_ADAPTER or RESUME_FROM_CHECKPOINT_B." >&2
        exit 2
    fi
    run_phase \
        phase_b "${PHASE_B_FREEZE_VIT}" "${PHASE_B_EPOCHS}" \
        "${PHASE_B_LLM_LR}" "${PHASE_B_VIT_LR}" "${PHASE_B_ALIGNER_LR}" \
        "${PHASE_B_OUTPUT}" "${PHASE_A_ADAPTER}" "${RESUME_FROM_CHECKPOINT_B}" "${PHASE_B_USE_CROP_DATA}" \
        "${PHASE_B_FREEZE_ALIGNER}"
fi

echo ">>> Training action '${ACTION}' completed."
echo ">>> Select the production checkpoint with generation-based validation metrics; do not use test labels."
