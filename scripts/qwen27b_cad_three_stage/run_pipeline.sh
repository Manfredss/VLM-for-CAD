#!/usr/bin/env bash
# Reproducible entry point: prepare -> audit -> train -> infer/evaluate.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
    cat <<'EOF'
Usage: bash run_pipeline.sh <prepare|audit|preflight|train|infer|eval|all>

The source paths, deployment paths, model, adapter and output directories are
configured with environment variables. See README.md for copy/paste examples.
"all" performs prepare, audit, preflight and 27B training. Test inference/eval
requires a separately validation-selected ADAPTER_PATH and is intentionally run
as a later `eval` action.
EOF
}

ACTION="${1:-}"
case "${ACTION}" in
    prepare|audit|preflight|train|infer|eval|all) ;;
    -h|--help|help|"") usage; exit 0 ;;
    *) echo "ERROR: unknown action: ${ACTION}" >&2; usage >&2; exit 2 ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
DATASET_DIR="${DATASET_DIR:-${SCRIPT_DIR}/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/${MODEL##*/}-cad-three-stage}"
PREDICTIONS="${PREDICTIONS:-${OUTPUT_ROOT}/predictions_three_stage.jsonl}"
METRICS_OUTPUT="${METRICS_OUTPUT:-${OUTPUT_ROOT}/metrics.json}"

# Repository defaults are provided where the checked-in source is unambiguous.
# Image/deployment paths are commonly different on a GPU pod and remain
# overridable. The merged workflow requires all three complementary sources.
JSON_10="${JSON_10:-${PROJECT_ROOT}/data/5k_10feats_v2_augmented.json}"
JSON_15="${JSON_15:-${PROJECT_ROOT}/data/5k_15feats_with_view_v2.json}"
JSON_788="${JSON_788:-${PROJECT_ROOT}/scripts/qwen3.5-27b_788_pipeline/788_11Feats_View.json}"
IMAGE_DIR_10="${IMAGE_DIR_10:-/workspace/data/IM_D03_PT_5k_Augmented}"
IMAGE_DIR_15="${IMAGE_DIR_15:-/workspace/data/IM_D03_PT_5K}"
IMAGE_DIR_788="${IMAGE_DIR_788:-${PROJECT_ROOT}/scripts/qwn3.5-27b_788_silver_plate_bend/simens_7feats}"
DEPLOY_DIR_10="${DEPLOY_DIR_10:-${IMAGE_DIR_10}}"
DEPLOY_DIR_15="${DEPLOY_DIR_15:-${IMAGE_DIR_15}}"
DEPLOY_DIR_788="${DEPLOY_DIR_788:-${IMAGE_DIR_788}}"

PROJECTION_10="${PROJECTION_10:-unknown}"
PROJECTION_15="${PROJECTION_15:-unknown}"
PROJECTION_788="${PROJECTION_788:-first_angle}"
# Empty means: use build_merged_dataset.py's audited source-specific defaults.
SCOPE_10="${SCOPE_10:-}"
SCOPE_15="${SCOPE_15:-}"
SCOPE_788="${SCOPE_788:-}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
SEED="${SEED:-42}"
MATERIALIZE_CROPS="${MATERIALIZE_CROPS:-true}"
CROP_DIR="${CROP_DIR:-${DATASET_DIR}/view_crops}"
CROP_DEPLOY_DIR="${CROP_DEPLOY_DIR:-${CROP_DIR}}"
CROP_PADDING="${CROP_PADDING:-0.12}"
MAX_EMPTY_CROPS_PER_IMAGE="${MAX_EMPTY_CROPS_PER_IMAGE:-1}"
IMAGE_WORKERS="${IMAGE_WORKERS:-8}"
CROP_WORKERS="${CROP_WORKERS:-8}"
STRICT_MISSING_IMAGES="${STRICT_MISSING_IMAGES:-true}"
AUDIT_STRICT="${AUDIT_STRICT:-true}"
EXPECTED_GPUS="${EXPECTED_GPUS:-3}"

TRAIN_PHASE="${TRAIN_PHASE:-both}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
FEATURE_MODE="${FEATURE_MODE:-hybrid}"
INFER_PROJECTION_METHOD="${INFER_PROJECTION_METHOD:-auto}"
RUN_INFERENCE_FOR_EVAL="${RUN_INFERENCE_FOR_EVAL:-true}"
EVAL_STRICT="${EVAL_STRICT:-true}"

mkdir -p "${DATASET_DIR}" "${OUTPUT_ROOT}"

run_prepare() {
    local command=(
        "${PYTHON_BIN}" "${SCRIPT_DIR}/build_merged_dataset.py"
        --output-dir "${DATASET_DIR}"
        --train-ratio "${TRAIN_RATIO}"
        --val-ratio "${VAL_RATIO}"
        --test-ratio "${TEST_RATIO}"
        --seed "${SEED}"
        --crop-padding "${CROP_PADDING}"
        --max-empty-crops-per-image "${MAX_EMPTY_CROPS_PER_IMAGE}"
        --image-workers "${IMAGE_WORKERS}"
        --crop-workers "${CROP_WORKERS}"
    )
    local source_json
    for source_json in "${JSON_10}" "${JSON_15}" "${JSON_788}"; do
        if [[ -z "${source_json}" ]]; then
            echo "ERROR: JSON_10, JSON_15 and JSON_788 are all required." >&2
            return 2
        fi
    done
    command+=(
        --json-10 "${JSON_10}" --image-dir-10 "${IMAGE_DIR_10}"
        --deploy-dir-10 "${DEPLOY_DIR_10}" --projection-10 "${PROJECTION_10}"
        --json-15 "${JSON_15}" --image-dir-15 "${IMAGE_DIR_15}"
        --deploy-dir-15 "${DEPLOY_DIR_15}" --projection-15 "${PROJECTION_15}"
        --json-788 "${JSON_788}" --image-dir-788 "${IMAGE_DIR_788}"
        --deploy-dir-788 "${DEPLOY_DIR_788}" --projection-788 "${PROJECTION_788}"
    )
    [[ -n "${SCOPE_10}" ]] && command+=(--scope-10 "${SCOPE_10}")
    [[ -n "${SCOPE_15}" ]] && command+=(--scope-15 "${SCOPE_15}")
    [[ -n "${SCOPE_788}" ]] && command+=(--scope-788 "${SCOPE_788}")
    if [[ "${MATERIALIZE_CROPS}" == "true" ]]; then
        command+=(
            --materialize-crops --crop-dir "${CROP_DIR}"
            --crop-deploy-dir "${CROP_DEPLOY_DIR}"
        )
    else
        command+=(--no-materialize-crops)
    fi
    if [[ "${STRICT_MISSING_IMAGES}" == "true" ]]; then
        command+=(--strict-missing-images)
    fi
    echo ">>> PREPARE: output=${DATASET_DIR} materialize_crops=${MATERIALIZE_CROPS}"
    "${command[@]}"
}

run_audit() {
    local command=(
        "${PYTHON_BIN}" "${SCRIPT_DIR}/audit_dataset.py"
        --train "${DATASET_DIR}/train_ground_truth.jsonl"
        --val "${DATASET_DIR}/val_ground_truth.jsonl"
        --test "${DATASET_DIR}/test_ground_truth.jsonl"
        --json-out "${DATASET_DIR}/audit_report.json"
    )
    if [[ "${AUDIT_STRICT}" == "true" ]]; then
        command+=(--strict)
    fi
    if [[ "${STRICT_MISSING_IMAGES}" == "true" ]]; then
        command+=(--require-images)
    fi
    echo ">>> AUDIT: strict=${AUDIT_STRICT}"
    "${command[@]}"
}

run_preflight() {
    if [[ -f "${SCRIPT_DIR}/preflight.py" ]]; then
        echo ">>> PREFLIGHT: model=${MODEL}"
        "${PYTHON_BIN}" "${SCRIPT_DIR}/preflight.py" \
            --model "${MODEL}" \
            --expected-gpus "${EXPECTED_GPUS}" \
            --json-out "${OUTPUT_ROOT}/preflight_report.json" \
            --strict
    else
        MODEL="${MODEL}" \
        DATA_DIR="${DATASET_DIR}" \
        PHASE_A_USE_CROP_DATA=false \
        PHASE_B_USE_CROP_DATA="${MATERIALIZE_CROPS}" \
        bash "${SCRIPT_DIR}/train_27b.sh" preflight
    fi
}

run_train() {
    echo ">>> TRAIN: phase=${TRAIN_PHASE} model=${MODEL}"
    MODEL="${MODEL}" \
    DATA_DIR="${DATASET_DIR}" \
    TRAIN_FULL_DATASET="${DATASET_DIR}/train_three_stage.jsonl" \
    TRAIN_CROP_DATASET="${DATASET_DIR}/train_view_crops.jsonl" \
    VAL_DATASET="${DATASET_DIR}/val_three_stage.jsonl" \
    PHASE_A_USE_CROP_DATA=false \
    PHASE_B_USE_CROP_DATA="${MATERIALIZE_CROPS}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    bash "${SCRIPT_DIR}/train_27b.sh" "${TRAIN_PHASE}"
}

resolve_adapter() {
    if [[ -z "${ADAPTER_PATH}" ]]; then
        echo "ERROR: set ADAPTER_PATH to a validation-selected checkpoint." >&2
        return 2
    fi
}

run_infer() {
    resolve_adapter
    echo ">>> INFER: adapter=${ADAPTER_PATH} mode=${FEATURE_MODE}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/infer_three_stage.py" \
        --model-path "${MODEL}" \
        --adapter-path "${ADAPTER_PATH}" \
        --input-jsonl "${DATASET_DIR}/test_three_stage.jsonl" \
        --output-file "${PREDICTIONS}" \
        --feature-mode "${FEATURE_MODE}" \
        --projection-method "${INFER_PROJECTION_METHOD}"
}

run_eval() {
    if [[ "${RUN_INFERENCE_FOR_EVAL}" == "true" ]]; then
        # infer_three_stage.py resumes a partial JSONL and skips completed rows.
        run_infer
    elif [[ ! -s "${PREDICTIONS}" ]]; then
        echo "ERROR: predictions missing: ${PREDICTIONS}" >&2
        return 2
    fi
    echo ">>> EVAL: predictions=${PREDICTIONS}"
    local command=(
        "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_predictions.py"
        --ground-truth "${DATASET_DIR}/test_ground_truth.jsonl"
        --predictions "${PREDICTIONS}"
        --output "${METRICS_OUTPUT}"
    )
    if [[ "${EVAL_STRICT}" == "true" ]]; then
        command+=(--strict)
    fi
    "${command[@]}"
}

case "${ACTION}" in
    prepare) run_prepare ;;
    audit) run_audit ;;
    preflight) run_preflight ;;
    train) run_train ;;
    infer) run_infer ;;
    eval) run_eval ;;
    all)
        run_prepare
        run_audit
        run_preflight
        run_train
        echo ">>> Training finished. Select ADAPTER_PATH using validation generation, then run 'eval'."
        ;;
esac

echo ">>> Pipeline action '${ACTION}' completed."
