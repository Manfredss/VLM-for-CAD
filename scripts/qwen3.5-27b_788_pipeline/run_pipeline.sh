#!/bin/bash
# ================================================================
# Full Pipeline Orchestrator — Siemens 788 Drawing Feature Detection
#
# Stages:
#   Stage 1 (SFT):  prepare_dataset → train_sft → swa → evaluate_checkpoints
#   Stage 2 (DPO):  generate predictions → score → make pairs → train_dpo  [optional]
#   Stage 3 (GRPO): prepare GRPO data → train_grpo                           [optional]
#   Stage 4 (Agentic RL): generate multi-pass data → train_agentic_grpo     [optional]
#
# Each stage can be run independently by setting STAGE=1|2|3|4.
# Set STAGE=all to run the full pipeline sequentially.
#
# Usage:
#   # Run everything
#   bash run_pipeline.sh all
#
#   # Run only SFT
#   bash run_pipeline.sh 1
#
#   # Run GRPO after SFT is done
#   bash run_pipeline.sh 3
#
#   # Custom paths
#   OUTPUT_DIR=/workspace/output/my_experiment \
#   IMAGE_DIR=/workspace/data/my_images \
#   bash run_pipeline.sh all
# ================================================================

set -euo pipefail

STAGE="${1:-all}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d /workspace/venv/bin ]; then
    export PATH="/workspace/venv/bin:${PATH}"
fi

# ================================================================
# Global config (overridable via env)
# ================================================================
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/swift_27b_788_pipeline}"
IMAGE_DIR="${IMAGE_DIR:-/workspace/data/simens_7feats}"
DATA_DIR="${DATA_DIR:-/workspace/data}"
ANNOTATION_JSON="${ANNOTATION_JSON:-${SCRIPT_DIR}/788_11Feats_View.json}"
TRAIN_JSONL="${TRAIN_JSONL:-${SCRIPT_DIR}/dataset/train_view_7feats_pipeline.jsonl}"
VAL_JSONL="${VAL_JSONL:-${SCRIPT_DIR}/dataset/val_view_7feats_pipeline.jsonl}"
TEST_JSONL="${TEST_JSONL:-${SCRIPT_DIR}/dataset/test_view_7feats_pipeline.jsonl}"

# GPU config
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2}"
INFERENCE_GPUS="${INFERENCE_GPUS:-0,1}"

# Stage control
RUN_SFT="${RUN_SFT:-true}"
RUN_SWA="${RUN_SWA:-true}"
RUN_CKPT_EVAL="${RUN_CKPT_EVAL:-true}"
RUN_DPO="${RUN_DPO:-false}"
RUN_GRPO="${RUN_GRPO:-false}"
RUN_AGENTIC="${RUN_AGENTIC:-false}"

# ================================================================
# Helper: find the best adapter path
# ================================================================
_find_swa_adapter() {
    local dir="${1:-${OUTPUT_DIR}}"
    # Try swa_adapter first
    local swa=$(find "$dir" -name "swa_adapter" -type d 2>/dev/null | head -1)
    if [ -n "$swa" ]; then
        echo "$swa"
        return
    fi
    # Fall back to final_adapter
    local fa=$(find "$dir" -name "final_adapter" -type d 2>/dev/null | head -1)
    if [ -n "$fa" ]; then
        echo "$fa"
        return
    fi
    # Last resort: latest checkpoint
    local ckpt=$(find "$dir" -name "checkpoint-*" -type d 2>/dev/null | sort -t- -k2 -n | tail -1)
    echo "$ckpt"
}

# ================================================================
# Stage 1: SFT
# ================================================================
run_stage1() {
    echo ""
    echo "================================================================"
    echo "  STAGE 1: Supervised Fine-Tuning (SFT)"
    echo "================================================================"
    echo ""

    # 1a. Prepare dataset
    echo "--- [1a] Preparing dataset ---"
    python3 "${SCRIPT_DIR}/prepare_dataset_swift.py" \
        --json_path "${ANNOTATION_JSON}" \
        --image_dir "${IMAGE_DIR}" \
        --deploy_image_dir "${IMAGE_DIR}" \
        --output_dir "${SCRIPT_DIR}/dataset" \
        --prefix "view_7feats_pipeline"

    # Copy to workspace data dir if different
    if [ "${DATA_DIR}" != "${SCRIPT_DIR}/dataset" ]; then
        mkdir -p "${DATA_DIR}"
        cp "${TRAIN_JSONL}" "${DATA_DIR}/" 2>/dev/null || true
        cp "${VAL_JSONL}" "${DATA_DIR}/" 2>/dev/null || true
        cp "${TEST_JSONL}" "${DATA_DIR}/" 2>/dev/null || true
        echo "Datasets copied to ${DATA_DIR}"
    fi

    # 1b. Train SFT
    echo ""
    echo "--- [1b] Training SFT ---"
    TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    TRAIN_DATASET="${DATA_DIR}/train_view_7feats_pipeline.jsonl" \
    VAL_DATASET="${DATA_DIR}/val_view_7feats_pipeline.jsonl" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    bash "${SCRIPT_DIR}/train_sft.sh"

    # 1c. SWA
    if [ "${RUN_SWA}" = "true" ]; then
        echo ""
        echo "--- [1c] Running SWA ---"
        python3 "${SCRIPT_DIR}/swa.py" \
            --output-dir "${OUTPUT_DIR}" \
            --top-n 3
    fi

    # 1d. Evaluate checkpoints
    if [ "${RUN_CKPT_EVAL}" = "true" ]; then
        echo ""
        echo "--- [1d] Evaluating checkpoints ---"
        python3 "${SCRIPT_DIR}/evaluate_checkpoints.py" \
            --output-dir "${OUTPUT_DIR}" \
            --val-jsonl "${VAL_JSONL}" \
            --image-dir "${IMAGE_DIR}" \
            --n-samples 50
    fi

    echo ""
    echo "=== Stage 1 (SFT) complete ==="
    SFT_ADAPTER=$(_find_swa_adapter "${OUTPUT_DIR}")
    echo "Best adapter: ${SFT_ADAPTER}"
    export SFT_ADAPTER
}

# ================================================================
# Stage 2: DPO (optional)
# ================================================================
run_stage2() {
    echo ""
    echo "================================================================"
    echo "  STAGE 2: Direct Preference Optimization (DPO)"
    echo "================================================================"
    echo ""

    local adapter="${SFT_ADAPTER:-$(_find_swa_adapter)}"
    if [ -z "$adapter" ]; then
        echo "ERROR: No SFT adapter found. Run Stage 1 first."
        return 1
    fi
    echo "Using adapter: ${adapter}"

    local dpo_dir="${OUTPUT_DIR}_dpo"
    local samples_jsonl="${dpo_dir}/samples.jsonl"
    local scored_jsonl="${dpo_dir}/scored.jsonl"
    local pairs_jsonl="${DATA_DIR}/dpo_view_7feats_pipeline.jsonl"

    mkdir -p "${dpo_dir}"

    # 2a. Generate predictions with temperature sampling
    echo ""
    echo "--- [2a] Generating predictions (K=4 per image) ---"
    python3 "${SCRIPT_DIR}/dpo/generate_predictions.py" \
        --adapter "${adapter}" \
        --train_jsonl "${TRAIN_JSONL}" \
        --image_dir "${IMAGE_DIR}" \
        --output "${samples_jsonl}" \
        --k 4 \
        --temperature 0.8

    # 2b. Score predictions
    echo ""
    echo "--- [2b] Scoring predictions ---"
    python3 "${SCRIPT_DIR}/dpo/score_predictions.py" \
        --samples "${samples_jsonl}" \
        --train_jsonl "${TRAIN_JSONL}" \
        --output "${scored_jsonl}"

    # 2c. Make DPO pairs
    echo ""
    echo "--- [2c] Creating DPO pairs ---"
    python3 "${SCRIPT_DIR}/dpo/make_dpo_jsonl.py" \
        --scored "${scored_jsonl}" \
        --train_jsonl "${TRAIN_JSONL}" \
        --output "${pairs_jsonl}" \
        --min_score_gap 0.05

    # 2d. Train DPO
    echo ""
    echo "--- [2d] Training DPO ---"
    TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    ADAPTER="${adapter}" \
    DPO_DATASET="${pairs_jsonl}" \
    OUTPUT_DIR="${dpo_dir}" \
    bash "${SCRIPT_DIR}/dpo/train_dpo.sh"

    echo ""
    echo "=== Stage 2 (DPO) complete ==="
}

# ================================================================
# Stage 3: GRPO (optional)
# ================================================================
run_stage3() {
    echo ""
    echo "================================================================"
    echo "  STAGE 3: Group Relative Policy Optimization (GRPO)"
    echo "================================================================"
    echo ""

    local adapter="${SFT_ADAPTER:-$(_find_swa_adapter)}"
    if [ -z "$adapter" ]; then
        echo "ERROR: No SFT adapter found. Run Stage 1 first."
        return 1
    fi
    echo "Using adapter: ${adapter}"

    local grpo_dir="${OUTPUT_DIR}_grpo"

    # GRPO uses the same training JSONL (prompts + GT)
    echo ""
    echo "--- [3a] Training GRPO ---"
    TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    SFT_CHECKPOINT="${adapter}" \
    TRAIN_DATASET="${TRAIN_JSONL}" \
    VAL_DATASET="${VAL_JSONL}" \
    OUTPUT_DIR="${grpo_dir}" \
    bash "${SCRIPT_DIR}/train_grpo.sh"

    echo ""
    echo "=== Stage 3 (GRPO) complete ==="
    echo "GRPO adapter: ${grpo_dir}"
}

# ================================================================
# Stage 4: Agentic RL (experimental)
# ================================================================
run_stage4() {
    echo ""
    echo "================================================================"
    echo "  STAGE 4: Agentic RL (Self-Refinement GRPO)"
    echo "================================================================"
    echo ""

    local adapter="${SFT_ADAPTER:-$(_find_swa_adapter)}"
    if [ -z "$adapter" ]; then
        echo "ERROR: No SFT adapter found. Run Stage 1 first."
        return 1
    fi
    echo "Using adapter: ${adapter}"

    local agentic_dir="${OUTPUT_DIR}_agentic"
    local agentic_results="${agentic_dir}/agentic_train_results.json"
    mkdir -p "${agentic_dir}"

    # 4a. Generate agentic training data (multi-pass self-refinement)
    echo ""
    echo "--- [4a] Generating agentic training data ---"
    CUDA_VISIBLE_DEVICES="${INFERENCE_GPUS}" \
    python3 "${SCRIPT_DIR}/agentic_rl_inference.py" \
        --adapter_path "${adapter}" \
        --test_jsonl "${TRAIN_JSONL}" \
        --image_dir "${IMAGE_DIR}" \
        --output_file "${agentic_results}" \
        --temperature 0.7 \
        --max_new_tokens 2048

    # 4b. Convert to GRPO-compatible format (full multi-turn conversation)
    echo ""
    echo "--- [4b] Preparing agentic GRPO data ---"
    python3 -c "
import json
from pathlib import Path

results = json.load(open('${agentic_results}'))
train = []
with open('${TRAIN_JSONL}') as f:
    for line in f:
        line = line.strip()
        if line:
            train.append(json.loads(line))

by_name = {Path(r['images'][0]).name: r for r in train}
agentic_jsonl = '${agentic_dir}/agentic_train.jsonl'

with open(agentic_jsonl, 'w') as out:
    for rec in results:
        name = rec['dataitem_name']
        if name not in by_name:
            continue
        t = by_name[name]
        # Build full agentic conversation: system + user1 + views + features + critique + refined
        msgs = []
        msgs.append(t['messages'][0])  # system
        msgs.append(t['messages'][1])  # user + image + step1
        msgs.append({'role': 'assistant', 'content': rec.get('raw_views', '[]')})
        msgs.append(t['messages'][3])  # user step2
        msgs.append({'role': 'assistant', 'content': rec.get('raw_features', '[]')})
        msgs.append({'role': 'user', 'content': '请审查并修正以上检测结果。'})
        msgs.append({'role': 'assistant', 'content': rec.get('raw_refined', rec.get('raw_features', '[]'))})
        out.write(json.dumps({'messages': msgs, 'images': t['images']}, ensure_ascii=False) + '\n')

print(f'Agentic GRPO data: {len(results)} samples -> {agentic_jsonl}')
"

    # 4c. Train agentic GRPO
    echo ""
    echo "--- [4c] Training agentic GRPO ---"
    TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    SFT_CHECKPOINT="${adapter}" \
    TRAIN_DATASET="${agentic_dir}/agentic_train.jsonl" \
    OUTPUT_DIR="${agentic_dir}" \
    bash "${SCRIPT_DIR}/train_agentic_grpo.sh"

    echo ""
    echo "=== Stage 4 (Agentic RL) complete ==="
    echo "Agentic adapter: ${agentic_dir}"
}

# ================================================================
# Main
# ================================================================
echo ">>> Siemens 788 Pipeline Orchestrator"
echo ">>> Script dir:  ${SCRIPT_DIR}"
echo ">>> Output dir:  ${OUTPUT_DIR}"
echo ">>> Image dir:   ${IMAGE_DIR}"
echo ">>> Data dir:    ${DATA_DIR}"
echo ">>> Stage:       ${STAGE}"
echo ""

case "${STAGE}" in
    all)
        run_stage1
        if [ "${RUN_DPO}" = "true" ]; then
            run_stage2
        fi
        if [ "${RUN_GRPO}" = "true" ]; then
            run_stage3
        fi
        if [ "${RUN_AGENTIC}" = "true" ]; then
            run_stage4
        fi
        ;;
    1) run_stage1 ;;
    2) run_stage2 ;;
    3) run_stage3 ;;
    4) run_stage4 ;;
    *)
        echo "Usage: bash run_pipeline.sh {all|1|2|3|4}"
        echo ""
        echo "Stages:"
        echo "  1    SFT (prepare data + train + SWA + checkpoint eval)"
        echo "  2    DPO (generate samples + score + pair + train)"
        echo "  3    GRPO (standard reward-based RL)"
        echo "  4    Agentic RL (self-refinement GRPO)"
        echo "  all  Run stages 1-4 with RUN_DPO/RUN_GRPO/RUN_AGENTIC flags"
        exit 1
        ;;
esac

echo ""
echo "================================================================"
echo "  Pipeline complete."
echo "  Output: ${OUTPUT_DIR}"
echo "================================================================"
