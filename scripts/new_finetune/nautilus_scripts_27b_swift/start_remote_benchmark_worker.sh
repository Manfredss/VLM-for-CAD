#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export DATA_DIR="${DATA_DIR:-/workspace/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/swift_27b}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-27B}"
export BENCHMARK_CUDA_VISIBLE_DEVICES="${BENCHMARK_CUDA_VISIBLE_DEVICES:-2,3}"
export BENCHMARK_SIZE="${BENCHMARK_SIZE:-50}"
export BENCHMARK_POLL_INTERVAL="${BENCHMARK_POLL_INTERVAL:-180}"
export BENCHMARK_MAX_NEW_TOKENS="${BENCHMARK_MAX_NEW_TOKENS:-4096}"
export BENCHMARK_IOU_THRESHOLD="${BENCHMARK_IOU_THRESHOLD:-0.4}"
export BENCHMARK_LOAD_IN_4BIT="${BENCHMARK_LOAD_IN_4BIT:-false}"
export BENCHMARK_MIN_MTIME="${BENCHMARK_MIN_MTIME:-$(date +%s)}"
export RUN_TAG="${RUN_TAG:-swift_27b_v13_remote}"
export CANONICAL_BENCHMARK_LOG="${CANONICAL_BENCHMARK_LOG:-/workspace/logs/train_swift_v13_benchmark.log}"

ensure_benchmark_deps() {
    if python3 - <<'PY' >/dev/null 2>&1
import importlib.util
mods = ['transformers', 'accelerate', 'peft', 'bitsandbytes', 'qwen_vl_utils']
raise SystemExit(0 if all(importlib.util.find_spec(mod) for mod in mods) else 1)
PY
    then
        echo ">>> Benchmark dependencies OK"
        return 0
    fi

    echo ">>> Installing benchmark dependencies..."
    python3 -m pip install -U transformers accelerate peft bitsandbytes qwen-vl-utils ninja
}

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate base
elif [ -f "/workspace/venv2/bin/activate" ]; then
    source /workspace/venv2/bin/activate
elif [ -f "/workspace/venv/bin/activate" ]; then
    source /workspace/venv/bin/activate
fi

if [ -d "$DATA_DIR/5k" ]; then
    IMAGE_DIR="$DATA_DIR/5k"
elif [ -d "$DATA_DIR/images" ]; then
    IMAGE_DIR="$DATA_DIR/images"
elif [ -d "$DATA_DIR/IM_D03_PT_5K" ]; then
    IMAGE_DIR="$DATA_DIR/IM_D03_PT_5K"
else
    echo ">>> ERROR: No image directory found under $DATA_DIR"
    exit 1
fi

if [ -f "$DATA_DIR/benchmarkdata_swift4.jsonl" ]; then
    VAL_FILE="$DATA_DIR/benchmarkdata_swift4.jsonl"
elif [ -f "$DATA_DIR/val_swift4.jsonl" ]; then
    VAL_FILE="$DATA_DIR/val_swift4.jsonl"
elif [ -f "$DATA_DIR/val_augmented.jsonl" ]; then
    VAL_FILE="$DATA_DIR/val_augmented.jsonl"
else
    VAL_FILE="$DATA_DIR/val.jsonl"
fi

mkdir -p /workspace/logs "$OUTPUT_DIR"

BENCHMARK_LOG="/workspace/logs/${RUN_TAG}.benchmark.log"
ln -sfn "$BENCHMARK_LOG" "$CANONICAL_BENCHMARK_LOG"

ensure_benchmark_deps

echo ">>> Remote benchmark worker starting"
echo ">>> Output dir:     $OUTPUT_DIR"
echo ">>> Validation set: $VAL_FILE"
echo ">>> Image dir:      $IMAGE_DIR"
echo ">>> Benchmark GPUs: $BENCHMARK_CUDA_VISIBLE_DEVICES"
echo ">>> Min checkpoint mtime: $BENCHMARK_MIN_MTIME"
echo ">>> Benchmark log:  $CANONICAL_BENCHMARK_LOG"

CUDA_VISIBLE_DEVICES="$BENCHMARK_CUDA_VISIBLE_DEVICES" \
OUTPUT_DIR="$OUTPUT_DIR" \
MODEL_PATH="$MODEL_NAME" \
VAL_DATASET="$VAL_FILE" \
IMAGE_DIR="$IMAGE_DIR" \
BENCHMARK_SIZE="$BENCHMARK_SIZE" \
BENCHMARK_POLL_INTERVAL="$BENCHMARK_POLL_INTERVAL" \
BENCHMARK_MAX_NEW_TOKENS="$BENCHMARK_MAX_NEW_TOKENS" \
BENCHMARK_IOU_THRESHOLD="$BENCHMARK_IOU_THRESHOLD" \
BENCHMARK_LOAD_IN_4BIT="$BENCHMARK_LOAD_IN_4BIT" \
BENCHMARK_MIN_MTIME="$BENCHMARK_MIN_MTIME" \
bash "$SCRIPT_DIR/watch_checkpoints_and_benchmark.sh" | tee -a "$BENCHMARK_LOG"