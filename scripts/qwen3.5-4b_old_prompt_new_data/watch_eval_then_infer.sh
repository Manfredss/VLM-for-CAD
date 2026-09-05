#!/bin/bash
# ================================================================
# Watch for new checkpoints and run eval + inference
# Qwen3.5-4B with OLD prompt on qwf-workspace
#
# Usage:
#   nohup bash watch_eval_then_infer.sh > /workspace/finetune/logs/watch_4b_oldprompt.log 2>&1 &
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-/workspace/finetune/output/swift_4b_oldprompt}"
BENCHMARK_DATASET="${BENCHMARK_DATASET:-/workspace/finetune/data/data/benchmark_10feats_oldprompt_4b.jsonl}"
GT_JSON="${GT_JSON:-/workspace/finetune/data/data/5k_10feats_v2_sampled_100.json}"
IMAGE_DIR="${IMAGE_DIR:-/workspace/finetune/data/data/5k_10feats_v2_sampled_100}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
POLL_INTERVAL="${POLL_INTERVAL:-120}"
MIN_MTIME="${MIN_MTIME:-0}"
DONE_FILE="${OUTPUT_DIR}/.done_eval_infer_4b_oldprompt"

BENCHMARK_RESULTS="${OUTPUT_DIR}/benchmark_results_4b_oldprompt.jsonl"
SAMPLED100_DIR="${OUTPUT_DIR}/sampled100_results_4b_oldprompt"

mkdir -p "$SAMPLED100_DIR"
touch "$DONE_FILE"

echo "============================================="
echo " Checkpoint Watcher — 4B Old Prompt"
echo " Output dir:  $OUTPUT_DIR"
echo " Benchmark:   $BENCHMARK_DATASET"
echo " GT JSON:     $GT_JSON"
echo " Image dir:   $IMAGE_DIR"
echo " Poll:        ${POLL_INTERVAL}s"
echo "============================================="

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate base
elif [ -f "/workspace/venv/bin/activate" ]; then
    source /workspace/venv/bin/activate
fi

while true; do
    for ckpt_dir in "$OUTPUT_DIR"/*/checkpoint-*; do
        [ -d "$ckpt_dir" ] || continue
        ckpt_name=$(basename "$ckpt_dir")
        version_dir=$(basename "$(dirname "$ckpt_dir")")
        full_name="${version_dir}/${ckpt_name}"

        if grep -qF "$full_name" "$DONE_FILE" 2>/dev/null; then
            continue
        fi

        if [ "$MIN_MTIME" -gt 0 ]; then
            dir_mtime=$(stat -c %Y "$ckpt_dir" 2>/dev/null || stat -f %m "$ckpt_dir" 2>/dev/null || echo 0)
            if [ "$dir_mtime" -lt "$MIN_MTIME" ]; then
                continue
            fi
        fi

        echo ""
        echo ">>> Found: $full_name ($(date))"

        echo ">>> [1/2] Benchmark eval..."
        BENCH_OUT="${OUTPUT_DIR}/benchmark_${ckpt_name}_4b_oldprompt.json"
        if python3 -u "$SCRIPT_DIR/evaluate_checkpoints_benchmark.py" \
            --adapter_path "$ckpt_dir" \
            --benchmark_jsonl "$BENCHMARK_DATASET" \
            --output_file "$BENCH_OUT" \
            --base_model "$BASE_MODEL" \
            --max_new_tokens "$MAX_NEW_TOKENS"; then
            METRICS=$(python3 -c "
import json
with open('$BENCH_OUT') as f: d = json.load(f)
o = d['overall']
print(json.dumps({'checkpoint': '$full_name', **o}))
")
            echo "$METRICS" >> "$BENCHMARK_RESULTS"
            echo ">>> Benchmark: $METRICS"
        else
            echo ">>> Benchmark FAILED"
        fi

        echo ">>> [2/2] Sampled-100 inference..."
        INFER_OUT="${SAMPLED100_DIR}/${ckpt_name}_results.json"
        if python3 -u "$SCRIPT_DIR/infer_sampled100.py" \
            --adapter_path "$ckpt_dir" \
            --gt_json "$GT_JSON" \
            --image_dir "$IMAGE_DIR" \
            --output_file "$INFER_OUT" \
            --base_model "$BASE_MODEL" \
            --max_new_tokens "$MAX_NEW_TOKENS"; then
            echo ">>> Inference done: $INFER_OUT"
        else
            echo ">>> Inference FAILED"
        fi

        echo "$full_name" >> "$DONE_FILE"
        echo ">>> Completed: $full_name"
    done

    sleep "$POLL_INTERVAL"
done
