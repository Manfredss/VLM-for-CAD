#!/bin/bash
# ================================================================
# Sequential watcher: for each new checkpoint, run:
#   1. Benchmark eval (evaluate_checkpoints_benchmark.py)
#   2. Sampled-100 inference (infer_sampled100.py)
# Both share a single GPU sequentially — no OOM risk.
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/swift_27b}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-27B}"
IMAGE_DIR="${IMAGE_DIR:-/workspace/data/5k}"
POLL_INTERVAL="${POLL_INTERVAL:-300}"
MIN_MTIME="${MIN_MTIME:-0}"

# Benchmark config
BENCHMARK_DATASET="${BENCHMARK_DATASET:-/workspace/data/benchmark_10feats.jsonl}"
BENCHMARK_SIZE="${BENCHMARK_SIZE:-48}"
BENCHMARK_IOU_THRESHOLD="${BENCHMARK_IOU_THRESHOLD:-0.4}"
BENCHMARK_MAX_NEW_TOKENS="${BENCHMARK_MAX_NEW_TOKENS:-4096}"
BENCHMARK_RESULTS="${BENCHMARK_RESULTS:-$OUTPUT_DIR/benchmark_results_10feats.jsonl}"

# Sampled-100 config
GT_JSON="${GT_JSON:-/workspace/data/5k_10feats_v2_sampled_100.json}"
RESULTS_DIR="${RESULTS_DIR:-$OUTPUT_DIR/sampled100_results}"

mkdir -p "$RESULTS_DIR"
touch "$BENCHMARK_RESULTS"

DONE_FILE="$RESULTS_DIR/.done_eval_infer"
touch "$DONE_FILE"

already_done() { grep -qxF "$1" "$DONE_FILE" 2>/dev/null; }
mark_done() { echo "$1" >> "$DONE_FILE"; }

echo ">>> Sequential eval+infer watcher starting"
echo ">>> Output dir:      $OUTPUT_DIR"
echo ">>> Benchmark data:  $BENCHMARK_DATASET"
echo ">>> GT JSON:         $GT_JSON"
echo ">>> Image dir:       $IMAGE_DIR"
echo ">>> Poll interval:   ${POLL_INTERVAL}s"
echo ">>> Min mtime:       $MIN_MTIME"

while true; do
    checkpoints=$(python3 - "$OUTPUT_DIR" "$MIN_MTIME" <<'PY'
import sys
from pathlib import Path
output_dir = Path(sys.argv[1])
min_mtime = float(sys.argv[2])
candidates = []
for path in output_dir.glob('*'):
    if not path.is_dir():
        continue
    for child in path.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith('checkpoint-') or child.name == 'final_adapter':
            try:
                mtime = child.stat().st_mtime
            except OSError:
                continue
            if mtime >= min_mtime:
                candidates.append(child)
for item in sorted(candidates, key=lambda p: str(p)):
    print(item)
PY
    )

    for ckpt in $checkpoints; do
        if already_done "$ckpt"; then
            continue
        fi

        ckpt_name=$(basename "$ckpt")
        run_name=$(basename "$(dirname "$ckpt")")

        echo ""
        echo "=========================================="
        echo ">>> Processing $run_name/$ckpt_name"
        echo "=========================================="

        # --- Step 1: Benchmark eval ---
        echo ">>> [1/2] Running benchmark eval..."
        tmp_json="$(mktemp /tmp/benchmark_ckpt.XXXXXX.json)"
        CUDA_VISIBLE_DEVICES=0 python3 -u "$SCRIPT_DIR/evaluate_checkpoints_benchmark.py" \
            --model_path "$MODEL_PATH" \
            --checkpoints_root "$OUTPUT_DIR" \
            --checkpoint_path "$ckpt" \
            --val_dataset "$BENCHMARK_DATASET" \
            --image_dir "$IMAGE_DIR" \
            --benchmark_size "$BENCHMARK_SIZE" \
            --max_new_tokens "$BENCHMARK_MAX_NEW_TOKENS" \
            --iou_threshold "$BENCHMARK_IOU_THRESHOLD" \
            --output_json "$tmp_json"

        # Append to results JSONL
        python3 -c "
import json, sys
with open('$tmp_json') as f:
    data = json.load(f)
rows = data if isinstance(data, list) else [data]
with open('$BENCHMARK_RESULTS', 'a') as out:
    for row in rows:
        out.write(json.dumps(row, ensure_ascii=False) + '\n')
"
        rm -f "$tmp_json"
        echo ">>> Benchmark eval done for $run_name/$ckpt_name"

        # --- Step 2: Sampled-100 inference ---
        echo ">>> [2/2] Running sampled-100 inference..."
        output_file="$RESULTS_DIR/${run_name}_${ckpt_name}_results.json"
        CUDA_VISIBLE_DEVICES=0 python3 -u "$SCRIPT_DIR/infer_sampled100.py" \
            --model_path "$MODEL_PATH" \
            --adapter_path "$ckpt" \
            --gt_json "$GT_JSON" \
            --image_dir "$IMAGE_DIR" \
            --output_file "$output_file" \
            --max_new_tokens 4096

        mark_done "$ckpt"
        echo ">>> Done: $run_name/$ckpt_name"

        # Print global ranking
        echo ""
        echo "=== Global Benchmark Ranking ==="
        python3 -c "
import json
from pathlib import Path
results = {}
with open('$BENCHMARK_RESULTS') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            row = json.loads(line)
            results[row['checkpoint']] = row
        except: pass
ranked = sorted(results.values(), key=lambda r: r.get('f1',0), reverse=True)
for i, r in enumerate(ranked, 1):
    ckpt = Path(r['checkpoint'])
    label = f'{ckpt.parent.name}/{ckpt.name}'
    print(f\"{i}. {label} | P={r.get('precision',0):.3f} R={r.get('recall',0):.3f} F1={r.get('f1',0):.3f} AQ={r.get('avg_quality',0):.3f}\")
"
    done

    sleep "$POLL_INTERVAL"
done
