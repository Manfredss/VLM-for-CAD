#!/bin/bash
# ================================================================
# Watch for new checkpoints and run sampled-100 inference + metrics
# Runs on the eval pod (1xA100) alongside the benchmark watcher
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/swift_27b}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-27B}"
GT_JSON="${GT_JSON:-/workspace/data/5k_10feats_v2_sampled_100.json}"
IMAGE_DIR="${IMAGE_DIR:-/workspace/data/5k}"
POLL_INTERVAL="${POLL_INTERVAL:-300}"
MIN_MTIME="${MIN_MTIME:-0}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/output/swift_27b/sampled100_results}"

mkdir -p "$RESULTS_DIR"

# Track which checkpoints we've already processed
DONE_FILE="$RESULTS_DIR/.done_checkpoints"
touch "$DONE_FILE"

already_done() {
    grep -qxF "$1" "$DONE_FILE" 2>/dev/null
}

mark_done() {
    echo "$1" >> "$DONE_FILE"
}

echo ">>> Sampled-100 inference watcher starting"
echo ">>> Output dir:  $OUTPUT_DIR"
echo ">>> Results dir: $RESULTS_DIR"
echo ">>> GT JSON:     $GT_JSON"
echo ">>> Image dir:   $IMAGE_DIR"
echo ">>> Poll interval: ${POLL_INTERVAL}s"

while true; do
    # Find checkpoints
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

        # Extract checkpoint name for output file
        ckpt_name=$(basename "$ckpt")
        run_name=$(basename "$(dirname "$ckpt")")
        output_file="$RESULTS_DIR/${run_name}_${ckpt_name}_results.json"

        echo ">>> Running sampled-100 inference on $run_name/$ckpt_name ..."
        CUDA_VISIBLE_DEVICES=0 python3 -u "$SCRIPT_DIR/infer_sampled100.py" \
            --model_path "$MODEL_PATH" \
            --adapter_path "$ckpt" \
            --gt_json "$GT_JSON" \
            --image_dir "$IMAGE_DIR" \
            --output_file "$output_file" \
            --max_new_tokens 4096

        mark_done "$ckpt"
        echo ">>> Done: $output_file"
        echo ""
    done

    sleep "$POLL_INTERVAL"
done
