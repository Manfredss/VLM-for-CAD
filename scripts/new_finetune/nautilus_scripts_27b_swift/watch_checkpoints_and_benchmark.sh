#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/swift_27b}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-27B}"
VAL_DATASET="${VAL_DATASET:-/workspace/data/benchmarkdata_swift4.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-/workspace/data/5k}"
TRAIN_PID="${TRAIN_PID:-}"
BENCHMARK_SIZE="${BENCHMARK_SIZE:-50}"
BENCHMARK_POLL_INTERVAL="${BENCHMARK_POLL_INTERVAL:-180}"
BENCHMARK_MAX_NEW_TOKENS="${BENCHMARK_MAX_NEW_TOKENS:-4096}"
BENCHMARK_IOU_THRESHOLD="${BENCHMARK_IOU_THRESHOLD:-0.4}"
BENCHMARK_LOAD_IN_4BIT="${BENCHMARK_LOAD_IN_4BIT:-false}"
RESULTS_JSONL="${RESULTS_JSONL:-$OUTPUT_DIR/benchmark_results.jsonl}"
BENCHMARK_MIN_MTIME="${BENCHMARK_MIN_MTIME:-0}"

mkdir -p "$OUTPUT_DIR"
touch "$RESULTS_JSONL"

format_checkpoint_label() {
    local checkpoint_path="$1"
    python3 - "$OUTPUT_DIR" "$checkpoint_path" <<'PY'
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2])

try:
    print(checkpoint_path.relative_to(output_dir))
except ValueError:
    parent_name = checkpoint_path.parent.name
    if parent_name:
        print(f'{parent_name}/{checkpoint_path.name}')
    else:
        print(checkpoint_path.name)
PY
}

already_done() {
    local target="$1"
    python3 - "$RESULTS_JSONL" "$target" <<'PY'
import json, sys
results_path, target = sys.argv[1], sys.argv[2]
done = False
with open(results_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get('checkpoint') == target:
            done = True
            break
print('1' if done else '0')
PY
}

append_result() {
    local json_path="$1"
    python3 - "$json_path" "$RESULTS_JSONL" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src, 'r', encoding='utf-8') as f:
    data = json.load(f)
if isinstance(data, list):
    rows = data
else:
    rows = [data]
with open(dst, 'a', encoding='utf-8') as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
PY
}

print_global_ranking() {
    python3 - "$OUTPUT_DIR" "$RESULTS_JSONL" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
results_path = sys.argv[2]
latest_by_checkpoint = {}


def format_checkpoint_label(checkpoint_path: str) -> str:
    path = Path(checkpoint_path)
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        if path.parent.name:
            return f'{path.parent.name}/{path.name}'
        return path.name

with open(results_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        checkpoint = row.get('checkpoint')
        if checkpoint:
            latest_by_checkpoint[checkpoint] = row

results = list(latest_by_checkpoint.values())
results.sort(
    key=lambda row: (
        row.get('f1', 0.0),
        row.get('avg_quality', 0.0),
        row.get('precision', 0.0),
        row.get('recall', 0.0),
        Path(row.get('checkpoint', '')).name,
    ),
    reverse=True,
)

print('')
print('=== Global Ranking by benchmark F1 ===', flush=True)
if not results:
    print('(no benchmark results yet)', flush=True)
    raise SystemExit(0)

for idx, result in enumerate(results, start=1):
    checkpoint_name = format_checkpoint_label(result['checkpoint'])
    print(
        f"{idx}. {checkpoint_name} | "
        f"P={result.get('precision', 0.0):.3f} "
        f"R={result.get('recall', 0.0):.3f} "
        f"F1={result.get('f1', 0.0):.3f} "
        f"AQ={result.get('avg_quality', 0.0):.3f} "
        f"TP={result.get('tp', 0)} "
        f"FP={result.get('fp', 0)} "
        f"FN={result.get('fn', 0)}"
    , flush=True)
PY
}

evaluate_one() {
    local checkpoint_path="$1"
    local tmp_json
    local checkpoint_label
    checkpoint_label="$(format_checkpoint_label "$checkpoint_path")"
    tmp_json="$(mktemp /tmp/benchmark_ckpt.XXXXXX.json)"
    cmd=(python3 -u "$SCRIPT_DIR/evaluate_checkpoints_benchmark.py"
        --model_path "$MODEL_PATH"
        --checkpoints_root "$OUTPUT_DIR"
        --checkpoint_path "$checkpoint_path"
        --val_dataset "$VAL_DATASET"
        --image_dir "$IMAGE_DIR"
        --benchmark_size "$BENCHMARK_SIZE"
        --max_new_tokens "$BENCHMARK_MAX_NEW_TOKENS"
        --iou_threshold "$BENCHMARK_IOU_THRESHOLD"
        --output_json "$tmp_json")
    if [ "$BENCHMARK_LOAD_IN_4BIT" = "true" ]; then
        cmd+=(--load_in_4bit)
    fi
    echo ">>> Benchmarking $checkpoint_label on GPU(s): ${CUDA_VISIBLE_DEVICES:-all}"
    "${cmd[@]}"
    append_result "$tmp_json"
    print_global_ranking
    rm -f "$tmp_json"
}

echo ">>> Watching $OUTPUT_DIR for new checkpoints"
if [ "$BENCHMARK_MIN_MTIME" != "0" ]; then
    echo ">>> Ignoring checkpoints older than unix mtime: $BENCHMARK_MIN_MTIME"
fi
echo ">>> Results will be appended to $RESULTS_JSONL"
print_global_ranking

while true; do
    found_new=0

    while IFS= read -r checkpoint_path; do
        done_flag="$(already_done "$checkpoint_path")"
        if [ "$done_flag" = "1" ]; then
            continue
        fi
        found_new=1
        evaluate_one "$checkpoint_path"
    done < <(
        python3 - "$OUTPUT_DIR" "$BENCHMARK_MIN_MTIME" <<'PY'
import os
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

    if [ -n "$TRAIN_PID" ] && ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        if [ "$found_new" -eq 0 ]; then
            echo ">>> Training PID $TRAIN_PID exited and no new checkpoints remain. Watcher stopping."
            break
        fi
    fi

    sleep "$BENCHMARK_POLL_INTERVAL"
done