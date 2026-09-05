#!/bin/bash
# ================================================================
# Qwen3.5-4B ms-swift LoRA Pipeline v2 — Nautilus
#
# Improvements:
#   - Builds balanced validation set before training
#   - Uses detection F1 for best model selection
#   - Enhanced eval metrics (P/R/F1 + per-category + size accuracy)
#
# Usage (in Pod):
#   bash /workspace/finetune/scripts/nautilus_4b/run_pipeline_4b_v2.sh
# ================================================================
set -euo pipefail
trap 'echo "ERROR: script failed at line $LINENO, exit code $?"' ERR

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================="
echo " Qwen3.5-4B ms-swift LoRA Pipeline v2"
echo " Hardware: 1x A40 / A100"
echo "============================================="

# ============ 0. Environment variables ============
export DATA_DIR="/workspace/finetune/data/data"
export OUTPUT_DIR="/workspace/finetune/output/swift_4b_v2"
export HF_HOME="/workspace/finetune/.cache/huggingface"
export MODELSCOPE_CACHE="/workspace/finetune/.cache/modelscope"
export TMPDIR="/workspace/finetune/tmp"
export TRITON_CACHE_DIR="/workspace/finetune/.cache/triton"
export PIP_CACHE_DIR="/workspace/finetune/.cache/pip"
export TORCH_HOME="/workspace/finetune/.cache/torch"
export XDG_CACHE_HOME="/workspace/finetune/.cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$TMPDIR" "$TRITON_CACHE_DIR" \
         "/workspace/finetune/logs" "/workspace/finetune/.cache"

# ============ 1. Check workspace ============
echo ""
echo ">>> [1/6] Checking workspace paths..."

[ -d "$DATA_DIR" ] || { echo "ERROR: $DATA_DIR not found"; exit 1; }

# Copy shared scripts (optimizer, metric, fix_data_paths)
SHARED_SCRIPTS="/workspace/finetune/scripts/nautilus_scripts_27b_swift"
for f in fix_data_paths.py; do
    if [ -f "$SHARED_SCRIPTS/$f" ] && [ ! -f "$SCRIPT_DIR/$f" ]; then
        cp "$SHARED_SCRIPTS/$f" "$SCRIPT_DIR/$f"
        echo ">>> Copied $f from 27b scripts"
    fi
done

# ============ 2. Install dependencies ============
echo ""
echo ">>> [2/6] Installing dependencies..."

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate base
    _PIP="pip"
elif [ -f "/workspace/venv/bin/activate" ]; then
    source /workspace/venv/bin/activate
    _PIP="/workspace/venv/bin/pip"
else
    echo ">>> Creating virtualenv at /workspace/venv..."
    python3 -m venv /workspace/venv
    source /workspace/venv/bin/activate
    _PIP="/workspace/venv/bin/pip"
fi

_swift_ver=$(python3 -c "import swift; print(swift.__version__)" 2>/dev/null || echo "none")
_tf_ver=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "none")

_tf_ok=0
if python3 -c "
import sys
try:
    import transformers
    v = tuple(int(x) for x in transformers.__version__.split('.')[:2])
    sys.exit(0 if v >= (5, 2) else 1)
except: sys.exit(1)
" 2>/dev/null; then
    _tf_ok=1
fi

if [[ "$_swift_ver" != "none" && "$_tf_ok" -eq 1 ]]; then
    echo ">>> Dependencies OK (ms-swift=$_swift_ver, transformers=$_tf_ver), skipping install."
else
    echo ">>> Installing ms-swift 4.0 + transformers>=5.2.0..."
    unset PYTORCH_CUDA_ALLOC_CONF

    $_PIP install -q \
        "ms-swift[llm] @ git+https://github.com/modelscope/ms-swift.git@main" \
        "transformers>=5.2.0" \
        "accelerate>=0.35.0" \
        "peft>=0.11.0" \
        "bitsandbytes>=0.43.0" \
        "scipy>=1.11" "scikit-learn>=1.3" \
        "numpy>=1.24,<2" \
        datasets pillow qwen-vl-utils

    echo ">>> Installing Flash Attention 2 (optional)..."
    $_PIP install flash-attn --no-build-isolation 2>/dev/null && \
        echo ">>> Flash Attention 2 OK!" || \
        echo ">>> Flash Attention failed — will use SDPA (no impact on correctness)"

    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
fi

echo ">>> ms-swift:     $(python3 -c 'import swift; print(swift.__version__)' 2>/dev/null || echo N/A)"
echo ">>> torch:        $(python3 -c 'import torch; print(torch.__version__)')"
echo ">>> transformers: $(python3 -c 'import transformers; print(transformers.__version__)')"

# ============ 3. Check GPU ============
echo ""
echo ">>> [3/6] Checking GPU..."
python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
    print(f'  GPU {i}: {name} ({mem:.0f} GB)')
" 2>/dev/null || python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f'  GPU {i}: {name} ({mem:.0f} GB)')
" || true

# ============ 4. Fix image paths ============
echo ""
echo ">>> [4/6] Fixing image paths (Mac → Pod)..."

if [ -d "$DATA_DIR/IM_D03_PT_5k_Augmented" ]; then
    IMAGE_DIR="$DATA_DIR/IM_D03_PT_5k_Augmented"
elif [ -d "$DATA_DIR/IM_D03_PT_5K" ]; then
    IMAGE_DIR="$DATA_DIR/IM_D03_PT_5K"
else
    echo "WARNING: No image directory found under $DATA_DIR"
    IMAGE_DIR="$DATA_DIR"
fi
echo ">>> Image dir: $IMAGE_DIR"

python3 "$SCRIPT_DIR/fix_data_paths.py" \
    --input_dir "$DATA_DIR" \
    --image_dir "$IMAGE_DIR"

# Verify data files
[ -f "$DATA_DIR/train_aug.jsonl" ] || { echo "ERROR: train_aug.jsonl not found in $DATA_DIR"; exit 1; }
[ -f "$DATA_DIR/val_aug.jsonl" ]   || { echo "ERROR: val_aug.jsonl not found in $DATA_DIR"; exit 1; }
[ -f "$DATA_DIR/test_aug.jsonl" ]  || { echo "ERROR: test_aug.jsonl not found in $DATA_DIR"; exit 1; }

echo "  Train: $(wc -l < "$DATA_DIR/train_aug.jsonl") samples"
echo "  Val:   $(wc -l < "$DATA_DIR/val_aug.jsonl") samples"
echo "  Test:  $(wc -l < "$DATA_DIR/test_aug.jsonl") samples"

# ============ 5. Build balanced validation set ============
echo ""
echo ">>> [5/6] Building balanced validation set..."

python3 "$SCRIPT_DIR/prepare_balanced_val.py" \
    --train "$DATA_DIR/train_aug.jsonl" \
    --val   "$DATA_DIR/val_aug.jsonl" \
    --test  "$DATA_DIR/test_aug.jsonl" \
    --output "$DATA_DIR/val_balanced.jsonl" \
    --target_size 400 \
    --seed 42

echo "  Balanced val: $(wc -l < "$DATA_DIR/val_balanced.jsonl") samples"

# ============ 6. Start training ============
echo ""
echo ">>> [6/6] Starting training (v2: balanced val + detection F1)..."
echo "    Model:  Qwen/Qwen3.5-4B"
echo "    Output: $OUTPUT_DIR"
echo "    Metric: detection_f1 (greater is better)"

bash "$SCRIPT_DIR/train_swift_4b_v2.sh"

echo ""
echo "============================================="
echo " Training complete! Adapter saved to:"
echo " $OUTPUT_DIR"
echo "============================================="
