#!/bin/bash
# ================================================================
# Qwen3.5-4B ms-swift LoRA Pipeline — Old Prompt + New Data
#
# PVC: qwf-workspace
# Data: IM_D03_PT_5k_Augmented.json → train/val JSONL with OLD prompt
# ================================================================
set -euo pipefail
trap 'echo "ERROR: script failed at line $LINENO, exit code $?"' ERR

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================="
echo " Qwen3.5-4B — Old Prompt + New Data"
echo " Hardware: 1x GPU"
echo "============================================="

# ============ 0. Environment variables ============
export DATA_DIR="/workspace/finetune/data/data"
export OUTPUT_DIR="/workspace/finetune/output/swift_4b_oldprompt"
export HF_HOME="/workspace/finetune/.cache/huggingface"
export MODELSCOPE_CACHE="/workspace/finetune/.cache/modelscope"
export TMPDIR="/workspace/finetune/tmp"
export TRITON_CACHE_DIR="/workspace/finetune/.cache/triton"
export PIP_CACHE_DIR="/workspace/finetune/.cache/pip"
export TORCH_HOME="/workspace/finetune/.cache/torch"
export XDG_CACHE_HOME="/workspace/finetune/.cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_DIR" "$TMPDIR" "$TRITON_CACHE_DIR" \
         "/workspace/finetune/logs" "/workspace/finetune/.cache"

# ============ 1. Check workspace ============
echo ""
echo ">>> [1/5] Checking workspace..."

[ -d "$DATA_DIR/IM_D03_PT_5k_Augmented" ] || { echo "ERROR: Image dir not found"; exit 1; }
[ -f "$DATA_DIR/5k_10feats_v2_augmented.json" ] || { echo "ERROR: Augmented JSON not found"; exit 1; }

# ============ 2. Install dependencies ============
echo ""
echo ">>> [2/5] Installing dependencies..."

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate base
    _PIP="pip"
elif [ -f "/workspace/venv/bin/activate" ]; then
    source /workspace/venv/bin/activate
    _PIP="/workspace/venv/bin/pip"
else
    echo ">>> Creating virtualenv..."
    python3 -m venv /workspace/venv
    source /workspace/venv/bin/activate
    _PIP="/workspace/venv/bin/pip"
fi

_swift_ver=$(python3 -c "import swift; print(swift.__version__)" 2>/dev/null || echo "none")
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
    echo ">>> Dependencies OK (ms-swift=$_swift_ver), skipping install."
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
        echo ">>> Flash Attention failed — will use SDPA"

    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
fi

echo ">>> ms-swift:     $(python3 -c 'import swift; print(swift.__version__)' 2>/dev/null || echo N/A)"
echo ">>> torch:        $(python3 -c 'import torch; print(torch.__version__)')"
echo ">>> transformers: $(python3 -c 'import transformers; print(transformers.__version__)')"

# ============ 3. Check GPU ============
echo ""
echo ">>> [3/5] Checking GPU..."
python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    try:
        mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
    except:
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f'  GPU {i}: {name} ({mem:.0f} GB)')
" || true

# ============ 4. Prepare data ============
echo ""
echo ">>> [4/5] Preparing data with OLD prompt..."

TRAIN_FILE="$DATA_DIR/train_10feats_oldprompt_4b.jsonl"
VAL_FILE="$DATA_DIR/val_10feats_oldprompt_4b.jsonl"
BENCH_FILE="$DATA_DIR/benchmark_10feats_oldprompt_4b.jsonl"

if [ -f "$TRAIN_FILE" ] && [ -f "$VAL_FILE" ]; then
    echo ">>> Data files already exist, skipping preparation."
else
    python3 -u "$SCRIPT_DIR/prepare_data.py"
fi

echo "  Train: $(wc -l < "$TRAIN_FILE") samples"
echo "  Val:   $(wc -l < "$VAL_FILE") samples"
echo "  Bench: $(wc -l < "$BENCH_FILE") samples"

# ============ 5. Start training ============
echo ""
echo ">>> [5/5] Starting training..."
echo "    Model:  Qwen/Qwen3.5-4B"
echo "    Prompt: OLD (short system + detailed user)"
echo "    Output: $OUTPUT_DIR"
echo "    Metric: detection_f1"

TRAIN_FILE="$TRAIN_FILE" \
VAL_FILE="$VAL_FILE" \
OUTPUT_DIR="$OUTPUT_DIR" \
bash "$SCRIPT_DIR/train_swift.sh"

echo ""
echo "============================================="
echo " Training complete!"
echo " Adapter: $OUTPUT_DIR"
echo "============================================="
