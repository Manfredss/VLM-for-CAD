#!/bin/bash
# ================================================================
# Qwen3.5-4B Post-Training Pipeline — Rare Features Enhancement
#
# Post-trains the original-dataset model (checkpoint-1200) with
# samples containing rare/absent features from augmented data:
#   - Rectangular Hole / Group (0.3% / absent in original)
#   - Slotted Hole / Group (1.8% / absent in original)
#   - Fillet Group (4.6% in original)
#
# Includes 15% replay of common-feature samples to prevent forgetting.
#
# Usage (in Pod):
#   bash /workspace/finetune/scripts/nautilus_4b/run_pipeline_4b_posttrain.sh
# ================================================================
set -euo pipefail
trap 'echo "ERROR: script failed at line $LINENO, exit code $?"' ERR

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================="
echo " Qwen3.5-4B Post-Training: Rare Features"
echo " Hardware: 1x A40 / A100 / RTX 5000"
echo "============================================="

# ============ 0. Environment ============
export DATA_DIR="/workspace/finetune/data/data"
export OUTPUT_DIR="/workspace/finetune/output/swift_4b_posttrain"
export RESUME_CHECKPOINT="/workspace/finetune/output/swift_4b/v5-20260312-054110/checkpoint-1200"
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
echo ">>> [1/6] Checking workspace..."

[ -d "$DATA_DIR" ] || { echo "ERROR: $DATA_DIR not found"; exit 1; }
[ -d "$RESUME_CHECKPOINT" ] || { echo "ERROR: Resume checkpoint not found: $RESUME_CHECKPOINT"; exit 1; }

# Copy fix_data_paths if needed
SHARED_SCRIPTS="/workspace/finetune/scripts/nautilus_scripts_27b_swift"
for f in fix_data_paths.py; do
    if [ -f "$SHARED_SCRIPTS/$f" ] && [ ! -f "$SCRIPT_DIR/$f" ]; then
        cp "$SHARED_SCRIPTS/$f" "$SCRIPT_DIR/$f"
        echo ">>> Copied $f from 27b scripts"
    fi
done

echo ">>> Resume checkpoint: $RESUME_CHECKPOINT"

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
        echo ">>> Flash Attention failed — will use SDPA"

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
echo ">>> [4/6] Fixing image paths..."

if [ -d "$DATA_DIR/IM_D03_PT_5k_Augmented" ]; then
    IMAGE_DIR="$DATA_DIR/IM_D03_PT_5k_Augmented"
elif [ -d "$DATA_DIR/IM_D03_PT_5K" ]; then
    IMAGE_DIR="$DATA_DIR/IM_D03_PT_5K"
else
    IMAGE_DIR="$DATA_DIR"
fi
echo ">>> Image dir: $IMAGE_DIR"

python3 "$SCRIPT_DIR/fix_data_paths.py" \
    --input_dir "$DATA_DIR" \
    --image_dir "$IMAGE_DIR"

# ============ 5. Prepare rare-feature post-training data ============
echo ""
echo ">>> [5/6] Preparing rare-feature post-training data..."

python3 "$SCRIPT_DIR/prepare_rare_features.py" \
    --augmented "$DATA_DIR/train_aug.jsonl" \
    --original  "$DATA_DIR/train_v2.jsonl" \
    --val_pool  "$DATA_DIR/val_aug.jsonl" "$DATA_DIR/test_aug.jsonl" \
    --output_train "$DATA_DIR/posttrain_rare.jsonl" \
    --output_val   "$DATA_DIR/posttrain_rare_val.jsonl" \
    --replay_ratio 0.15 \
    --val_size 100 \
    --seed 42

echo "  Post-train: $(wc -l < "$DATA_DIR/posttrain_rare.jsonl") samples"
echo "  Val:        $(wc -l < "$DATA_DIR/posttrain_rare_val.jsonl") samples"

# ============ 6. Post-training ============
echo ""
echo ">>> [6/6] Starting post-training (rare features)..."
echo "    Base:   $RESUME_CHECKPOINT"
echo "    LR:     5e-6 (LLM), 1e-5 (ViT/aligner)"
echo "    Epochs: 2"
echo "    Metric: detection_f1"

bash "$SCRIPT_DIR/train_swift_4b_posttrain.sh"

echo ""
echo "============================================="
echo " Post-training complete! Adapter saved to:"
echo " $OUTPUT_DIR"
echo "============================================="
