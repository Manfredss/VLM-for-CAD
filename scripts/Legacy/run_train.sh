#!/bin/bash
# =============================================================================
# Training launch script
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="${1:-configs/train_config.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"

echo "============================================"
echo "  Industrial Drawing Feature Recognition"
echo "  Training with Qwen2.5-VL"
echo "============================================"
echo "Config: $CONFIG"
echo "GPUs: $NUM_GPUS"
echo ""

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching multi-GPU training with torchrun..."
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=29500 \
        src/train.py \
        --config "$CONFIG" \
        "${@:2}"
else
    echo "Launching single-GPU training..."
    python src/train.py \
        --config "$CONFIG" \
        "${@:2}"
fi
