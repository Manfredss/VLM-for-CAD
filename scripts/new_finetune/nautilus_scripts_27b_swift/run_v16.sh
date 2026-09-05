#!/bin/bash
# ================================================================
# v16 Training Launcher — LoRA r=32, alpha=128 (v10 config + higher LoRA)
#
# Pod: mingtaoqwen3-5b-2a100-b-6cd68954d5-gdwvm (2x A100 80GB)
# Dataset: train_augmented.jsonl / val_augmented.jsonl
# No benchmark watcher (only 2 GPUs, both for training)
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================="
echo " v16: LoRA r=32 alpha=128 + v10 config"
echo "============================================="

# ── v16 hyperparameters ──
export LORA_RANK=32
export LORA_ALPHA=128
export LEARNING_RATE=2e-5
export VIT_LR=1e-5
export ALIGNER_LR=2e-5
export NUM_TRAIN_EPOCHS=3
export PER_DEVICE_TRAIN_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=3   # effective = 2 × 2gpu × 3 = 12 (same as v10)
export IMAGE_MAX_TOKEN_NUM=1560
export MAX_LENGTH=3860
export EVAL_STEPS=400
export SAVE_STEPS=400
export SAVE_TOTAL_LIMIT=10
export WARMUP_RATIO=0.05

# ── Training setup ──
export TRAIN_CUDA_VISIBLE_DEVICES=0,1
export RUN_BENCHMARK_WATCHER=false

# ── Force augmented dataset (uses new TRAIN_FILE/VAL_FILE override in run_pipeline.sh) ──
export TRAIN_FILE=/workspace/data/train_augmented.jsonl
export VAL_FILE=/workspace/data/val_swift4.jsonl

# ── Benchmark dataset (sampled_50, 49 images) ──
export BENCHMARK_FILE=/workspace/data/benchmarkdata_sampled50.jsonl

# ── Log naming ──
export RUN_TAG=swift_27b_v16
export CANONICAL_TRAIN_LOG=/workspace/logs/train_swift_v16.log
export CANONICAL_RUNNER_LOG=/workspace/logs/train_swift_v16_runner.log
export CANONICAL_BENCHMARK_LOG=/workspace/logs/train_swift_v16_benchmark.log

echo ">>> LoRA:          r=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo ">>> LRs:           LLM=${LEARNING_RATE}, ViT=${VIT_LR}, Aligner=${ALIGNER_LR}"
echo ">>> Epochs:        ${NUM_TRAIN_EPOCHS}"
echo ">>> Effective BS:  ${PER_DEVICE_TRAIN_BATCH_SIZE} × 2gpu × ${GRADIENT_ACCUMULATION_STEPS} = $((PER_DEVICE_TRAIN_BATCH_SIZE * 2 * GRADIENT_ACCUMULATION_STEPS))"
echo ">>> Image tokens:  ${IMAGE_MAX_TOKEN_NUM}"
echo ">>> Dataset:       train_augmented.jsonl / val_augmented.jsonl"

exec bash "$SCRIPT_DIR/run_pipeline.sh"
