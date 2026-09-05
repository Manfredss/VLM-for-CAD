#!/bin/bash
# ================================================================
# 10-Feature Training Launcher — Qwen3.5-27B on 5k_10feats_v2_augmented
#
# Pod: wenfeiqwen3-5-27b-2a100-a (2x A100 80GB)
# Dataset: train_10feats.jsonl / val_10feats.jsonl
# Benchmark: on separate 1xA100 pod
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================="
echo " 10-Feature: LoRA r=32 alpha=128 + 2xA100"
echo "============================================="

# ── Hyperparameters (same as v16) ──
export LORA_RANK=32
export LORA_ALPHA=128
export LEARNING_RATE=2e-5
export VIT_LR=1e-5
export ALIGNER_LR=2e-5
export NUM_TRAIN_EPOCHS=3
export PER_DEVICE_TRAIN_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=3   # effective = 2 × 2gpu × 3 = 12
export IMAGE_MAX_TOKEN_NUM=1560
export MAX_LENGTH=3860
export EVAL_STEPS=400
export SAVE_STEPS=400
export SAVE_TOTAL_LIMIT=10
export WARMUP_RATIO=0.05

# ── Training setup ──
export TRAIN_CUDA_VISIBLE_DEVICES=0,1
export RUN_BENCHMARK_WATCHER=false   # benchmark runs on separate pod

# ── Dataset ──
export TRAIN_FILE=/workspace/data/train_10feats.jsonl
export VAL_FILE=/workspace/data/val_10feats.jsonl

# ── Benchmark (not used on training pod, but set for compatibility) ──
export BENCHMARK_FILE=/workspace/data/benchmark_10feats.jsonl

# ── Log naming ──
export RUN_TAG=swift_27b_10feats
export CANONICAL_TRAIN_LOG=/workspace/logs/train_swift_10feats.log
export CANONICAL_RUNNER_LOG=/workspace/logs/train_swift_10feats_runner.log
export CANONICAL_BENCHMARK_LOG=/workspace/logs/train_swift_10feats_benchmark.log

echo ">>> LoRA:          r=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo ">>> LRs:           LLM=${LEARNING_RATE}, ViT=${VIT_LR}, Aligner=${ALIGNER_LR}"
echo ">>> Epochs:        ${NUM_TRAIN_EPOCHS}"
echo ">>> Effective BS:  ${PER_DEVICE_TRAIN_BATCH_SIZE} × 2gpu × ${GRADIENT_ACCUMULATION_STEPS} = $((PER_DEVICE_TRAIN_BATCH_SIZE * 2 * GRADIENT_ACCUMULATION_STEPS))"
echo ">>> Image tokens:  ${IMAGE_MAX_TOKEN_NUM}"
echo ">>> Dataset:       train_10feats.jsonl / val_10feats.jsonl"

exec bash "$SCRIPT_DIR/run_pipeline.sh"
