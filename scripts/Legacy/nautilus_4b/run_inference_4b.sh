#!/bin/bash
# ================================================================
# Qwen3-VL-4B 推理流水线 — Nautilus (1x A40/A100)
#
# 用法（在 Pod 内）：
#   nohup bash /workspace/finetune/scripts/nautilus_4b/run_inference_4b.sh \
#     > /workspace/finetune/logs/infer_4b.log 2>&1 &
#
# 环境变量可覆盖默认值：
#   MODEL_PATH    — 合并模型或 HF 模型 ID
#   ADAPTER_PATH  — LoRA adapter 路径（留空则用合并模型）
#   DATA_DIR      — 数据目录
#   OUTPUT_DIR    — 输出目录
# ================================================================
set -euo pipefail
trap 'echo "ERROR: script failed at line $LINENO, exit code $?"' ERR

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================="
echo " Qwen3-VL-4B Inference Pipeline"
echo "============================================="

# ============ 0. 环境变量 ============
export DATA_DIR="${DATA_DIR:-/workspace/finetune/data/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/finetune/output}"
export HF_HOME="/workspace/finetune/.cache/huggingface"
export TMPDIR="/workspace/finetune/tmp"
export TRITON_CACHE_DIR="/workspace/finetune/.cache/triton"
export PIP_CACHE_DIR="/workspace/finetune/.cache/pip"
export TORCH_HOME="/workspace/finetune/.cache/torch"
export XDG_CACHE_HOME="/workspace/finetune/.cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

# ============================================================
# 推理模式选择（二选一）
# ============================================================
# 模式 A：合并模型（已 merge LoRA，直接推理）
MODEL_PATH="${MODEL_PATH:-$OUTPUT_DIR/qwen3-vl-4b-merged}"
ADAPTER_PATH="${ADAPTER_PATH:-}"

# 模式 B：基础模型 + adapter（取消注释下面两行）
# MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
# ADAPTER_PATH="${ADAPTER_PATH:-$OUTPUT_DIR/swift_4b/checkpoint-XXXX}"

# ============================================================

mkdir -p "$OUTPUT_DIR" "$TMPDIR" "/workspace/finetune/logs" "/workspace/finetune/.cache"

# ============ 1. 检查依赖 ============
echo ""
echo ">>> [1/4] Checking dependencies..."

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate base
fi

# 检查必要的包是否安装
_missing=0
python3 -c "import torch; import transformers; import peft" 2>/dev/null || _missing=1

if [ "$_missing" -eq 1 ]; then
    echo ">>> Installing dependencies..."
    pip install -q \
        "transformers>=5.2.0" \
        "accelerate>=0.35.0" \
        "peft>=0.11.0" \
        "bitsandbytes>=0.43.0" \
        pillow qwen-vl-utils

    echo ">>> Installing Flash Attention 2 (optional)..."
    pip install flash-attn --no-build-isolation 2>/dev/null && \
        echo ">>> Flash Attention 2 OK!" || \
        echo ">>> Flash Attention failed — will use SDPA"
fi

echo ">>> torch:        $(python3 -c 'import torch; print(torch.__version__)')"
echo ">>> transformers: $(python3 -c 'import transformers; print(transformers.__version__)')"

# ============ 2. 检查 GPU ============
echo ""
echo ">>> [2/4] Checking GPU..."
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
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# ============ 3. 确定图片目录和数据集 ============
echo ""
echo ">>> [3/4] Locating data..."

if [ -d "$DATA_DIR/IM_D03_PT_5K" ]; then
    IMAGE_DIR="$DATA_DIR/IM_D03_PT_5K"
else
    IMAGE_DIR="$DATA_DIR"
fi
echo ">>> Image dir: $IMAGE_DIR"
echo ">>> Images:    $(find "$IMAGE_DIR" -maxdepth 1 -name '*.png' | wc -l) png files"

# 找数据集文件（用于白名单）
TRAIN_DS="" VAL_DS="" TEST_DS=""
for f in train_v2.jsonl train.jsonl; do
    [ -f "$DATA_DIR/$f" ] && { TRAIN_DS="$DATA_DIR/$f"; break; }
done
for f in val_v2.jsonl val.jsonl; do
    [ -f "$DATA_DIR/$f" ] && { VAL_DS="$DATA_DIR/$f"; break; }
done
for f in test_v2.jsonl test_v2_orig.jsonl test.jsonl; do
    [ -f "$DATA_DIR/$f" ] && { TEST_DS="$DATA_DIR/$f"; break; }
done

echo ">>> Train dataset: ${TRAIN_DS:-not found}"
echo ">>> Val dataset:   ${VAL_DS:-not found}"
echo ">>> Test dataset:  ${TEST_DS:-not found}"

# ============ 4. 执行推理 ============
echo ""
echo ">>> [4/4] Starting inference..."
echo "    Model:   $MODEL_PATH"
echo "    Adapter:  ${ADAPTER_PATH:-none (merged model)}"
echo "    Output:  $OUTPUT_DIR/infer_results_4b.json"

INFER_ARGS=(
    --model_path   "$MODEL_PATH"
    --image_dir    "$IMAGE_DIR"
    --output_file  "$OUTPUT_DIR/infer_results_4b.json"
    --max_new_tokens 4096
    --save_every   50
)

[ -n "$ADAPTER_PATH" ] && INFER_ARGS+=(--adapter_path "$ADAPTER_PATH")
[ -n "$TRAIN_DS" ]     && INFER_ARGS+=(--train_dataset "$TRAIN_DS")
[ -n "$VAL_DS" ]       && INFER_ARGS+=(--val_dataset   "$VAL_DS")
[ -n "$TEST_DS" ]      && INFER_ARGS+=(--test_dataset  "$TEST_DS")

# 如果没有找到任何数据集文件，关闭白名单
if [ -z "$TRAIN_DS" ] && [ -z "$VAL_DS" ] && [ -z "$TEST_DS" ]; then
    echo ">>> WARNING: No dataset files found — disabling whitelist"
    INFER_ARGS+=(--no_whitelist)
fi

python3 "$SCRIPT_DIR/inference_4b.py" "${INFER_ARGS[@]}"

echo ""
echo "============================================="
echo " Inference complete!"
echo " Results: $OUTPUT_DIR/infer_results_4b.json"
echo "============================================="
