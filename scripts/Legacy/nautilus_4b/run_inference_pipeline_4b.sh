#!/bin/bash
# ================================================================
# Qwen3.5-4B 推理流水线 — Nautilus Pod 内执行
#
# 自动查找最佳 checkpoint，对全数据集（train+val+test）推理
# ================================================================
set -euo pipefail
trap 'echo "ERROR: script failed at line $LINENO, exit code $?"' ERR

echo "============================================="
echo " Qwen3.5-4B Inference Pipeline"
echo "============================================="

# ============ 0. 环境变量 ============
export DATA_DIR="/workspace/finetune/data/data"
export OUTPUT_DIR="/workspace/finetune/output/swift_4b"
export HF_HOME="/workspace/finetune/.cache/huggingface"
export MODELSCOPE_CACHE="/workspace/finetune/.cache/modelscope"
export TMPDIR="/workspace/finetune/tmp"
export PIP_CACHE_DIR="/workspace/finetune/.cache/pip"
export TORCH_HOME="/workspace/finetune/.cache/torch"
export XDG_CACHE_HOME="/workspace/finetune/.cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$TMPDIR" "/workspace/finetune/logs"

# ============ 1. 安装依赖 ============
echo ""
echo ">>> [1/4] Installing dependencies..."

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate base
fi

_swift_ok=0
python3 -c "import swift" 2>/dev/null && _swift_ok=1

if [ "$_swift_ok" -eq 0 ]; then
    echo ">>> Installing ms-swift..."
    pip install -q \
        "ms-swift[llm] @ git+https://github.com/modelscope/ms-swift.git@main" \
        "transformers>=5.2.0" \
        "accelerate>=0.35.0" \
        "peft>=0.11.0" \
        pillow qwen-vl-utils
fi

echo ">>> ms-swift:     $(python3 -c 'import swift; print(swift.__version__)' 2>/dev/null || echo N/A)"
echo ">>> torch:        $(python3 -c 'import torch; print(torch.__version__)')"
echo ">>> transformers: $(python3 -c 'import transformers; print(transformers.__version__)')"

# ============ 2. 检查 GPU ============
echo ""
echo ">>> [2/4] Checking GPU..."
python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
" || true

# ============ 3. 查找 checkpoint ============
echo ""
echo ">>> [3/4] Locating checkpoint..."

# 优先找 best_model，不然找最新的 checkpoint
if [ -d "$OUTPUT_DIR/best_model" ]; then
    ADAPTER_PATH="$OUTPUT_DIR/best_model"
else
    # 找最大 step 的 checkpoint
    ADAPTER_PATH=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
fi

if [ -z "$ADAPTER_PATH" ] || [ ! -d "$ADAPTER_PATH" ]; then
    echo "ERROR: No checkpoint found in $OUTPUT_DIR"
    echo "Contents of $OUTPUT_DIR:"
    ls -la "$OUTPUT_DIR/" 2>/dev/null || echo "  (directory not found)"
    exit 1
fi

echo ">>> Using adapter: $ADAPTER_PATH"

# ============ 4. 查找数据集和修复路径 ============
echo ""
echo ">>> [4/4] Preparing data and running inference..."

# 确定图片目录（优先用原始数据）
if [ -d "$DATA_DIR/IM_D03_PT_5K" ]; then
    IMAGE_DIR="$DATA_DIR/IM_D03_PT_5K"
elif [ -d "$DATA_DIR/IM_D03_PT_5k_Augmented" ]; then
    IMAGE_DIR="$DATA_DIR/IM_D03_PT_5k_Augmented"
else
    IMAGE_DIR="$DATA_DIR"
fi
echo ">>> Image dir: $IMAGE_DIR ($(find "$IMAGE_DIR" -maxdepth 1 -name '*.png' | wc -l) png files)"

# 修复数据集中的路径，确保指向实际存在的图片目录
INFER_DIR="/workspace/finetune/output/inference_input"
mkdir -p "$INFER_DIR"

IMAGE_DIR_BASENAME=$(basename "$IMAGE_DIR")
for src in train_v2.jsonl val_v2.jsonl test_v2.jsonl test_v2_orig.jsonl; do
    if [ -f "$DATA_DIR/$src" ]; then
        # 替换所有图片目录名为实际目录
        sed "s|data/IM_D03_PT_5k_Augmented/|${IMAGE_DIR}/|g; s|data/IM_D03_PT_5K/|${IMAGE_DIR}/|g" \
            "$DATA_DIR/$src" > "$INFER_DIR/$src"
        echo ">>> Prepared $src → $INFER_DIR/$src ($(wc -l < "$INFER_DIR/$src") samples)"
    fi
done

# 收集所有可用的数据集文件
DATASETS=()
for f in train_v2.jsonl val_v2.jsonl test_v2.jsonl test_v2_orig.jsonl; do
    [ -f "$INFER_DIR/$f" ] && DATASETS+=("$INFER_DIR/$f")
done

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "ERROR: No dataset files found"
    exit 1
fi

echo ">>> Datasets: ${DATASETS[*]}"
echo ""

# ============ 5. 运行 swift infer ============
RESULT_PATH="/workspace/finetune/output/infer_results_4b.jsonl"

CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen3.5-4B \
    --adapters "$ADAPTER_PATH" \
    --val_dataset "${DATASETS[@]}" \
    --system '你是一个专业的工业图纸分析助手。你的任务是识别和分析工程图纸中的工件特征，包括但不限于：螺纹孔、圆角、矩形孔、圆孔、长圆孔、倒角、槽等特征。对于每种特征，请输出其类型、规格尺寸（如M8、R3）以及在图纸中的位置坐标。请根据图纸内容给出准确、完整的分析结果。' \
    --max_new_tokens 4096 \
    --temperature 0 \
    --result_path "$RESULT_PATH"

echo ""
echo "============================================="
echo " Inference complete!"
echo " Results: $RESULT_PATH"
echo "============================================="
