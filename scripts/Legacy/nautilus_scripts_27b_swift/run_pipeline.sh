#!/bin/bash
# ================================================================
# Qwen3.5-27B ms-swift LoRA 训练 — 2x A100 80GB 完整流水线
#
# 硬件：2x NVIDIA A100 80GB
# 模型：Qwen/Qwen3.5-27B (27B 混合 MoE + GatedDeltaNet, 原生多模态)
# 方法：ms-swift + LoRA + 解锁 ViT + DeepSpeed Zero-3
#
# 用法：
#   kubectl exec -it <pod-name> -n nsf-maica -- bash
#   cd /workspace
#   nohup bash scripts_27b_swift/run_pipeline.sh > /workspace/logs/swift_27b_train.log 2>&1 &
# ================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================="
echo " Qwen3.5-27B ms-swift + Unlocked ViT Pipeline"
echo " Hardware: 2x A100 80GB"
echo "============================================="

# ============ 0. 环境变量 ============
export DATA_DIR="/workspace/data"
export OUTPUT_DIR="/workspace/output/swift_27b"
export HF_HOME="/workspace/.cache/huggingface"
export MODEL_NAME="Qwen/Qwen3.5-27B"
export HF_TOKEN="${HF_TOKEN:-}"
export MODELSCOPE_CACHE=/workspace/.cache/modelscope

export WANDB_PROJECT="industrial-drawing-vl-swift-27b"
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE="disabled"
    echo ">>> WANDB_API_KEY not set → wandb disabled"
else
    export WANDB_MODE="online"
fi

# === 临时目录重定向到 PVC（防止 ephemeral-storage 超限被驱逐） ===
export TMPDIR="/workspace/tmp"
export TEMP="/workspace/tmp"
export TMP="/workspace/tmp"
export TRITON_CACHE_DIR="/workspace/.cache/triton"
export PIP_CACHE_DIR="/workspace/.cache/pip"
export TORCH_HOME="/workspace/.cache/torch"
export XDG_CACHE_HOME="/workspace/.cache"

# CUDA 内存优化
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
# NCCL 优化
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$TMPDIR" "$TRITON_CACHE_DIR" "$PIP_CACHE_DIR"

# ============ 1. 安装依赖 ============
echo ""
echo ">>> [1/5] Installing dependencies..."

# Qwen3.5-27B 需要 transformers>=5.2.0 和 ms-swift 4.0.0.dev0
# 优先使用 conda 环境（/opt/conda），也支持 venv2
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate base
    _ENV_TYPE="conda"
    _PIP="pip"
elif [ -f "/workspace/venv2/bin/activate" ]; then
    source /workspace/venv2/bin/activate
    _ENV_TYPE="venv2"
    _PIP="/workspace/venv2/bin/pip"
else
    echo ">>> No conda or venv2 found, creating venv2..."
    python3 -m venv /workspace/venv2
    source /workspace/venv2/bin/activate
    _ENV_TYPE="venv2"
    _PIP="/workspace/venv2/bin/pip"
fi

echo ">>> Using environment: $_ENV_TYPE"

_need_install=0
_swift_ver=$(python3 -c "import swift; print(swift.__version__)" 2>/dev/null || echo "none")
_ds_ver=$(python3 -c "import deepspeed; print(deepspeed.__version__)" 2>/dev/null || echo "none")
_tf_ver=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "none")

# 检查 transformers 版本是否 >= 5.2.0
_tf_ok=0
if [[ "$_tf_ver" =~ ^5\. ]]; then
    _tf_major_minor=$(echo "$_tf_ver" | cut -d. -f1,2)
    if python3 -c "exit(0 if tuple(map(int, '$_tf_ver'.split('.')[:2])) >= (5, 2) else 1)" 2>/dev/null; then
        _tf_ok=1
    fi
fi

if [[ "$_swift_ver" != "none" && "$_ds_ver" != "none" && "$_tf_ok" -eq 1 ]]; then
    echo ">>> Dependencies OK (ms-swift=$_swift_ver, deepspeed=$_ds_ver, transformers=$_tf_ver), skipping install."
else
    echo ">>> Installing/upgrading dependencies..."
    _need_install=1
fi

if [ "$_need_install" -eq 1 ]; then
    # 某些 torch/驱动组合在 pip 构建 deepspeed 时会因 expandable_segments 报错
    _OLD_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}"
    unset PYTORCH_CUDA_ALLOC_CONF

    # ms-swift 4.0 dev0 + 核心依赖（PyPI 版不支持 Qwen3.5，必须从 git 安装）
    echo ">>> Installing ms-swift 4.0 dev0 + core dependencies..."
    $_PIP install -q \
        "ms-swift[llm] @ git+https://github.com/modelscope/ms-swift.git@main" \
        "transformers>=5.2.0" \
        "accelerate>=0.35.0" \
        "peft>=0.11.0" \
        "bitsandbytes>=0.43.0" \
        "scipy>=1.11" "scikit-learn>=1.3" \
        "numpy>=1.24,<2" \
        datasets pillow qwen-vl-utils

    # deepspeed 在 Python 3.11 + 某些 torch 组合下，最新版可能触发 torch.compile 兼容错误
    # 先尝试较新稳定版，失败再回退到 0.13.x
    echo ">>> Installing deepspeed (with compatibility fallback)..."
    if ! $_PIP install -q "deepspeed==0.14.4"; then
        echo ">>> deepspeed 0.14.4 failed, fallback to 0.13.5..."
        $_PIP install -q "deepspeed==0.13.5"
    fi

    # Flash Attention 2
    echo ">>> Installing Flash Attention 2..."
    $_PIP install flash-attn --no-build-isolation 2>/dev/null && \
        echo ">>> Flash Attention 2 OK!" || \
        echo ">>> Flash Attention install failed — will use SDPA"

    # 依赖安装完成后恢复该选项（供训练阶段显存分配优化）
    if [ -n "${_OLD_CUDA_ALLOC_CONF}" ]; then
        export PYTORCH_CUDA_ALLOC_CONF="${_OLD_CUDA_ALLOC_CONF}"
    else
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    fi
fi

echo ">>> Python:        $(which python3)"
echo ">>> ms-swift:      $(python3 -c 'import swift; print(swift.__version__)' 2>/dev/null || echo 'N/A')"
echo ">>> torch:         $(python3 -c 'import torch; print(torch.__version__)')"
echo ">>> transformers:  $(python3 -c 'import transformers; print(transformers.__version__)')"
echo ">>> deepspeed:     $(python3 -c 'import deepspeed; print(deepspeed.__version__)' 2>/dev/null || echo 'N/A')"

# ============ 2. 检查 GPU ============
echo ""
echo ">>> [2/5] Checking GPUs..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f'  GPU {i}: {name} ({mem:.0f} GB)')
" || true

GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "WARNING: Expected 2 GPUs but found $GPU_COUNT."
fi

# ============ 3. 准备数据 ============
echo ""
echo ">>> [3/5] Preparing dataset..."

# 优先使用新上传的 5k 数据目录；若不存在则回退旧目录
if [ -d "$DATA_DIR/5k" ]; then
    IMAGE_DIR="$DATA_DIR/5k"
elif [ -d "$DATA_DIR/images" ]; then
    IMAGE_DIR="$DATA_DIR/images"
elif [ -d "$DATA_DIR/IM_D03_PT_5K" ]; then
    IMAGE_DIR="$DATA_DIR/IM_D03_PT_5K"
    ln -s "$DATA_DIR/IM_D03_PT_5K" "$DATA_DIR/images" 2>/dev/null || true
else
    echo ">>> ERROR: No image directory found under $DATA_DIR (expected 5k/images/IM_D03_PT_5K)"
    exit 1
fi
echo ">>> Using image dir: $IMAGE_DIR"

# 修复图片路径（Mac → Pod）— JSONL 版
echo ">>> Fixing image paths (Mac → Pod) for all JSONL datasets..."
python3 "$SCRIPT_DIR/fix_data_paths.py" \
    --input_dir "$DATA_DIR" \
    --image_dir "$IMAGE_DIR"

# 注意：坐标归一化已内置在 prepare_dataset_swift.py 中，无需单独步骤

if [ -f "$DATA_DIR/train_augmented.jsonl" ]; then
    TRAIN_FILE="$DATA_DIR/train_augmented.jsonl"
elif [ -f "$DATA_DIR/train_swift4.jsonl" ]; then
    TRAIN_FILE="$DATA_DIR/train_swift4.jsonl"
else
    TRAIN_FILE="$DATA_DIR/train.jsonl"
fi

if [ -f "$DATA_DIR/val_augmented.jsonl" ]; then
    VAL_FILE="$DATA_DIR/val_augmented.jsonl"
elif [ -f "$DATA_DIR/val_swift4.jsonl" ]; then
    VAL_FILE="$DATA_DIR/val_swift4.jsonl"
else
    VAL_FILE="$DATA_DIR/val.jsonl"
fi

echo "  Train: $(wc -l < "$TRAIN_FILE") samples ($(basename "$TRAIN_FILE"))"
echo "  Val:   $(wc -l < "$VAL_FILE") samples ($(basename "$VAL_FILE"))"

# ============ 4. 训练 (ms-swift + DeepSpeed Zero-3) ============
echo ""
echo ">>> [4/5] Starting ms-swift training on ${GPU_COUNT} GPUs..."
echo "    Model: $MODEL_NAME"
echo "    Method: ms-swift + LoRA + Unlocked ViT + DeepSpeed Zero-3"
echo "    Output: $OUTPUT_DIR"

bash "$SCRIPT_DIR/train_swift.sh"

# ============ 5. 推理（可选，取消注释启用） ============
echo ""
echo ">>> [5/5] Training complete!"
echo "    To run inference:"
echo "    python3 $SCRIPT_DIR/inference_swift.py \\" 
echo "        --adapter_path $OUTPUT_DIR/final_adapter \\"
echo "        --image_dir $DATA_DIR/IM_D03_PT_5K \\"
echo "        --output_file /workspace/output/swift_27b_inference_results.json"

echo ""
echo "============================================="
echo " Swift 27B training complete!"
echo " Adapter: $OUTPUT_DIR"
echo "============================================="
