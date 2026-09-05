#!/bin/bash
# ================================================================
# Qwen3.5-27B ms-swift LoRA 训练 — 多卡 + 在线 benchmark 完整流水线
#
# 硬件：多张 NVIDIA A100 80GB
# 模型：Qwen/Qwen3.5-27B (27B 混合 MoE + GatedDeltaNet, 原生多模态)
# 方法：ms-swift + LoRA + 解锁 ViT + DeepSpeed Zero-3
#
# 用法：
#   kubectl exec -it <pod-name> -n nsf-maica -- bash
#   cd /workspace
#   nohup bash scripts_27b_swift/run_pipeline.sh > /workspace/logs/swift_27b_v13.log 2>&1 &
# ================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================="
echo " Qwen3.5-27B ms-swift + Unlocked ViT Pipeline"
echo " Hardware: multi-GPU A100 80GB"
echo "============================================="

# ============ 0. 环境变量 ============
export DATA_DIR="${DATA_DIR:-/workspace/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/swift_27b}"
export HF_HOME="/workspace/.cache/huggingface"
export MODEL_NAME="Qwen/Qwen3.5-27B"
export HF_TOKEN="${HF_TOKEN:-}"
export MODELSCOPE_CACHE=/workspace/.cache/modelscope
export RUN_TAG="${RUN_TAG:-swift_27b_v13}"
export CANONICAL_TRAIN_LOG="${CANONICAL_TRAIN_LOG:-/workspace/logs/train_swift_v13.log}"
export CANONICAL_RUNNER_LOG="${CANONICAL_RUNNER_LOG:-/workspace/logs/train_swift_v13_runner.log}"
export CANONICAL_BENCHMARK_LOG="${CANONICAL_BENCHMARK_LOG:-/workspace/logs/train_swift_v13_benchmark.log}"

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
export RUN_BENCHMARK_WATCHER="${RUN_BENCHMARK_WATCHER:-true}"
export BENCHMARK_SIZE="${BENCHMARK_SIZE:-50}"
export BENCHMARK_POLL_INTERVAL="${BENCHMARK_POLL_INTERVAL:-180}"
export BENCHMARK_MAX_NEW_TOKENS="${BENCHMARK_MAX_NEW_TOKENS:-4096}"
export BENCHMARK_IOU_THRESHOLD="${BENCHMARK_IOU_THRESHOLD:-0.4}"
export BENCHMARK_LOAD_IN_4BIT="${BENCHMARK_LOAD_IN_4BIT:-false}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
export RESUME_ONLY_MODEL="${RESUME_ONLY_MODEL:-false}"
export ADD_VERSION="${ADD_VERSION:-true}"
export SKIP_FIX_DATA_PATHS="${SKIP_FIX_DATA_PATHS:-auto}"
export IGNORE_DATA_SKIP="${IGNORE_DATA_SKIP:-false}"
export FLASH_ATTN_INSTALL_MODE="${FLASH_ATTN_INSTALL_MODE:-auto}"
export FLASH_ATTN_INSTALL_TIMEOUT="${FLASH_ATTN_INSTALL_TIMEOUT:-900}"

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

    # Flash Attention 2 是可选优化；训练脚本会在缺失时自动回退到 SDPA。
    if python3 -c "import flash_attn" >/dev/null 2>&1; then
        echo ">>> Flash Attention 2 already available, skipping install."
    elif [ "$FLASH_ATTN_INSTALL_MODE" = "never" ]; then
        echo ">>> Flash Attention install disabled, will use SDPA."
    else
        echo ">>> Installing Flash Attention 2 (timeout=${FLASH_ATTN_INSTALL_TIMEOUT}s)..."
        _flash_status=0
        if command -v timeout >/dev/null 2>&1; then
            timeout "$FLASH_ATTN_INSTALL_TIMEOUT" $_PIP install flash-attn --no-build-isolation || _flash_status=$?
        else
            $_PIP install flash-attn --no-build-isolation || _flash_status=$?
        fi

        if [ "$_flash_status" -eq 0 ]; then
            echo ">>> Flash Attention 2 OK!"
        elif [ "$_flash_status" -eq 124 ]; then
            echo ">>> Flash Attention install timed out — will use SDPA"
        else
            echo ">>> Flash Attention install failed — will use SDPA"
        fi
    fi

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
    echo "WARNING: Expected at least 2 GPUs but found $GPU_COUNT."
fi

_all_gpus=$(python3 - <<'PY'
import torch
print(','.join(str(i) for i in range(torch.cuda.device_count())))
PY
)
if [ "$RUN_BENCHMARK_WATCHER" = "true" ] && [ "$GPU_COUNT" -ge 4 ]; then
    export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
    export BENCHMARK_CUDA_VISIBLE_DEVICES="${BENCHMARK_CUDA_VISIBLE_DEVICES:-2,3}"
else
    export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-${_all_gpus}}"
    export BENCHMARK_CUDA_VISIBLE_DEVICES="${BENCHMARK_CUDA_VISIBLE_DEVICES:-}"
fi

echo ">>> Training GPUs:  ${TRAIN_CUDA_VISIBLE_DEVICES}"
if [ -n "$BENCHMARK_CUDA_VISIBLE_DEVICES" ]; then
    echo ">>> Benchmark GPUs: ${BENCHMARK_CUDA_VISIBLE_DEVICES}"
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

# Allow caller to override dataset selection via TRAIN_FILE / VAL_FILE env vars
if [ -n "${TRAIN_FILE:-}" ] && [ -f "$TRAIN_FILE" ]; then
    echo ">>> Using caller-specified TRAIN_FILE: $TRAIN_FILE"
elif [ -f "$DATA_DIR/train_swift_v13.jsonl" ]; then
    TRAIN_FILE="$DATA_DIR/train_swift_v13.jsonl"
elif [ -f "$DATA_DIR/train_swift4.jsonl" ]; then
    TRAIN_FILE="$DATA_DIR/train_swift4.jsonl"
elif [ -f "$DATA_DIR/train_augmented.jsonl" ]; then
    TRAIN_FILE="$DATA_DIR/train_augmented.jsonl"
else
    TRAIN_FILE="$DATA_DIR/train.jsonl"
fi

if [ -n "${VAL_FILE:-}" ] && [ -f "$VAL_FILE" ]; then
    echo ">>> Using caller-specified VAL_FILE: $VAL_FILE"
elif [ -f "$DATA_DIR/val_swift4.jsonl" ]; then
    VAL_FILE="$DATA_DIR/val_swift4.jsonl"
elif [ -f "$DATA_DIR/val_augmented.jsonl" ]; then
    VAL_FILE="$DATA_DIR/val_augmented.jsonl"
else
    VAL_FILE="$DATA_DIR/val.jsonl"
fi

if [ -n "${BENCHMARK_FILE:-}" ] && [ -f "$BENCHMARK_FILE" ]; then
    echo ">>> Using caller-specified BENCHMARK_FILE: $BENCHMARK_FILE"
elif [ -f "$DATA_DIR/benchmarkdata_sampled50.jsonl" ]; then
    BENCHMARK_FILE="$DATA_DIR/benchmarkdata_sampled50.jsonl"
elif [ -f "$DATA_DIR/benchmarkdata_swift4.jsonl" ]; then
    BENCHMARK_FILE="$DATA_DIR/benchmarkdata_swift4.jsonl"
else
    BENCHMARK_FILE="$VAL_FILE"
fi

needs_path_fix() {
    local dataset_file="$1"
    python3 - "$dataset_file" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
need_fix = False
with path.open('r', encoding='utf-8') as handle:
    for line in handle:
        if '/Users/' in line:
            need_fix = True
            break
print('true' if need_fix else 'false')
PY
}

# 修复图片路径（Mac → Pod）— 仅在检测到本地绝对路径时执行
DO_FIX_PATHS="$SKIP_FIX_DATA_PATHS"
if [ "$DO_FIX_PATHS" = "auto" ]; then
    train_needs_fix="$(needs_path_fix "$TRAIN_FILE")"
    val_needs_fix="$(needs_path_fix "$VAL_FILE")"
    if [ "$train_needs_fix" = "true" ] || [ "$val_needs_fix" = "true" ]; then
        DO_FIX_PATHS="true"
    else
        DO_FIX_PATHS="false"
    fi
fi

if [ "$DO_FIX_PATHS" = "true" ]; then
    echo ">>> Fixing image paths (Mac → Pod) for selected datasets only..."
    FIX_DIR="$(mktemp -d "$TMPDIR/fix_jsonl.XXXXXX")"
    cp "$TRAIN_FILE" "$FIX_DIR/$(basename "$TRAIN_FILE")"
    cp "$VAL_FILE" "$FIX_DIR/$(basename "$VAL_FILE")"
    python3 "$SCRIPT_DIR/fix_data_paths.py" \
        --input_dir "$FIX_DIR" \
        --image_dir "$IMAGE_DIR"
    mv "$FIX_DIR/$(basename "$TRAIN_FILE")" "$TRAIN_FILE"
    mv "$FIX_DIR/$(basename "$VAL_FILE")" "$VAL_FILE"
    rm -rf "$FIX_DIR"
else
    echo ">>> Skipping image path fix (dataset paths already point to pod files)"
fi

# 注意：坐标归一化已内置在 prepare_dataset_swift.py 中，无需单独步骤

echo "  Train:     $(wc -l < "$TRAIN_FILE") samples ($(basename "$TRAIN_FILE"))"
echo "  Val:       $(wc -l < "$VAL_FILE") samples ($(basename "$VAL_FILE"))"
echo "  Benchmark: $(wc -l < "$BENCHMARK_FILE") samples ($(basename "$BENCHMARK_FILE"))"

# ============ 4. 训练 (ms-swift + DeepSpeed Zero-3) ============
echo ""
echo ">>> [4/5] Starting ms-swift training on ${TRAIN_CUDA_VISIBLE_DEVICES}..."
echo "    Model: $MODEL_NAME"
echo "    Method: ms-swift + LoRA + Unlocked ViT + DeepSpeed Zero-3"
echo "    Output: $OUTPUT_DIR"
echo "    V13 config: train=$(basename "$TRAIN_FILE"), val=$(basename "$VAL_FILE"), benchmark=$(basename "$BENCHMARK_FILE"), eval/save every 400 steps"
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    echo "    Resume from: $RESUME_FROM_CHECKPOINT"
    echo "    Target total epochs: $NUM_TRAIN_EPOCHS"
    echo "    Add version: $ADD_VERSION"
    echo "    Resume only model: $RESUME_ONLY_MODEL"
    echo "    Ignore data skip: $IGNORE_DATA_SKIP"
fi
echo "    Main train log: $CANONICAL_TRAIN_LOG"
echo "    Runner log:     $CANONICAL_RUNNER_LOG"
echo "    Benchmark log:  $CANONICAL_BENCHMARK_LOG"

mkdir -p /workspace/logs
BENCHMARK_LOG="/workspace/logs/${RUN_TAG}.benchmark.log"
TRAIN_LOG="/workspace/logs/${RUN_TAG}.train_inner.log"
ln -sfn "$TRAIN_LOG" "$CANONICAL_TRAIN_LOG"
ln -sfn "/workspace/logs/${RUN_TAG}.runner.log" "$CANONICAL_RUNNER_LOG"

if [ "$RUN_BENCHMARK_WATCHER" = "true" ] && [ -n "${BENCHMARK_CUDA_VISIBLE_DEVICES:-}" ]; then
    ln -sfn "$BENCHMARK_LOG" "$CANONICAL_BENCHMARK_LOG"
fi

TRAIN_DATASET="$TRAIN_FILE" \
VAL_DATASET="$VAL_FILE" \
OUTPUT_DIR="$OUTPUT_DIR" \
TRAIN_CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
RESUME_FROM_CHECKPOINT="$RESUME_FROM_CHECKPOINT" \
NUM_TRAIN_EPOCHS="$NUM_TRAIN_EPOCHS" \
RESUME_ONLY_MODEL="$RESUME_ONLY_MODEL" \
ADD_VERSION="$ADD_VERSION" \
IGNORE_DATA_SKIP="$IGNORE_DATA_SKIP" \
bash "$SCRIPT_DIR/train_swift.sh" > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo ">>> Training PID: $TRAIN_PID"

if [ "$RUN_BENCHMARK_WATCHER" = "true" ] && [ -n "$BENCHMARK_CUDA_VISIBLE_DEVICES" ]; then
    BENCHMARK_MIN_MTIME="$(date +%s)"
    CUDA_VISIBLE_DEVICES="$BENCHMARK_CUDA_VISIBLE_DEVICES" \
    TRAIN_PID="$TRAIN_PID" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    MODEL_PATH="$MODEL_NAME" \
    VAL_DATASET="$BENCHMARK_FILE" \
    IMAGE_DIR="$IMAGE_DIR" \
    BENCHMARK_SIZE="$BENCHMARK_SIZE" \
    BENCHMARK_POLL_INTERVAL="$BENCHMARK_POLL_INTERVAL" \
    BENCHMARK_MAX_NEW_TOKENS="$BENCHMARK_MAX_NEW_TOKENS" \
    BENCHMARK_IOU_THRESHOLD="$BENCHMARK_IOU_THRESHOLD" \
    BENCHMARK_LOAD_IN_4BIT="$BENCHMARK_LOAD_IN_4BIT" \
    BENCHMARK_MIN_MTIME="$BENCHMARK_MIN_MTIME" \
    bash "$SCRIPT_DIR/watch_checkpoints_and_benchmark.sh" > "$BENCHMARK_LOG" 2>&1 &
    BENCHMARK_PID=$!
    echo ">>> Benchmark watcher PID: $BENCHMARK_PID"
fi

if wait "$TRAIN_PID"; then
    TRAIN_STATUS=0
else
    TRAIN_STATUS=$?
fi

if [ -n "${BENCHMARK_PID:-}" ]; then
    wait "$BENCHMARK_PID" || true
fi

if [ "$TRAIN_STATUS" -ne 0 ]; then
    echo ">>> Training failed with status $TRAIN_STATUS"
    exit "$TRAIN_STATUS"
fi

# ============ 5. 推理（可选，取消注释启用） ============
echo ""
echo ">>> [5/5] Training complete!"
echo "    To run inference:"
echo "    python3 $SCRIPT_DIR/inference_swift.py \\" 
echo "        --adapter_path $OUTPUT_DIR/final_adapter \\"
echo "        --image_dir $DATA_DIR/IM_D03_PT_5K \\"
echo "        --output_file /workspace/output/swift_27b_inference_results.json"
echo "    Online benchmark results: $OUTPUT_DIR/benchmark_results.jsonl"
echo "    Default benchmark data:   $BENCHMARK_FILE"

echo ""
echo "============================================="
echo " Swift 27B training complete!"
echo " Adapter: $OUTPUT_DIR"
echo "============================================="
