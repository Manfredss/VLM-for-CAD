#!/bin/bash
# ================================================================
# Qwen3.5-27B ms-swift LoRA 训练 — 2x A100 80GB
#
# 核心创新（vs FSDP 方案）：
#   - 解锁 ViT + Aligner（freeze_vit=false, freeze_aligner=false）
#   - 分层学习率（ViT 5e-5, Aligner 5e-5, LLM 2e-5）
#   - DeepSpeed Zero-3 分布式
#   - LoRA rank=16 all-linear（参数量更小但覆盖更广）
#
# Qwen3.5-27B 注意事项：
#   - 混合 MoE + GatedDeltaNet 架构，27B 参数
#   - 原生多模态（无需 "-VL" 后缀）
#   - 默认开启 thinking mode，训练时需关闭
#   - 需要 transformers>=5.2.0, ms-swift 4.0.0.dev0
#
# 用法：
#   source /opt/conda/etc/profile.d/conda.sh && conda activate base
#   bash /workspace/scripts_27b_swift/train_swift.sh
# ================================================================

ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
if ! python3 -c "import flash_attn" >/dev/null 2>&1; then
    ATTN_IMPL="sdpa"
fi
echo ">>> Attention backend: ${ATTN_IMPL}"

NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
USE_HF=1 \
QWENVL_BBOX_FORMAT='new' \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1280 \
swift sft \
    --model Qwen/Qwen3.5-27B \
    --dataset '/workspace/data/train_augmented.jsonl' \
    --val_dataset '/workspace/data/val_augmented.jsonl' \
    --external_plugins /workspace/scripts_27b_swift/optimizer.py /workspace/scripts_27b_swift/metric.py \
    --optimizer IMoptimizer \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --strict true \
    --load_from_cache_file true \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --attn_impl "$ATTN_IMPL" \
    --lora_rank 16 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --freeze_vit false \
    --freeze_aligner false \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing true \
    --packing false \
    --truncation_strategy left \
    --gradient_accumulation_steps 2 \
    --eval_steps 150 \
    --save_steps 150 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 3860 \
    --output_dir /workspace/output/swift_27b \
    --warmup_ratio 0.05 \
    --deepspeed zero3 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --lr_scheduler_type cosine \
    --vit_lr 5e-5 \
    --aligner_lr 5e-5 \
    --learning_rate 2e-5 \
    --report_to none

    # === 可选：启用 loss_scale（需要 swift 框架补丁，见 loss_scale.py 注释） ===
    # --loss_scale IMLossScale \
    # --enable_channel_loss true \
    # 启用时需将 loss_scale.py 加入 --external_plugins:
    #   --external_plugins /workspace/scripts_27b_swift/optimizer.py /workspace/scripts_27b_swift/metric.py /workspace/scripts_27b_swift/loss_scale.py
