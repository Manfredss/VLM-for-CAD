NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
QWENVL_BBOX_FORMAT='new' \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1280 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
swift sft \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --dataset 'data/qwen_dataset_5k_v2.jsonl' \
    --custom_register_path template.py \
    --template IMTemplate \
    --enable_channel_loss true \
    --greater_is_better true \
    --strict true \
    --load_from_cache_file true \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --attn_impl flash_attn \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit false \
    --freeze_aligner false \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing true \
    --packing true \
    --padding_free true \
    --truncation_strategy left \
    --gradient_accumulation_steps 2 \
    --eval_steps 500 \
    --save_steps 25 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --deepspeed zero3 \
    --dataset_num_proc 1 \
    --dataloader_num_workers 0 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 1e-6}'\
    --vit_lr 5e-5 \
    --aligner_lr 5e-5 \
    --learning_rate 2e-5 \
    --report_to wandb

    # --loss_scale IMLossScale \
    # --val_dataset 'test.jsonl' \
    # --external_plugins optimizer.py loss_scale.py metric.py \
    # --optimizer IMoptimizer \
    # --metric IMMetric \
    # --metric_for_best_model IMMetric \
    # --per_device_eval_batch_size 1 \
    
