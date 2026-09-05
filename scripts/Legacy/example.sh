QWENVL_BBOX_FORMAT='new' \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1280 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model /home/jupyter/models/Qwen3-VL-4B-Instruct \
    --dataset 'train.jsonl' \
    --val_dataset 'test.jsonl' \
    --custom_register_path template.py \
    --template IMTemplate \
    --external_plugins optimizer.py loss_scale.py metric.py \
    --optimizer IMoptimizer \
    --enable_channel_loss true \
    --loss_scale IMLossScale \
    --metric IMMetric \
    --metric_for_best_model IMMetric \
    --greater_is_better true \
    --strict true \
    --load_from_cache_file true \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
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
    --save_steps 500 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir /home/jupyter/output \
    --warmup_ratio 0.05 \
    --deepspeed zero3 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4\
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 1e-6}'\
    --vit_lr 5e-5 \
    --aligner_lr 5e-5 \
    --learning_rate 2e-5