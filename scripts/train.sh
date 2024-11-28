output_dir=outputs/mmduet-train_output
mkdir -p $output_dir

torchrun --nproc_per_node 8 --master_port 29506 train.py --deepspeed configs/deepspeed/zero2.json \
    --bf16 true --tf32 true \
    --dataset_config configs/datasets/mmduetit.json \
    --llm_pretrained lmms-lab/llava-onevision-qwen2-7b-ov \
    --num_train_epochs 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 --gradient_checkpointing true \
    --evaluation_strategy no --prediction_loss_only false \
    --save_strategy steps --save_steps 500 --save_total_limit 5 \
    --learning_rate 0.00002 --optim adamw_torch --lr_scheduler_type cosine --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --logging_steps 10 \
    --report_to tensorboard \
    --output_dir $output_dir \
    > $output_dir/train.log 2>&1 &

# check `configs/datasets/mmduetit.json` for datasets used.
# If you want to use your own dataset to train MMDuet, write your own data config file like this file.
