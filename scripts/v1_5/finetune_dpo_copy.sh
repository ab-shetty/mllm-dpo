#!/bin/bash
# The following parameter were found to have the best performance on llava bench out of differen beta, aver  average and learning rate
# Model Name: llava-lora-dpo-beta-0.1-lr-5e-5-avg-False
# Beta: 0.1
# Use Average: False
# Learning Rate: 5e-5
DATA_PATH=/content/mllm-dpo/playground/data/dpo/chains.json
IMAGE_FOLDER=/content/mllm-dpo/playground/data 
run_name=mllm-dpo-2k
ouput_dir=/content/checkpoints/${run_name}
# Notice that I am loading the latest model checkopint 
model_name=/content/checkpoints/llava-v1.5-7b # Use the previous model checkpoint
deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ${model_name} \
    --version v1 \
    --task DPO --dpo_use_average False --dpo_beta 0.1 \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ${ouput_dir} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 5e-5\
    --is_multimodal True \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${run_name} \