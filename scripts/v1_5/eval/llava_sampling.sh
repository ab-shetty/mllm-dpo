#!/bin/bash
# Loop to run the command 8 times with different output files
for i in {0..3}  # Change the range to 0..7 to run it 8 times
do
    gpu=$((i % 4))  # Use modulo 4 to cycle through the 4 available GPUs
    CUDA_VISIBLE_DEVICES=$gpu python -m llava.eval.model_vqa_loader \
        --model-path ./checkpoints/llava-v1.5-13b \
        --question-file ./playground/data/lrv_instruct/lrv_filter_cap_10kto13k.json \
        --image-folder /home/ubuntu/latest_llava/llava_1dot5data/coco/train2017/lrv_image_folder \
        --answers-file "./playground/data/lrv_instruct/answers/lrv_10kto13k_$i.json" \
        --temperature 1.2 \
        --conv-mode vicuna_v1 \
        --max_new_tokens 1024 &
done
wait
