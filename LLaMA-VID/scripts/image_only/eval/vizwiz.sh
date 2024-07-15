#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT="llama-vid/llama-vid-7b-full-336"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llamavid.eval.model_vqa_loader \
    --model-path ./work_dirs/$CKPT \
    --question-file ./data/LLaMA-VID-Eval/vizwiz/llava_test.jsonl \
    --image-folder ./data/LLaMA-VID-Eval/vizwiz/test \
    --answers-file ./data/LLaMA-VID-Eval/vizwiz/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1 &
done

wait

output_file=./data/LLaMA-VID-Eval/vizwiz/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./data/LLaMA-VID-Eval/vizwiz/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./data/LLaMA-VID-Eval/vizwiz/llava_test.jsonl \
    --result-file $output_file \
    --result-upload-file ./data/LLaMA-VID-Eval/vizwiz/answers_upload/$CKPT.json
