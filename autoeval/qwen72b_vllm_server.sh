#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: GPU count not specified."
    echo "Usage: $0 <N_gpu>"
    exit 1
fi

N_gpu=$1

if ! [[ "$N_gpu" =~ ^[0-9]+$ ]] || [ "$N_gpu" -eq 0 ]; then
    echo "Error: N_gpu must be a positive integer."
    exit 1
fi

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-72B-Instruct \
    --port 20000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --tensor-parallel-size $N_gpu \
    --trust-remote-code \
    --dtype bfloat16 \
    --limit-mm-per-prompt '{"images": 1, "videos": 0}' \
    --mm-processor-kwargs '{"max_pixels": 262144}'
