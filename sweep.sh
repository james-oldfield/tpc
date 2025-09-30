#!/bin/bash

LAYER=40

MODEL_NAME="google/gemma-3-27b-it"
# MODEL_NAME="meta-llama/Llama-3.2-3B"
# MODEL_NAME="openai/gpt-oss-20b"
# MODEL_NAME="Qwen/Qwen3-30B-A3B-Base"

DATASET="WildGuard" # options: [WildGuard, BeaverTrails]
# DATASET="BeaverTrails" # options: [WildGuard, BeaverTrails]

# example sweep:

for RANK in 64 128 256; do
    python -u sweep_monitors.py \
        --rank $RANK \
        --layer_idx $LAYER \
        --train_mode "progressive" \
        --max_order 5 \
        --model_name $MODEL_NAME \
        --dataset $DATASET \
        --max_epochs 50 \
        --pool_type 'mean'
done
