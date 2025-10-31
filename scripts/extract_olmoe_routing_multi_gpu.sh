#!/bin/bash

# OLMoE Routing Extraction - Multi-GPU Example
# This script demonstrates how to use multiple GPUs to accelerate routing extraction

# Configuration
MODEL_PATH="allenai/OLMoE-1B-7B-0924"
DATASET="alpaca_en_demo"
DATASET_DIR="data"
TEMPLATE="default"
OUTPUT_DIR="routing_outputs"
NUM_GPUS=4  # 修改为你想使用的GPU数量

echo "=========================================="
echo "OLMoE Multi-GPU Routing Extraction"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET"
echo "Number of GPUs: $NUM_GPUS"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# Method 1: 使用 accelerate launch 命令行参数
echo -e "\nMethod 1: Using accelerate launch with command-line arguments"
accelerate launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision=no \
    scripts/extract_olmoe_routing.py \
    --model_name_or_path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --dataset_dir "$DATASET_DIR" \
    --template "$TEMPLATE" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 1

echo -e "\n=========================================="
echo "Extraction completed!"
echo "=========================================="
