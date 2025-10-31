#!/bin/bash
# Copyright 2025 the LlamaFactory team.
#
# Example script for extracting OLMoE routing outputs
# This script demonstrates basic usage of extract_olmoe_routing.py

set -e

echo "====================================="
echo "OLMoE Routing Extraction Example"
echo "====================================="

# Configuration
MODEL_PATH="allenai/OLMoE-1B-7B-0924"
DATASET="alpaca_en_demo"
TEMPLATE="default"
MAX_SAMPLES=100
OUTPUT_DIR="routing_outputs"
BATCH_SIZE=4

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Dataset: $DATASET"
echo "  Max samples: $MAX_SAMPLES"
echo "  Batch size: $BATCH_SIZE"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Run extraction (uses Trainer infrastructure, automatically handles multi-GPU if available)
python scripts/extract_olmoe_routing.py \
    --model_name_or_path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --template "$TEMPLATE" \
    --max_samples "$MAX_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE"

echo ""
echo "====================================="
echo "Extraction complete!"
echo "====================================="
echo ""
echo "Output saved to: $OUTPUT_DIR/alpaca_en_demo.npz"
echo ""
echo "To view the results in Python:"
echo "  import numpy as np"
echo "  data = np.load('$OUTPUT_DIR/alpaca_en_demo.npz')"
echo "  print('Number of samples:', len(data.keys()))"
echo "  print('First sample shape:', data['sample_0'].shape)"
echo "  print('Format: (num_layers, valid_seq_len, num_experts)')"
echo ""
