#!/bin/bash
# Copyright 2025 the LlamaFactory team.
#
# Example script for extracting OLMoE routing outputs
# This is a simple example showing how to use extract_olmoe_routing.py

set -e

echo "====================================="
echo "OLMoE Routing Extraction Example"
echo "====================================="

# Configuration
MODEL_PATH="allenai/OLMoE-1B-7B-0924"
DATASET="alpaca_en_demo"
TEMPLATE="default"
MAX_SAMPLES=10
OUTPUT_DIR="routing_outputs"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Dataset: $DATASET"
echo "  Max samples: $MAX_SAMPLES"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Run extraction
python scripts/extract_olmoe_routing.py \
    --model_name_or_path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --template "$TEMPLATE" \
    --max_samples "$MAX_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 1

echo ""
echo "====================================="
echo "Extraction complete!"
echo "====================================="
echo ""
echo "Output saved to: $OUTPUT_DIR/alpaca_en_demo.npz"
echo ""
echo "To view the results:"
echo "  python -c \"import numpy as np; data = np.load('$OUTPUT_DIR/alpaca_en_demo.npz'); print('Samples:', list(data.keys())[:5])\""
echo ""
