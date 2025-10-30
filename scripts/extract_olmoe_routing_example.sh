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
OUTPUT_FILE="olmoe_routing_sample.jsonl"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Dataset: $DATASET"
echo "  Max samples: $MAX_SAMPLES"
echo "  Output: $OUTPUT_FILE"
echo ""

# Run extraction
python scripts/extract_olmoe_routing.py \
    --model_name_or_path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --template "$TEMPLATE" \
    --max_samples "$MAX_SAMPLES" \
    --save_name "$OUTPUT_FILE" \
    --batch_size 1

echo ""
echo "====================================="
echo "Extraction complete!"
echo "====================================="
echo ""
echo "Output file: $OUTPUT_FILE"
echo ""
echo "To view the results:"
echo "  head -n 1 $OUTPUT_FILE | python -m json.tool"
echo ""
echo "To analyze expert usage:"
echo "  python scripts/api_example/analyze_routing.py --input $OUTPUT_FILE"
echo ""
