#!/bin/bash

# Code released for anonymous review. License: CC-BY-NC-4.0  


# Simple script to generate SFT/DPO datasets
set -e

echo "========================================"
echo "SFT/DPO Dataset Generation Pipeline"
echo "========================================"

# Get paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
ESCI_SEARCH_DIR="$PROJECT_ROOT/src/esci_search"
OUTPUT_DIR="$ESCI_SEARCH_DIR/evals/test_generations_sft_dpo"
DATA_DIR="$PROJECT_ROOT/data/esci"

# Create output dir
mkdir -p "$OUTPUT_DIR"

# Settings
NUM_SAMPLES=${NUM_SAMPLES:-5000}
TEMPERATURE=${TEMPERATURE:-0.2}
BATCH_SIZE=${BATCH_SIZE:-512}
METRIC=${METRIC:-"ap"}

echo "Generating trajectories (run 1/2)..."
python "$SCRIPT_DIR/generate_dpo_sft_data.py" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --bs "$BATCH_SIZE"

sleep 2

echo "Generating trajectories (run 2/2)..."
python "$SCRIPT_DIR/generate_dpo_sft_data.py" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --bs "$BATCH_SIZE"

echo "Creating SFT and DPO datasets..."
python "$SCRIPT_DIR/process_sft_dpo_data.py" \
    --input_dir "$OUTPUT_DIR" \
    --metric "$METRIC" \
    --sft_out "$DATA_DIR/sft_dataset.jsonl" \
    --dpo_out "$DATA_DIR/dpo_dataset.jsonl" \
    --sft_threshold 0.3 \
    --dpo_threshold 0.1 \
    --min_diff 0.05

echo ""
echo "Complete! Check:"
echo "  - CSVs: $OUTPUT_DIR"
echo "  - SFT: $DATA_DIR/sft_dataset.jsonl"
echo "  - DPO: $DATA_DIR/dpo_dataset.jsonl"