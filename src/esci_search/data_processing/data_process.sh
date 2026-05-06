#!/bin/bash

# Code released for anonymous review. License: CC-BY-NC-4.0  


set -e  # Exit on any error

echo "=========================================="
echo "ESCI Data Processing Pipeline"
echo "=========================================="
echo ""

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

cd "$SCRIPT_DIR"

echo "Step 1/5: Creating item metadata catalog..."
echo "Runtime: ~30-60 minutes"
python create_item_metadata.py
echo "✓ Item catalog created"
echo ""

echo "Step 2/5: Generating embeddings..."
echo "Runtime: ~2-3 hours"
python generate_embeddings.py \
    --model_name sentence-transformers/all-mpnet-base-v2 \
    --batch_size 1024
echo "✓ Embeddings generated"
echo ""

echo "Step 3/5: Building FAISS index..."
echo "Runtime: ~2-5 minutes"
python build_index.py
echo "✓ FAISS index built"
echo ""

echo "Step 4/5: Sampling ground truth queries..."
python sample_queries.py \
    --num_queries 40000 \
    --min_matches 3 \
    --max_matches 10 \
    --seed 42
echo "✓ Ground truth queries sampled"
echo ""

echo "Step 5/5: Generating RL training dataset..."
 
python generate_rl_data.py
echo "✓ RL dataset generated"
echo ""

echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Output directory: $PROJECT_ROOT/data/esci/"
echo ""
 