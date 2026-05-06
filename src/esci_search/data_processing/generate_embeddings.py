# Code released for anonymous review. License: CC-BY-NC-4.0  


import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import argparse
from pathlib import Path

"""
ESCI Embedding Generation Script

Generates dense vector embeddings for the product catalog using sentence-transformers.
Creates both the embedding file (binary numpy array) and ASIN mapping (index→ASIN lookup).

Multi-GPU support with automatic batch distribution across available devices.

USAGE:
python generate_embeddings.py \
    --data_path data/esci/raw/sampled_item_metadata_esci.jsonl \
    --output_dir data/esci/raw/dense_embeddings_raw \
    --model_name sentence-transformers/all-mpnet-base-v2 \
    --batch_size 1024
"""

def create_embeddings_st_with_mapping(data_path, embedding_path, model_name='all-mpnet-base-v2', batch_size=512, sanity_check=False):
    """Create embeddings using sentence-transformers with multi-GPU acceleration AND ASIN mapping"""
    
    # Setup devices for multi-GPU processing
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(num_gpus)]
        print(f"Using {num_gpus} GPUs: {devices}")
        
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        total_batch_size = batch_size * num_gpus
        print(f"Total effective batch size across all GPUs: {total_batch_size}")
    else:
        devices = "cpu"
        print("CUDA not available, using CPU")
    
    # Load model
    print(f"Loading model: {model_name}")
    if torch.cuda.is_available():
        model = SentenceTransformer(model_name, device="cuda:0")
        print(f"Model loaded on: {next(model.parameters()).device}")
        print(f"Will distribute across devices during encoding: {devices}")
    else:
        model = SentenceTransformer(model_name, device="cpu")
        print(f"Model loaded on CPU")
    
    # Load data AND preserve ASIN order
    print("Loading data...")
    with open(data_path, 'r') as file:
        data = [json.loads(line) for line in file]
    # SANITY CHECK: subsample to 1000 items
    if sanity_check:
        data = data[:50000]
        print(f"⚠️ SANITY CHECK MODE: Using only {len(data)} items")
    
    texts = [item['metadata'].strip() for item in data]
    asins = [item['item'] for item in data]  # PRESERVE ASIN ORDER
    print(f"Loaded {len(texts)} documents")
    
    # Optimize batch size for multi-GPU processing
    if torch.cuda.is_available():
        per_gpu_batch = batch_size
        total_batch_size = per_gpu_batch * torch.cuda.device_count()
        print(f"Per-GPU batch size: {per_gpu_batch}")
        print(f"Total effective batch size: {total_batch_size}")
        batch_size = per_gpu_batch
    
    print(f"Creating embeddings for {len(texts)} documents...")
    
    # Create embeddings with multi-GPU acceleration
    embeddings = model.encode(
        texts, 
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
        device=devices if torch.cuda.is_available() else None
    )
    
    # Save embeddings as binary file
    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
    embeddings.astype(np.float32).tofile(embedding_path)
    
    # SAVE ASIN MAPPING (index -> ASIN)
    mapping_path = str(Path(embedding_path).parent / f"{Path(embedding_path).stem}_asin_mapping.json")
    asin_mapping = {str(i): asin for i, asin in enumerate(asins)}
    with open(mapping_path, 'w') as f:
        json.dump(asin_mapping, f, indent=2)
    
    print(f"Embeddings saved to {embedding_path}")
    print(f"ASIN mapping saved to {mapping_path}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"File size: {os.path.getsize(embedding_path) / (1024*1024):.2f} MB")
    print(f"Mapping size: {len(asin_mapping)} ASINs")
    
    return embeddings, asin_mapping

def monitor_gpu_usage():
    """Monitor GPU memory usage during processing"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1e9
            memory_reserved = torch.cuda.memory_reserved(i) / 1e9
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"GPU {i}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved, {memory_total:.2f}GB total")


EMBED_MODEL_CHOICES = [
    'sentence-transformers/all-mpnet-base-v2',  # default (Sentence-Transformers)
    'BAAI/bge-large-en-v1.5',
    'intfloat/e5-large-v2',
    'princeton-nlp/sup-simcse-roberta-large',
    'BAAI/bge-base-en-v1.5',
    'intfloat/e5-base-v2',
]

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate sentence-transformer embeddings for ESCI item metadata with ASIN mapping."
    )
    p.add_argument(
        "--data_path",
        type=str,
        default="data/esci/metadata/item_catalog.jsonl",  # Default relative path
        help="Path to item catalog (default: data/esci/metadata/item_catalog.jsonl)"
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="data/esci/embeddings",  # Default relative path
        help="Directory for embeddings output (default: data/esci/embeddings)"
    )
    p.add_argument(
        "--model_name",
        type=str,
        choices=EMBED_MODEL_CHOICES,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Encoder to use"
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size"
    )
    p.add_argument(
        '--sanity_check', 
        action='store_true', 
        help='Only process 1000 items for testing'
    )
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Auto-resolve paths relative to project root
    script_dir = Path(__file__).resolve().parent  
    project_root = script_dir.parent.parent.parent
    # Resolve paths from args (relative to project root)
    data_path = project_root / args.data_path if not Path(args.data_path).is_absolute() else Path(args.data_path)
    embeddings_dir = project_root / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Simplified filename: just model name (e.g., "all-mpnet-base-v2.npy")
    model_stem = args.model_name.split('/')[-1]  # Get last part after /
    embedding_path = embeddings_dir / f"{model_stem}.npy"
    mapping_path = embeddings_dir / f"{model_stem}_asin_mapping.json"

    print("=== Config ===")
    print(f"data_path   : {data_path}")
    print(f"embeddings_dir: {embeddings_dir}")
    print(f"model_name  : {args.model_name}")
    print(f"batch_size  : {args.batch_size}")
    print(f"embedding_path: {embedding_path}")
    print(f"mapping_path: {mapping_path}")
    
    # Monitor initial GPU state
    print("\n=== Initial GPU State ===")
    monitor_gpu_usage()

    # Create embeddings with multi-GPU acceleration
    print("\n=== Creating Embeddings with ASIN Mapping ===")
    embeddings, asin_mapping = create_embeddings_st_with_mapping(
    data_path=str(data_path),
    embedding_path=str(embedding_path),
    model_name=args.model_name,
    batch_size=args.batch_size,
    sanity_check=args.sanity_check  
)

    # Monitor final GPU state
    print("\n=== Final GPU State ===")
    monitor_gpu_usage()

    print("\n✅ Embedding creation with ASIN mapping completed!")
    print(f"Sample mapping: {dict(list(asin_mapping.items())[:3])}")
