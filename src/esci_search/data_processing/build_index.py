# Code released for anonymous review. License: CC-BY-NC-4.0  


import faiss
import numpy as np
import os
import json
import time
from tqdm import tqdm
import sys
from datetime import datetime
from pathlib import Path
import argparse

"""
FAISS Index Builder for Dense Retrieval

Builds an HNSW (Hierarchical Navigable Small World) FAISS index from pre-computed embeddings.
The index enables fast approximate nearest neighbor search for product retrieval during RL training.

INPUTS:
- Embedding file (.npy): Binary float32 array of shape (N, D) where N=num_items, D=embedding_dim
- ASIN mapping (.json): Index→ASIN lookup created during embedding generation

OUTPUTS:
- FAISS index (.bin): Searchable HNSW index for dense retrieval
- Metadata (.json): Index configuration (dim, num_vectors, M, ef_construction, build_time)
- ASIN mapping copy (.json): Copied to index directory for easy access

Auto-detects embedding dimensions (768 for all-mpnet-base-v2, 1024 for simcse-large, etc.)

USAGE:
python build_index.py  # Uses defaults from data/esci/embeddings/
python build_index.py --embeddings custom/path.npy --output_dir custom/index/ --M 64
"""

def build_faiss_hnsw_index_with_mapping(embedding_path, output_index_path, M=32, ef_construction=200):
    # Build index using existing function
    index = build_faiss_hnsw_index(embedding_path, output_index_path, M, ef_construction)
    
    # Copy existing ASIN mapping to index directory
    mapping_source = embedding_path.replace('.simcse-largeCLS', '_asin_mapping.json')
    mapping_dest = output_index_path.replace('.bin', '_asin_mapping.json')
    
    if os.path.exists(mapping_source):
        import shutil
        shutil.copy(mapping_source, mapping_dest)
        print(f"ASIN mapping copied to {mapping_dest}")
    else:
        print(f"Warning: ASIN mapping not found at {mapping_source}")
    
    return index

def build_faiss_hnsw_index(embedding_path, output_index_path, M=32, ef_construction=200):
    """Build FAISS HNSW index with auto-detected dimensions."""
    start_time = time.time()
    
    # Auto-detect embedding dimension
    embeddings = np.fromfile(embedding_path, dtype=np.float32)
    # Try common dimensions
    for dim in [768, 1024, 384, 512]:
        if len(embeddings) % dim == 0:
            embeddings = embeddings.reshape(-1, dim)
            break
    
    print(f"Detected embedding shape: {embeddings.shape}")
    faiss.normalize_L2(embeddings)
    
    # Build index
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(embedding_dim, M)
    index.hnsw.efConstruction = ef_construction
    
    add_embeddings_with_progress(index, embeddings, batch_size=10000)
    
    # Save index
    faiss.write_index(index, output_index_path)
    
    # Save metadata
    metadata = {
        'embedding_dim': embedding_dim,
        'num_vectors': embeddings.shape[0],
        'M': M,
        'ef_construction': ef_construction,
        'normalized': True,
        'build_time_seconds': time.time() - start_time
    }
    
    metadata_path = output_index_path.replace('.bin', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Index saved to {output_index_path}")
    return index
    
def add_embeddings_with_progress(index, embeddings, batch_size=10000):
    """
    Add embeddings to FAISS index with progress bar.
    """
    total_vectors = len(embeddings)
    # Calculate number of batches
    num_batches = (total_vectors + batch_size - 1) // batch_size
    
    print(f"Adding {total_vectors:,} vectors in {num_batches} batches of {batch_size:,}")
    
    # Progress bar for batches
    with tqdm(total=num_batches, desc="Building index", unit="batch") as pbar:
        for i in range(0, total_vectors, batch_size):
            end_idx = min(i + batch_size, total_vectors)
            batch = embeddings[i:end_idx]
            
            # Add batch to index
            index.add(batch)
            
            # Update progress
            pbar.set_postfix({
                'vectors': f"{end_idx:,}/{total_vectors:,}",
                'batch_size': len(batch)
            })
            pbar.update(1)
    
    print(f"Successfully added all {total_vectors:,} vectors to index")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build FAISS HNSW index for dense retrieval with ASIN mapping."
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default="data/esci/embeddings/all-mpnet-base-v2.npy",
        help="Path to embedding file (default: data/esci/embeddings/all-mpnet-base-v2.npy)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/esci/index",
        help="Output directory for index (default: data/esci/index)"
    )
    parser.add_argument(
        "--M", type=int, default=32, help="HNSW M parameter"
    )
    parser.add_argument(
        "--ef_construction", type=int, default=200, help="HNSW efConstruction"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Resolve paths from project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent
    
    embedding_path = project_root / args.embeddings
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output files
    model_name = embedding_path.stem  # e.g., "all-mpnet-base-v2"
    index_path = output_dir / f"{model_name}_faiss.bin"
    mapping_dest = output_dir / f"{model_name}_asin_mapping.json"
    
    print(f"Embeddings: {embedding_path}")
    print(f"Index output: {index_path}")
    
    # Build index
    build_faiss_hnsw_index(
        str(embedding_path),
        str(index_path),
        M=args.M,
        ef_construction=args.ef_construction
    )
    
    # Copy ASIN mapping
    mapping_source = embedding_path.parent / f"{model_name}_asin_mapping.json"
    if mapping_source.exists():
        import shutil
        shutil.copy(mapping_source, mapping_dest)
        print(f"Mapping copied to {mapping_dest}")

if __name__ == "__main__":
    main()
 