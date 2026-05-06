# Code released for anonymous review. License: CC-BY-NC-4.0  


from collections import defaultdict
import json
import random
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from pathlib import Path

"""
Ground Truth Query Dataset Generator for ESCI

Creates a curated dataset of search queries with known correct product matches for evaluation.
Samples queries from the ESCI dataset that have multiple verified exact matches, ensuring
high-quality ground truth for computing retrieval metrics (NDCG, MRR, Recall@k).

FILTERING LOGIC:
1. Extracts queries labeled 'Exact' in ESCI (US locale, small_version=1)
2. Groups products by query and counts exact matches per query
3. Filters queries with 3-10 exact matches (avoids too few or too many)
4. Validates semantic relevance: requires ≥30% token overlap between query and product titles
5. Samples specified number of queries (default: 100) with fixed random seed for reproducibility

OUTPUT FORMAT:
{
  "query": "desk with chair",
  "query_id": "...",
  "num_exact_matches": 6,
  "exact_matches": [{product_id, product_title}, ...],
  "ground_truth_product_ids": ["ASIN1", "ASIN2", ...]  # Used for NDCG calculation
}
USAGE:
python sample_queries.py  # Uses defaults (40000 queries)
python sample_queries.py --num_queries 100  # Small test set
"""

def create_query_ground_truth_dataset(num_queries=100, min_matches=2, max_matches = 10, output_path='query_ground_truth_dataset.json'):
    """
    Create a dataset of queries with their ground truth exact matches for NDCG evaluation.
    
    Args:
        num_queries: Number of unique queries to include
        min_matches: Minimum number of exact matches required per query
        output_path: Path to save the dataset
    """
    
    print("Loading ESCI dataset...")
    raw_dataset = load_dataset("tasksource/esci") #downlaod raw data from huggingface datasets. 
    
    # Group exact matches by query
    query_to_matches = defaultdict(list)
 
    print("Processing ESCI data to group exact matches by query...")
    for item in tqdm(raw_dataset['train'], desc="Processing ESCI items"):
        if item['esci_label'] == 'Exact' and item['product_locale'] == 'us' and item['small_version'] ==1:
            query = item['query'].strip().lower()
            query_to_matches[query].append({
                'product_id': item['product_id'],
                'product_title': item['product_title'],
                'query_id': item['query_id']
            })
    
    # Filter queries with sufficient exact matches
    print(f"Filtering queries with at least {min_matches} exact matches...")
    filtered_queries = {
        query: matches for query, matches in query_to_matches.items() 
        if len(matches) >= min_matches and len(matches) <= max_matches
    }
    valid_queries = {}
    for query, matches in filtered_queries.items():
        # Check if at least one match has >30% token overlap with query
        query_tokens = set(query.lower().split())
        has_overlap = any(
            len(query_tokens & set(m['product_title'].lower().split())) >= max(2, len(query_tokens) * 0.3)
            for m in matches
        )
        if has_overlap:
            valid_queries[query] = matches

    print(f"Filtered from {len(filtered_queries)} to {len(valid_queries)} queries with title overlap")
    filtered_queries = valid_queries
        
    print(f"Found {len(filtered_queries)} queries with >= {min_matches} exact matches")
    
    # Sample queries for the dataset
    if len(filtered_queries) < num_queries:
        print(f"Warning: Only {len(filtered_queries)} queries available, using all of them")
        selected_queries = list(filtered_queries.keys())
    else:
        selected_queries = random.sample(list(filtered_queries.keys()), num_queries)
    
    query_lengths = [len(q.split()) for q in selected_queries]
    query_char_lengths = [len(q) for q in selected_queries]
    print(f"\nQuery Length Statistics:")
    print(f"Word count - Min: {min(query_lengths)}, Max: {max(query_lengths)}, Mean: {np.mean(query_lengths):.1f}, Median: {np.median(query_lengths):.1f}")
    print(f"Char count - Min: {min(query_char_lengths)}, Max: {max(query_char_lengths)}, Mean: {np.mean(query_char_lengths):.1f}")
    
 
    length_bins = {
        '1-2 words': sum(1 for l in query_lengths if l <= 2),
        '3-4 words': sum(1 for l in query_lengths if 3 <= l <= 4),
        '5-6 words': sum(1 for l in query_lengths if 5 <= l <= 6),
        '7+ words': sum(1 for l in query_lengths if l >= 7)
    }

    print(f"\nLength Distribution:")
    for bin_name, count in length_bins.items():
        pct = 100 * count / len(query_lengths)
        print(f"  {bin_name}: {count} ({pct:.1f}%)")

    # Show examples from each bin
    print(f"\nSample queries by length:")
    for l in [2, 3, 5, 7]:
        examples = [q for q in selected_queries if len(q.split()) == l]
        if examples:
            print(f"  {l} words: '{examples[0]}'")

    # Create the dataset
    dataset = []
    
    print("Creating query ground truth dataset...")
    for query in tqdm(selected_queries, desc="Building dataset"):
        matches = filtered_queries[query]
        
        # Deduplicate products (same product can appear multiple times)
        unique_matches = {}
        for match in matches:
            product_id = match['product_id']
            if product_id not in unique_matches:
                unique_matches[product_id] = match
        
        dataset_entry = {
            'query': query,
            'query_id': matches[0]['query_id'],  # Use first query_id as representative
            'num_exact_matches': len(unique_matches),
            'exact_matches': [
                {
                    'product_id': product_id,
                    'product_title': match['product_title']
                }
                for product_id, match in unique_matches.items()
            ],
            'ground_truth_product_ids': list(unique_matches.keys())  # For NDCG calculation
        }
        dataset.append(dataset_entry)
    
    # Sort by number of exact matches (descending) for interesting test cases
    dataset.sort(key=lambda x: x['num_exact_matches'], reverse=True)
    
    # Save dataset
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Print statistics
    match_counts = [entry['num_exact_matches'] for entry in dataset]
    print(f"\nDataset Statistics:")
    print(f"Total queries: {len(dataset)}")
    print(f"Average exact matches per query: {sum(match_counts)/len(match_counts):.1f}")
    print(f"Min exact matches: {min(match_counts)}")
    print(f"Max exact matches: {max(match_counts)}")
    print(f"Dataset saved to: {output_path}")
    return dataset

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ground truth query dataset from ESCI with multiple exact matches."
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=80000,
        help="Number of queries to sample (default: 80000)"
    )
    parser.add_argument(
        "--min_matches",
        type=int,
        default=3,
        help="Minimum exact matches per query (default: 3)"
    )
    parser.add_argument(
        "--max_matches",
        type=int,
        default=10,
        help="Maximum exact matches per query (default: 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    import argparse
    args = parse_args()
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Setup paths
    script_dir = Path(__file__).resolve().parent  
    project_root = script_dir.parent.parent.parent
    raw_esci_path = project_root / 'data' / 'esci' / 'metadata'
    proc_esci_path = project_root / 'data' / 'esci' / 'ground_truth'
    proc_esci_path.mkdir(parents=True, exist_ok=True)
    
    metadata_path = raw_esci_path / 'item_catalog.jsonl'
    output_path = proc_esci_path / f'query_ground_truth_{args.num_queries}.json'
    
    dataset = create_query_ground_truth_dataset(
        num_queries=args.num_queries,
        min_matches=args.min_matches,
        max_matches=args.max_matches,
        output_path=str(output_path)
    )