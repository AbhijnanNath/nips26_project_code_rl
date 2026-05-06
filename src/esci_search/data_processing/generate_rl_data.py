# Code released for anonymous review. License: CC-BY-NC-4.0  


import json
import random
import argparse
from collections import defaultdict
from typing import Dict, Tuple 
from datasets import Dataset, DatasetDict
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def load_metadata(metadata_path: str) -> Dict[str, str]:
    """Load product metadata and create ASIN to category mapping."""
    metadata = []
    with open(metadata_path, 'r') as f:
        for line in f:
            metadata.append(json.loads(line))
    
    asin_to_category = {}
    for item in metadata:
        asin = item.get('item')
        category = item.get('product_category') or item.get('category') or 'unknown'
        asin_to_category[asin] = category
    
    return asin_to_category

def create_category_mapping(data: pd.DataFrame, asin_to_category: Dict[str, str]) -> pd.DataFrame:
    """Map dataset rows to product categories based on target ASINs."""
    rows_with_categories = []
    for idx, row in enumerate(data['reward_model']):
        targets = row['ground_truth']['target']
        for asin in targets:
            cat = asin_to_category.get(asin, 'unknown')
            rows_with_categories.append({'asin': asin, 'category': cat, 'row_idx': idx})
    
    return pd.DataFrame(rows_with_categories)

def group_items_by_category(asin_to_category):
    """Create reverse mapping: category -> list of ASINs."""
    category_to_asins = defaultdict(list)
    for asin, category in asin_to_category.items():
        category_to_asins[category].append(asin)
    return category_to_asins

def sample_diverse_negatives(target_asin, asin_to_category, category_to_asins, n_total=100):
    """
    Sample diverse negatives:
    - 50 from same category (in-domain hard negatives)
    - 30 from related categories (medium difficulty)
    - 20 random from catalog (easy negatives)
    """
    target_category = asin_to_category.get(target_asin, 'unknown')
    negatives = []
    
    # Same category negatives (excluding target)
    same_cat_items = [a for a in category_to_asins.get(target_category, []) if a != target_asin]
    n_same = min(50, len(same_cat_items))
    negatives.extend(random.sample(same_cat_items, n_same))
    
    # Related categories (those sharing prefix with target category)
    related_cats = [c for c in category_to_asins.keys() 
                   if c != target_category and c.split()[0] == target_category.split()[0]]
    related_items = []
    for cat in related_cats:
        related_items.extend(category_to_asins[cat])
    related_items = [a for a in related_items if a not in negatives and a != target_asin]
    n_related = min(30, len(related_items))
    if related_items:
        negatives.extend(random.sample(related_items, n_related))
    
    # Random negatives from entire catalog
    all_asins = list(asin_to_category.keys())
    random_pool = [a for a in all_asins if a not in negatives and a != target_asin]
    n_random = min(20, len(random_pool))
    if random_pool:
        negatives.extend(random.sample(random_pool, n_random))
    
    return negatives

def stratified_split(cats_df: pd.DataFrame, test_ratio: float = 0.1) -> Tuple[list, list]:
    """Create stratified train/test split based on categories."""
    test_indices = []
    train_indices = []
    
    for cat in cats_df['category'].unique():
        if cat == 'unknown':
            continue
        
        cat_indices = cats_df[cats_df['category'] == cat]['row_idx'].unique().tolist()
        n_test = max(1, int(len(cat_indices) * test_ratio))
        
        random.shuffle(cat_indices)
        test_indices.extend(cat_indices[:n_test])
        train_indices.extend(cat_indices[n_test:])
        
        print(f"{cat}: {len(cat_indices)} total -> {n_test} test, {len(cat_indices)-n_test} train")
    
    # Remove duplicates and ensure no overlap
    test_indices = list(set(test_indices))
    train_indices = list(set(train_indices) - set(test_indices))
    
    return train_indices, test_indices

def verify_index_coverage(data: pd.DataFrame, index_mapping_path: str) -> float:
    """Verify what percentage of target ASINs exist in FAISS index."""
    with open(index_mapping_path, 'r') as f:
        mapping = json.load(f)
    asins_in_index = set(mapping.values())
    
    test_asins = set()
    for _, row in data.iterrows():
        test_asins.update(row['reward_model']['ground_truth']['target'])
    
    coverage = len(test_asins & asins_in_index) / len(test_asins) if test_asins else 0
    return coverage


def create_rl_dataset_with_pools(queries_data, asin_to_category, category_to_asins):
    """Convert queries to RL format with candidate pools."""
    rl_data = []
    
    system_prompt = {
        'role': 'system',
        'content': """You are an expert in query expansion for product retrieval. Your task: enrich the customer's query with product attributes and context that improve dense retrieval accuracy.
    Important: RETAIN the original query terms and ADD relevant product details. Your expansion should be a natural phrase combining the original query with product attributes.
    Strategy:
    1. Keep all original query terms
    2. Add product category if not explicit (e.g., "laptop" for vague queries)
    3. Include key distinguishing attributes: brand, model, specs, materials, use-case

    Your expansion should be a natural phrase combining the original query with product attributes, NOT a first-person request.
    Format your response as follows:
    - First, analyze the query: What product category is this? What key attributes would help distinguish the right item from similar products? What buyer intent or use-case is implied?
    - Then, create an enriched search query by naturally weaving in the relevant attributes and context while keeping the original terms."""
        }
    
    for idx, item in enumerate(tqdm(queries_data, desc="Processing queries")):
        query = item['query']
        target_asins = item['ground_truth_product_ids']

        target_asin = None
        for asin in target_asins:
            if asin in asin_to_category:
                target_asin = asin
                break
        if not target_asin:
            continue

        negatives = sample_diverse_negatives(target_asin, asin_to_category, category_to_asins, n_total=99)
        candidate_pool = [target_asin] + negatives
                
        user_msg = {
            'role': 'user',
            'content': query
        }
        
        rl_data.append({
            'sample_idx': idx,
            'question': query,
            'prompt': [system_prompt, user_msg],
            'answer': target_asins,
            'target_item_id': target_asin,
            'candidate_pool': candidate_pool
        })
    
    return rl_data

if __name__ == '__main__':
    # Setup paths  
    script_dir = Path(__file__).resolve().parent  
    project_root = script_dir.parent.parent.parent
    
    metadata_path = project_root / 'data' / 'esci' / 'metadata' / 'item_catalog.jsonl'
    ground_truth_path = project_root / 'data' / 'esci' / 'ground_truth'
    output_path = project_root / 'data' / 'esci' / 'rl_dataset'
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print("Loading metadata...")
    asin_to_category = load_metadata(str(metadata_path))
    
    # Find the ground truth file (it will have format query_ground_truth_N.json)
    gt_files = list(ground_truth_path.glob('query_ground_truth_*.json'))
    if not gt_files:
        raise FileNotFoundError(f"No ground truth files found in {ground_truth_path}")
    
    sampled_queries_path = gt_files[0]  # Use the first/only one found
    print(f"Loading ground truth from: {sampled_queries_path}")
    
    with open(sampled_queries_path, 'r') as f:
        queries_data = json.load(f)
    
    print(f"Loaded {len(queries_data)} queries")
    # Create category mapping
    category_to_asins = {}
    for asin, cat in asin_to_category.items():
        category_to_asins.setdefault(cat, []).append(asin)
    
    # Split 90/10
    train_size = int(0.9 * len(queries_data))
    train_queries = queries_data[:train_size]
    test_queries = queries_data[train_size:]
    
    print(f"Train: {len(train_queries)}, Test: {len(test_queries)}")
    
    # Create RL datasets
    train_rl = create_rl_dataset_with_pools(train_queries, asin_to_category, category_to_asins)
    test_rl = create_rl_dataset_with_pools(test_queries, asin_to_category, category_to_asins)
    
    print("\nTrain dataset:")
    print(f"Size: {len(train_rl)}")
    print("\nSample:")
    print(f"Question: {train_rl[0]['question']}")
    print(f"Target ASIN: {train_rl[0]['target_item_id']}")
    
    # Save as HuggingFace DatasetDict
    rl_dataset = DatasetDict({
        'train': Dataset.from_list(train_rl),
        'test': Dataset.from_list(test_rl),
    })
    
    rl_dataset.save_to_disk(str(output_path))
    print(f"Dataset saved to: {output_path}")