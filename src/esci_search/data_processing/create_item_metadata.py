# Code released for anonymous review. License: CC-BY-NC-4.0  


import os
import re
import html
import json
import requests
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import random
import argparse
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_neg', type=int, default=50, help='the number of negative samples')
    parser.add_argument('--output_path', type=str, default='data/esci/raw')
    parser.add_argument('--n_workers', type=int, default=16)
    return parser.parse_args()


def get_asin2category():
    filepath = hf_hub_download(
        repo_id='McAuley-Lab/Amazon-Reviews-2023',
        filename='asin2category.json',
        repo_type='dataset'
    )
    with open(filepath, 'r') as file:
        asin2category = json.loads(file.read())
    return asin2category


def clean_text(raw_text):
    if isinstance(raw_text, list):
        cleaned_text = ' '.join(raw_text)
    elif isinstance(raw_text, dict):
        cleaned_text = str(raw_text)
    else:
        cleaned_text = raw_text
    cleaned_text = html.unescape(cleaned_text)
    cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'
    return cleaned_text


def clean_metadata(example):
    meta_text = ''
    features_needed = ['title', 'description']
    for feature in features_needed:
        if feature in example and example[feature] is not None:
            meta_text += clean_text(example[feature]) + ' '
    example['cleaned_metadata'] = meta_text.replace('\t', ' ')
    return example


def download_category_metadata_with_cleaning(category, category2items, output_file):
    """Download and process metadata for a specific category with cleaning"""
    
    category_clean = category.replace(' ', '_').replace('&', 'and')
    url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_{category_clean}.jsonl"
    
    print(f"Processing category: {category}")
    print(f"Expected items: {len(category2items[category])}")
    
    try:
        response = requests.get(url, stream=True, timeout=300)  # 5 min timeout
        if response.status_code != 200:
            print(f"Failed to download {category}: {response.status_code}")
            return 0
        
        items_written = 0
        lines_processed = 0
        target_items = len(category2items[category])
        
        for line in tqdm(response.iter_lines(decode_unicode=True), desc=f"Processing {category}"):
            lines_processed += 1
            
            # Early exit if we've found all needed items
            if items_written >= target_items:
                print(f"Found all {target_items} items, stopping early")
                break
                
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                item_id = data.get('parent_asin')
                
                if item_id and item_id in category2items[category]:
                    cleaned_data = clean_metadata(data)
                    
                    output_data = {
                        'item': item_id,
                        'category': category,
                        'metadata': cleaned_data['cleaned_metadata']
                    }
                    output_file.write(json.dumps(output_data) + '\n')
                    items_written += 1
                    
                    # Progress update
                    if items_written % 10000 == 0:
                        print(f"Found {items_written}/{target_items} items")
                    
            except json.JSONDecodeError:
                continue
                
        print(f"Wrote {items_written} items for {category}")
        return items_written
        
    except Exception as e:
        print(f"Error processing {category}: {e}")
        return 0

 

def get_existing_categories(metadata_file_path):
    existing_categories = set()
    if os.path.exists(metadata_file_path):
        with open(metadata_file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing_categories.add(data['category'])
                except:
                    continue
        print(f"Found existing categories: {existing_categories}")
    else:
        print("No existing metadata file found")
    
    return existing_categories

def create_metadata_from_jsonl_files(category2items, output_path):
    """Create metadata file using JSONL files from HuggingFace"""
    metadata_file_path = os.path.join(output_path, 'sampled_item_metadata_esci.jsonl') 
    # Get categories that already exist
    existing_categories = get_existing_categories(metadata_file_path)
    # Filter out already processed categories
    categories_to_process = [cat for cat in category2items.keys() if cat not in existing_categories]
    print(f"Categories to process: {categories_to_process}")
    print(f"Skipping existing categories: {existing_categories}")
    if not categories_to_process:
        print("All categories already processed!")
        return metadata_file_path

    # Open in append mode to add to existing file
    with open(metadata_file_path, 'a') as f:   
        total_items = 0
        
        for category in categories_to_process:
            print(f"Processing new category: {category}")
            items_written = download_category_metadata_with_cleaning(category, category2items, f)
            total_items += items_written
            
    print(f"Added {total_items} new items")
    return metadata_file_path
 
if __name__ == '__main__':

    script_dir = Path(__file__).resolve().parent  
    project_root = script_dir.parent.parent.parent
    output_path = project_root / 'data' / 'esci' / 'raw'
    output_file = output_path / 'sampled_item_metadata_esci.jsonl'

    n_neg = 50
    n_workers = 16
    # Collect potential negative samples
    asin2category = get_asin2category()
    category2item = defaultdict(list)
    for asin, cat in tqdm(asin2category.items()):
        category2item[cat].append(asin)

    # Create output directory
    os.makedirs(os.path.join(output_path),exist_ok=True)
    # Filter ESCI dataset
    query_data = {
        'qid': [],
        'query': [],
        'item_id': []
    }
    # Step 1: Process ESCI and collect needed items first
    candidate_item = set()
    category2items = defaultdict(set)

    print("Processing ESCI dataset to identify needed items...")
    raw_dataset = load_dataset("tasksource/esci")
    qid = 0
    for line in tqdm(raw_dataset['train']):
        item_id = line['product_id']
        
        if item_id not in asin2category:
            continue
        cat = asin2category[item_id]
        if 'Unknown' in cat:
            continue
        if line['product_locale'] != 'us':
            continue
        if line['esci_label'] != 'Exact':
            continue
        if line['small_version'] != 1:
            continue
        
        candidate_item.add(item_id)
        category2items[cat].add(item_id)
        neg_items = random.sample(category2item[cat], n_neg)
        for neg_item in neg_items:
            candidate_item.add(neg_item)
            category2items[cat].add(neg_item)
        query = line['query'].strip().replace('\t', ' ')

        query_data['qid'].append(qid)
        query_data['query'].append(query)
        query_data['item_id'].append(item_id)

        qid += 1

    print(f"Found {len(candidate_item)} items across {len(category2items)} categories")
    query_data_hf = Dataset.from_dict(query_data)
    query_file_path = os.path.join(output_path, 'train.csv')
    query_data_hf.to_csv(query_file_path)
    print(f'Total number of training data: {len(query_data["qid"])}')
    create_metadata_from_jsonl_files(category2items, output_path)
#    