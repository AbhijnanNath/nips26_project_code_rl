#!/usr/bin/env python3

# Code released for anonymous review. License: CC-BY-NC-4.0  


import os
import re
import json
import math
import random
random.seed(42)
import time
import pandas as pd
import argparse
from tqdm import tqdm
from typing import List, Tuple, Iterable
import torch
from dense_search.search import SearchRewardFunction
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from collections import OrderedDict

# ================
# Project Structure
# ================
def get_project_paths():
    """
    Establishes project directory structure relative to this script.
    Assumes structure: PROJECT_ROOT/src/esci_search/data_processing/this_script.py
    """
    script_dir = Path(__file__).resolve().parent      # data_processing/
    esci_search_dir = script_dir.parent               # esci_search/
    src_dir = esci_search_dir.parent                  # src/
    project_root = src_dir.parent                     # LLM-Seq-Shapley-Owen-PO/
    
    return {
        "project_root": project_root,
        "src_dir": src_dir,
        "esci_search_dir": esci_search_dir,
        "data_dir": project_root / "data" / "esci",
        "output_dir": esci_search_dir / "evals" / "test_generations_sft_dpo",
    }



# ================
# Config
# ================
BASE_MODEL    = "Qwen/Qwen2.5-7B-Instruct"  # change if needed
 
# Generation settings (tuned for 8×H200; adjust if needed)
TEMPERATURE     = 0.2
BATCH_SIZE      = 512
MAX_NEW_TOKENS  = 512

 

# Local baselines (name -> directory with checkpoint-* subdirs)
HF_BASELINES = {
 
    "Qwen2.5-7B-Instruct":    "Qwen/Qwen2.5-7B-Instruct",
 
}

MODEL_BS_HINT = {
 
    "Qwen2.5-7B-Instruct":   800,
 
}

# --- Fixed-K metrics helpers ---
from collections import OrderedDict

    # choose fixed K’s you want
FIXED_KS = [10, 50]

# ================
# Utils
# ================

def load_hf_model(model_id: str):
    print(f"Loading base model: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="left", truncation_side="left")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",          # let HF shard across your 8x H200s
        torch_dtype="auto",         # bfloat16 if available
        trust_remote_code=True,
    )
    # align gen config
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.eos_token_id = tok.eos_token_id
    return model, tok

def generate_responses_and_scores(
    model,
    tokenizer,
    prompt_messages_list,
    *,
    search_fn,
    # aligned full columns for the same order as prompt_messages_list:
    sample_idxs,           # List[int]
    questions,             # List[str or None]
    answers,               # List[str or None]
    target_item_ids,       # List[str | List[str]]
    # run/baseline metadata for DF:
    baseline: str,
    best_ckpt: str,
    temperature: float,
    max_new_tokens: int = 800,
    batch_size: int = 16,
):
    """
    Generates responses in batches; after each batch, computes NDCG via `search_fn`
    using candidate pools, and returns both the concatenated responses and a list
    of per-batch DataFrames (caller can concat & save).
    """


    def _dedup(lst):
        return list(OrderedDict.fromkeys(lst or []))

    def _trunc(lst, k):
        return (lst or [])[:k]

    def _ndcg_at_k(items, targets, k):
        items_k = _trunc(_dedup(items), k)
        return search_fn._calculate_ndcg(items_k, targets, k=k)

    def _recall_at_k(items, targets, k):
        items_k = _trunc(_dedup(items), k)
        return search_fn._calculate_recall_at_k(items_k, targets, k=k)

    def _mrr_at_k(items, targets, k):
        items_k = _trunc(_dedup(items), k)
        return search_fn._mrr(items_k, targets, k=k)

    def _ap_at_k(items, targets, k):
        items_k = _trunc(_dedup(items), k)
        return search_fn._average_precision(items_k, targets, k=k)

    all_responses = []
    batch_dfs = []
    total = len(prompt_messages_list)
    total_batches = math.ceil(total / batch_size)
    temperature = 0.2
    print(f"\nGenerating responses for {total} prompts in {total_batches} batches (bs={batch_size})...")

    for i in tqdm(range(0, total, batch_size), total=total_batches, desc="Generating", ncols=100):
        # -------- slice this batch --------
        j = i + batch_size
        batch_prompts        = prompt_messages_list[i:j]
        batch_sample_idxs    = sample_idxs[i:j]
        batch_questions      = questions[i:j]
        batch_answers        = answers[i:j]
        batch_target_item_ids= target_item_ids[i:j]

        # -------- apply chat template --------
        formatted = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in batch_prompts
        ]

        # -------- tokenize / generate --------
        inputs = tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        # -------- decode each row --------
        in_lens = inputs.attention_mask.sum(dim=1).tolist()
        batch_responses = []
        for row_idx, out_ids in enumerate(outputs):
            gen_ids = out_ids[int(in_lens[row_idx]):]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            batch_responses.append(text)
            all_responses.append(text)
                # Right before the search_fn call, add:
        print(f"Target item IDs sample: {batch_target_item_ids[:3]}")
        print(f"Types: {[type(t) for t in batch_target_item_ids[:3]]}")
 
        # -------- search + ndcg for this batch --------
        ndcg_rewards = search_fn(
            prompts=batch_prompts,
            completions=batch_responses,
            target_item_id=batch_target_item_ids,
            sample_idx=batch_sample_idxs,  # aligns candidate pools per row
        )
        retrieval_logs = getattr(search_fn, "last_retrievals", [{}]*len(batch_responses))

        # -------- build per-batch DF --------
        batch_df = pd.DataFrame({
            "sample_idx": batch_sample_idxs,
            "question": batch_questions,
            "answer": batch_answers,
            "prompt": [json.dumps(p) for p in batch_prompts],  # store messages JSON
            "baseline": baseline,
            "checkpoint_dir": best_ckpt,
            "temperature": temperature,
            "response": batch_responses,
            # "ndcg": ndcg_rewards,
            # optional compact retrievals:
            "expanded_query": [log.get("expanded_query") for log in retrieval_logs],
            "retrieved_items": [json.dumps((log.get("retrieved_items") or [])[:50]) for log in retrieval_logs],
            "target_items": [json.dumps(t) for t in batch_target_item_ids]
        })

        batch_df["ndcg"]   = ndcg_rewards
        batch_df["ap"]     = [log.get("ap") for log in retrieval_logs]
        batch_df["mrr"]    = [log.get("mrr") for log in retrieval_logs]
        batch_df["recall"] = [log.get("recall") for log in retrieval_logs]
        batch_df["pool_size"]    = [log.get("pool_size") for log in retrieval_logs]
        batch_df["post_k_unique"]= [len(set(log.get("retrieved_items") or [])) for log in retrieval_logs]
        batch_df["k_eval"]       = batch_df["post_k_unique"]

    
        batch_df["ndcg@pool"]   = [log.get("ndcg@pool") for log in retrieval_logs]
        batch_df["recall@pool"] = [log.get("recall@pool") for log in retrieval_logs]
        batch_df["mrr@pool"]    = [log.get("mrr@pool") for log in retrieval_logs]
        batch_df["ap@pool"]     = [log.get("ap@pool") for log in retrieval_logs]
       
        # --- compute fixed-K columns ---
        targets_list  = [log.get("target") for log in retrieval_logs]
        retr_list     = [log.get("retrieved_items") for log in retrieval_logs]
        for K in FIXED_KS:
            batch_df[f"ndcg@{K}"]  = [_ndcg_at_k(r, t, K)  for r, t in zip(retr_list, targets_list)]
            batch_df[f"recall@{K}"]= [_recall_at_k(r, t, K) for r, t in zip(retr_list, targets_list)]
            batch_df[f"mrr@{K}"]   = [_mrr_at_k(r, t, K)   for r, t in zip(retr_list, targets_list)]
            batch_df[f"ap@{K}"]    = [_ap_at_k(r, t, K)    for r, t in zip(retr_list, targets_list)]
        batch_dfs.append(batch_df)
        # -------- quick preview from first batch --------
 
        if i == 0:
            print("Sample prompt–completion pairs (first batch):")
            # use retrieval logs aligned with batch order
            logs = getattr(search_fn, "last_retrievals", [{}]*len(batch_responses))
            for k, (p, r, ndcg, log) in enumerate(zip(batch_prompts[:3], batch_responses[:3], ndcg_rewards[:3], logs[:3]), start=1):
                fp = "".join(f"[{t['role'].upper()}] {t['content'].strip()}\n" for t in p)
                retrieved_preview = (log.get("retrieved_items") or [])[:10]  # first 10 items
                print(f"\n=== Sample {k} ===")
                print(" Prompt:\n", fp.strip())
                print("\n Completion:\n", (r or ""))
                print(f"\n NDCG: {ndcg:.4f}")
                print(" Retrieved (first 10):", ", ".join(map(str, retrieved_preview)))
                print("-" * 80)


    return all_responses, batch_dfs

 

def parse_args():
    """Parse command line arguments with sensible defaults based on project structure."""
    paths = get_project_paths()
    data_dir = paths["data_dir"]
    output_dir = paths["output_dir"]
    
    ap = argparse.ArgumentParser(description="Generate test trajectories for ESCI search task")
    
    # Data paths
    ap.add_argument(
        "--dataset_path", 
        type=str, 
        default=str(data_dir / "rl_dataset"),
        help="Path to the dataset directory"
    )
    ap.add_argument(
        "--faiss_index_path", 
        type=str, 
        default=str(data_dir / "index" / "all-mpnet-base-v2_faiss.bin"),
        help="Path to FAISS index file"
    )
    ap.add_argument(
        "--mapping_path", 
        type=str, 
        default=str(data_dir / "index" / "all-mpnet-base-v2_asin_mapping.json"),
        help="Path to ASIN mapping JSON"
    )
    ap.add_argument(
        "--metadata_path", 
        type=str, 
        default=str(data_dir / "metadata" / "item_catalog.jsonl"),
        help="Path to item catalog metadata"
    )
    
    # Output paths
    ap.add_argument(
        "--output_dir", 
        type=str, 
        default=str(output_dir),
        help="Directory to write generated CSVs and aggregates"
    )
    
    # Model settings
    ap.add_argument(
        "--expert_model", 
        type=str, 
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model to use for generation"
    )
    ap.add_argument(
        "--sent_emb_model", 
        type=str, 
        default="sentence-transformers/all-mpnet-base-v2",
        help="Sentence embedding model for FAISS"
    )
    
    # Generation parameters
    ap.add_argument(
        "--num_samples", 
        type=int, 
        default=5000,
        help="Number of samples to generate"
    )
    ap.add_argument(
        "--bs", 
        type=int, 
        default=512,
        help="Generation batch size"
    )
    ap.add_argument(
        "--temperature", 
        type=float, 
        default=0.2,
        help="Sampling temperature"
    )
    ap.add_argument(
        "--top_k", 
        type=int, 
        default=1000,
        help="Top k for FAISS retrieval"
    )
    
    # Flags
    ap.add_argument(
        "--include_pools", 
        action="store_true",
        help="Use candidate pools from dataset if available"
    )
    ap.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    
    return ap.parse_args()

def load_dataset_with_sampling(dataset_path: str, num_samples: int = 5000):
    """Load dataset and sample specified number of examples."""
    dataset = load_from_disk(dataset_path)
    total_samples = len(dataset["train"])
    num_samples = min(num_samples, total_samples)
    indices = random.sample(range(total_samples), num_samples)
    ds = dataset["train"].select(indices)
    
    print(f"Loaded {len(ds)} samples from {dataset_path}")
    
    # Analyze target distribution
    def as_list(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            try:
                v = ast.literal_eval(x)
                return v if isinstance(v, list) else [x]
            except Exception:
                return [x]
        return [x]
    
    targets = ds["answer"] if "answer" in ds.column_names else []
    if targets:
        lens = [len(as_list(t)) for t in targets]
        print("\nTarget distribution:")
        print(pd.Series(lens).describe())
        print("Counts by |target|:", pd.Series(lens).value_counts().sort_index().to_dict())
    
    return ds


def extract_dataset_columns(ds):
    """Extract standard columns from dataset with fallbacks."""
    n = len(ds)
    
    return {
        "sample_idxs": ds["sample_idx"] if "sample_idx" in ds.column_names else list(range(n)),
        "questions": ds["question"] if "question" in ds.column_names else [None] * n,
        "answers": ds["answer"] if "answer" in ds.column_names else [None] * n,
        "prompts": ds["prompt"] if "prompt" in ds.column_names else [None] * n,
        "target_item_ids": ds["answer"] if "answer" in ds.column_names else [[] for _ in range(n)],
        "candidate_pools": {int(r["sample_idx"]): r["candidate_pool"] for r in ds} if "candidate_pool" in ds.column_names else None,
    }

# ================
# Main
# ================
def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load dataset
    ds = load_dataset_with_sampling(args.dataset_path, args.num_samples)
    data_cols = extract_dataset_columns(ds)
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.expert_model, 
        trust_remote_code=True, 
        padding_side="left", 
        truncation_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Build search function
    candidate_pools = data_cols["candidate_pools"] if args.include_pools else None
    search_fn = SearchRewardFunction(
        faiss_index_path=args.faiss_index_path,
        mapping_path=args.mapping_path,
        model_name=args.sent_emb_model,
        device=args.device,
        top_k=args.top_k,
        candidate_pools=candidate_pools,
        debug=False
    )
    
    # Process each baseline
    manifest = {}
    all_csv_paths = []
    
    for baseline, model_id in HF_BASELINES.items():
        print(f"\n{'='*60}")
        print(f"Processing baseline: {baseline}")
        print(f"{'='*60}")
        
        ckpt_tag = "base"
        
        # Load model
        model, tok = load_hf_model(model_id)
        
        # Generate responses and scores
        responses, batch_dfs = generate_responses_and_scores(
            model=model,
            tokenizer=tok,
            prompt_messages_list=data_cols["prompts"],
            search_fn=search_fn,
            sample_idxs=data_cols["sample_idxs"],
            questions=data_cols["questions"],
            answers=data_cols["answers"],
            target_item_ids=data_cols["target_item_ids"],
            baseline=baseline,
            best_ckpt=ckpt_tag,
            temperature=args.temperature,
            max_new_tokens=MAX_NEW_TOKENS,
            batch_size=args.bs,
        )
        
        # Save per-baseline CSV
        df = pd.concat(batch_dfs, ignore_index=True)
        csv_name = f"test_generations__{baseline}__{ckpt_tag}__temp{args.temperature}.csv"
        csv_path = output_dir / csv_name
        df.to_csv(csv_path, index=False)
        all_csv_paths.append(str(csv_path))
        
        print(f"\nSaved results to: {csv_path}")
        print(f"Number of samples: {len(df)}")
        print(f"Mean NDCG: {df['ndcg'].mean():.4f}")
        
        # Update manifest
        manifest.setdefault(baseline, {})
        manifest[baseline][ckpt_tag] = {
            "checkpoint_dir": ckpt_tag,
            "temperature": args.temperature,
            "csv_path": str(csv_path),
            "num_samples": int(len(df)),
            "model_id": model_id,
            "mean_ndcg": float(df["ndcg"].mean()),
        }
        
        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save manifest
    manifest_path = output_dir / f"manifest_{RUN_ID}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest to: {manifest_path}")
    
    print(f"\n{'='*60}")
    print("Generation complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Generated {len(all_csv_paths)} CSV files")


if __name__ == "__main__":
    main()