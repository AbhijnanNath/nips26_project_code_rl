#!/usr/bin/env python3

# Code released for anonymous review. License: CC-BY-NC-4.0  

import os, re, json, math, time, glob, ast
import pandas as pd
from tqdm import tqdm
from typing import List, Iterable, Dict, Any
import torch
from collections import OrderedDict
import argparse
from ..trainers.dense_search.search import SearchRewardFunction
from datasets import load_from_disk


# Helpers
# =================

def as_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)  # handles "['A','B']"
            return v if isinstance(v, list) else [x]
        except Exception:
            return [x]
    return [x]

def pick_text_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No text column found. Tried {candidates}. Present: {list(df.columns)}")


def _dedup(lst):
    return list(OrderedDict.fromkeys(lst or []))

def _trunc(lst, k):
    return (lst or [])[:k]

def _ndcg_at_k(search_fn, items, targets, k):
    return search_fn._calculate_ndcg(_trunc(_dedup(items), k), targets, k=k)

def _recall_at_k(search_fn, items, targets, k):
    return search_fn._calculate_recall_at_k(_trunc(_dedup(items), k), targets, k=k)

def _mrr_at_k(search_fn, items, targets, k):
    return search_fn._mrr(_trunc(_dedup(items), k), targets, k=k)

def _ap_at_k(search_fn, items, targets, k):
    return search_fn._average_precision(_trunc(_dedup(items), k), targets, k=k)

def load_all_csvs(pattern: str) -> List[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CSVs matched: {pattern}")
    return paths

def build_pools(dataset) -> Dict[int, List[str]]:
    return {int(r["sample_idx"]): r["candidate_pool"] for r in dataset["test"]}

def map_by_sample_idx(series):
    """Returns dict[int -> value] for fast alignment from dataset by sample_idx"""
    return {int(i): v for i, v in zip(series["sample_idx"], series)}

# =================
# Main
# =================

def main():
    cfg = parse_cli()
    DATASET_PATH        = cfg["DATASET_PATH"]
    OUTPUT_DIR          = cfg["OUTPUT_DIR"]
    INPUT_CSV_GLOB      = cfg["INPUT_CSV_GLOB"]
    TEXT_COL_CANDIDATES = cfg["TEXT_COL_CANDIDATES"]
    FIXED_KS            = cfg["FIXED_KS"]
    SEARCH_CFG          = cfg["SEARCH_CFG"]
    BATCH_SIZE          = cfg["BATCH_SIZE"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    # Load dataset & pools
    dataset = load_from_disk(DATASET_PATH)
    test_ds = dataset["test"]
    print(f"Test size: {len(test_ds)} | Fields: {test_ds.column_names}")

    # Build candidate pools
    candidate_pools = build_pools(dataset)
    num_keys = len(candidate_pools)
    lens = [len(v) for v in candidate_pools.values()]
    print(f"[POOLS] samples_with_pools={num_keys} | mean_size={sum(lens)/max(1,num_keys):.2f} | "
          f"min={min(lens) if lens else 0} | max={max(lens) if lens else 0}")

    # Prepare dataset lookup by sample_idx
    ds_df = pd.DataFrame({
        "sample_idx": test_ds["sample_idx"] if "sample_idx" in test_ds.column_names else list(range(len(test_ds))),
        "question":   test_ds["question"]    if "question"    in test_ds.column_names else [None]*len(test_ds),
        "answer":     test_ds["answer"]      if "answer"      in test_ds.column_names else [None]*len(test_ds),
        "prompt":     test_ds["prompt"]      if "prompt"      in test_ds.column_names else [None]*len(test_ds),
    })
    ds_df["target_items"] = ds_df["answer"]  # alias
    ds_by_idx = ds_df.set_index("sample_idx").to_dict(orient="index")

    # Build search function
    search_fn = SearchRewardFunction(candidate_pools=candidate_pools, debug=False, **SEARCH_CFG)

    # Load all CSVs to rescore
    csv_paths = load_all_csvs(INPUT_CSV_GLOB)
    print(f"[INPUT] Found {len(csv_paths)} CSV files to rescore.")

    manifest = {}
    out_paths = []

    for csv_path in csv_paths:
        print(f"\n=== Rescoring: {csv_path} ===")
        df_in = pd.read_csv(csv_path)
        print("df_in cols", df_in.columns)
        # sanity: must have sample_idx and a text column
        if "sample_idx" not in df_in.columns:
            raise ValueError(f"{csv_path} missing 'sample_idx' column.")
        text_col = pick_text_column(df_in, TEXT_COL_CANDIDATES)

        # Align dataset fields to rows (by sample_idx)
        aligned_prompt = []
        aligned_question = []
        aligned_answer = []
        aligned_targets = []
        for idx in df_in["sample_idx"].tolist():
            row = ds_by_idx.get(int(idx), {"prompt": None, "question": None, "answer": None, "target_items": []})
            aligned_prompt.append(row["prompt"])
            aligned_question.append(row["question"])
            aligned_answer.append(row["answer"])
            aligned_targets.append(row["target_items"])

        # In chunks, call search_fn using the CSV text column as "completions"
        total = len(df_in)
        total_batches = math.ceil(total / BATCH_SIZE)
        ndcg_scores = []
        logs_list   = []

        for i in tqdm(range(0, total, BATCH_SIZE), total=total_batches, ncols=100, desc="Rescoring"):
            j = i + BATCH_SIZE
            batch_prompts = aligned_prompt[i:j]     # not used by metrics, but keeps interface
            batch_comps   = df_in[text_col].iloc[i:j].fillna("").astype(str).tolist()
            batch_targets = aligned_targets[i:j]
            batch_idxs    = df_in["sample_idx"].iloc[i:j].tolist()

            # search + ndcg
            ndcgs = search_fn(
                prompts=batch_prompts,
                completions=batch_comps,
                target_item_id=batch_targets,
                sample_idx=batch_idxs,
            )
            ndcg_scores.extend(ndcgs)
            logs = getattr(search_fn, "last_retrievals", [{}]*len(batch_comps))
            logs_list.extend(logs)

        # Build output DF (preserve original columns + new metrics)
        df_out = df_in.copy()
        df_out["ndcg"] = ndcg_scores
        df_out["question"] = aligned_question
        df_out["answer"]   = aligned_answer
        df_out["prompt"]   = [json.dumps(p) if isinstance(p, list) else p for p in aligned_prompt]

        # unpack retrieval logs
        df_out["expanded_query_rescored"] = [log.get("expanded_query") for log in logs_list]
        df_out["retrieved_items"]         = [json.dumps((log.get("retrieved_items") or [])[:50]) for log in logs_list]
        df_out["pool_size"]               = [log.get("pool_size") for log in logs_list]
        df_out["post_k_unique"]           = [len(set(log.get("retrieved_items") or [])) for log in logs_list]
        df_out["k_eval"]                  = df_out["post_k_unique"]

        df_out["ndcg@pool"]   = [log.get("ndcg@pool") for log in logs_list]
        df_out["recall@pool"] = [log.get("recall@pool") for log in logs_list]
        df_out["mrr@pool"]    = [log.get("mrr@pool") for log in logs_list]
        df_out["ap@pool"]     = [log.get("ap@pool") for log in logs_list]

        # fixed-K metrics recomputed on retrieved_items vs targets
        targets_list = [log.get("target") for log in logs_list]
        retr_list    = [log.get("retrieved_items") for log in logs_list]
        for K in FIXED_KS:
            df_out[f"ndcg@{K}"]   = [_ndcg_at_k(search_fn, r, t, K)  for r, t in zip(retr_list, targets_list)]
            df_out[f"recall@{K}"] = [_recall_at_k(search_fn, r, t, K) for r, t in zip(retr_list, targets_list)]
            df_out[f"mrr@{K}"]    = [_mrr_at_k(search_fn, r, t, K)   for r, t in zip(retr_list, targets_list)]
            df_out[f"ap@{K}"]     = [_ap_at_k(search_fn, r, t, K)    for r, t in zip(retr_list, targets_list)]

        # Save side-by-side “rescored” CSV
        base = os.path.basename(csv_path)
        out_name = re.sub(r"\.csv$", "", base) + "__rescored.csv"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        df_out.to_csv(out_path, index=False)
        out_paths.append(out_path)

        # Manifest entry
        manifest[base] = {
            "input_csv": csv_path,
            "output_csv": out_path,
            "num_rows": len(df_out),
            "text_col_used": text_col,
            "fixed_Ks": FIXED_KS,
        }
        print(f"[DONE] -> {out_path}")

    # Save manifest
    manifest_path = os.path.join(OUTPUT_DIR, "manifest_search_only_rescoring.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "dataset_path": DATASET_PATH,
            "output_dir": OUTPUT_DIR,
            "faiss_cfg": {k: v for k, v in SEARCH_CFG.items() if k != "device"},
            "batch_size": BATCH_SIZE,
            "fixed_Ks": FIXED_KS,
            "outputs": out_paths,
            "inputs": csv_paths,
        }, f, indent=2)

    print("\nRescoring complete.")
    print("Manifest:", manifest_path)
    for p in out_paths:
        print(" -", p)

def parse_cli() -> dict:
    parser = argparse.ArgumentParser(description="Search-only rescoring config")
    parser.add_argument("--dataset_path",   default="/home/greenland-user/hnm_context_search/src/esci_search_candidate_pools")
    parser.add_argument("--output_dir",     default="/home/greenland-user/esci_search/test_generations_with_search_RESCORING")
    parser.add_argument("--input_csv_glob", default="/home/greenland-user/esci_search/test_generations_with_search/*.csv")
    parser.add_argument("--text_cols",      default="expanded_query,response")
    parser.add_argument("--fixed_k",        default="10,50")
    parser.add_argument("--faiss_index_path", default="/home/greenland-user/esci_search/esci_faiss_index_asin_metadata/faiss_index/simcse-large/faiss_hnsw_index.bin")
    parser.add_argument("--mapping_path",     default="/home/greenland-user/esci_search/esci_faiss_index_asin_metadata/faiss_index/simcse-large/faiss_hnsw_index_asin_mapping.json")
    parser.add_argument("--model_name",     default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top_k",          type=int, default=1000)
    parser.add_argument("--batch_size",     type=int, default=256)
    args = parser.parse_args()

    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    fixed_k   = [int(x) for x in args.fixed_k.split(",") if x.strip()]
    return {
        "DATASET_PATH": args.dataset_path,
        "OUTPUT_DIR": args.output_dir,
        "INPUT_CSV_GLOB": args.input_csv_glob,
        "TEXT_COL_CANDIDATES": text_cols,
        "FIXED_KS": fixed_k,
        "SEARCH_CFG": {
            "faiss_index_path": args.faiss_index_path,
            "mapping_path": args.mapping_path,
            "model_name": args.model_name,
            "device": args.device,
            "top_k": args.top_k,
        },
        "BATCH_SIZE": args.batch_size,
    }


 
if __name__ == "__main__":
    main()
