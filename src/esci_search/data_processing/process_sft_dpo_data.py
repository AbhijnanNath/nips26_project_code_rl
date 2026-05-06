# Code released for anonymous review. License: CC-BY-NC-4.0  

 
import pandas as pd
import json
import ast
from pathlib import Path
from typing import List, Dict
import argparse
import glob

def create_sft_dataset_simple(
    data1_path: str,
    data2_path: str,
    output_path: str,
    ndcg_threshold: float = 0.3
):
    """
    Simplified version: Take ALL samples above threshold from both files.
    Includes both winners and individual high-quality samples.
    """
    
    print("="*80)
    print("Creating SFT Dataset (Simple Mode - All High-Quality Samples)")
    print("="*80)
    
    # Load data
    data1 = pd.read_csv(data1_path)
    data2 = pd.read_csv(data2_path)
    
    # Combine both datasets
    all_data = pd.concat([
        data1[['sample_idx', 'question', 'prompt', 'response', 'ndcg']],
        data2[['sample_idx', 'question', 'prompt', 'response', 'ndcg']]
    ], ignore_index=True)
    
    print(f"Total samples: {len(all_data)}")
    
    # Filter by NDCG threshold
    high_quality = all_data[all_data['ndcg'] > ndcg_threshold].copy()
    print(f"High-quality samples (NDCG > {ndcg_threshold}): {len(high_quality)}")
    # Create SFT samples
    sft_samples = []
    
    for idx, row in high_quality.iterrows():
        # Parse system prompt
        try:
            prompt_messages = json.loads(row['prompt'])
        except json.JSONDecodeError:
            try:
                prompt_messages = ast.literal_eval(row['prompt'])
            except:
                continue
        
        system_content = prompt_messages[0]['content'] if len(prompt_messages) > 0 else ""

         # Extract just the assistant's actual response
        response_text = row['response']

        # Find where "assistant" appears and take everything after it
        if '\nassistant\n' in response_text:
            # Split on '\nassistant\n' and take the last part
            clean_response = response_text.split('\nassistant\n')[-1]
        elif 'assistant\n' in response_text:
            clean_response = response_text.split('assistant\n')[-1]
        else:
            # If no "assistant" marker, use the full response
            clean_response = response_text

        # Strip any leading/trailing whitespace
        clean_response = clean_response.strip()
        
        sft_sample = {
            "messages": [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": row['question']
                },
                {
                    "role": "assistant",
                    "content": clean_response
                }
            ],
            "metadata": {
                "sample_idx": int(row['sample_idx']),
                "ndcg": float(row['ndcg'])
            }
        }
        
        sft_samples.append(sft_sample)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for sample in sft_samples:
            f.write(json.dumps(sample) + '\n')
    
    return sft_samples

def clean_reponse(response_text):
            # Find where "assistant" appears and take everything after it
            if '\nassistant\n' in response_text:
                # Split on '\nassistant\n' and take the last part
                clean_response = response_text.split('\nassistant\n')[-1]
            elif 'assistant\n' in response_text:
                clean_response = response_text.split('assistant\n')[-1]
            else:
                # If no "assistant" marker, use the full response
                clean_response = response_text

            # Strip any leading/trailing whitespace
            clean_response = clean_response.strip()
            return clean_response

def create_dpo_dataset(
    data1_path: str,
    data2_path: str,
    output_path: str,
    metric: str = "ap",                  # e.g., "ap", "ndcg", "mrr"
    metric_threshold: float = 0.1,       # inclusion threshold for the metric
    min_metric_diff: float = 0.05,       # minimum absolute difference
    require_both_above_threshold: bool = False,
    format: str = "jsonl"
):
    """
    Create a DPO dataset with chosen/rejected pairs based on a generic ranking metric.

    Args:
        data1_path: Path to first generation CSV
        data2_path: Path to second generation CSV
        output_path: Path to save DPO dataset
        metric: Column name of the ranking metric (e.g., "ap", "ndcg", "mrr")
        metric_threshold: Minimum metric value for samples to be considered
        min_metric_diff: Minimum absolute difference between chosen and rejected
        require_both_above_threshold: If True, both samples must exceed threshold
                                      If False, at least one must exceed threshold
        format: Output format ("jsonl" or "json")
    """

    print("="*80)
    print(f"Creating DPO Dataset (metric='{metric}')")
    print("="*80)

    # Load data
    data1 = pd.read_csv(data1_path)
    data2 = pd.read_csv(data2_path)

    # Basic sanity on sample_idx
    common = set(data1.get('sample_idx', [])) & set(data2.get('sample_idx', []))
    if len(common) == 0:
        print("No common sample_idx across the two inputs; aborting.")
        return []

    # Merge pairs
    merged = pd.merge(
        data1,
        data2,
        on='sample_idx',
        suffixes=('_1', '_2'),
        how='inner'
    )
    print(f"After merge: {len(merged)} pairs")

    # Resolve metric columns (expect metric_1 and metric_2 after merge)
    metric_1 = f"{metric}_1"
    metric_2 = f"{metric}_2"
    if metric_1 not in merged.columns or metric_2 not in merged.columns:
        raise KeyError(
            f"Expected columns '{metric_1}' and '{metric_2}' not found. "
            f"Available columns include: {sorted(merged.columns)}"
        )

    # Threshold counts
    above_1 = (merged[metric_1] > metric_threshold).sum()
    above_2 = (merged[metric_2] > metric_threshold).sum()
    both_above = ((merged[metric_1] > metric_threshold) & (merged[metric_2] > metric_threshold)).sum()
    print(f"\n{metric.upper()} > {metric_threshold}:")
    print(f"  Run1: {above_1}/{len(merged)}")
    print(f"  Run2: {above_2}/{len(merged)}")
    print(f"  Both: {both_above}/{len(merged)}")

    # Apply threshold filter
    if require_both_above_threshold:
        valid = merged[(merged[metric_1] > metric_threshold) & (merged[metric_2] > metric_threshold)].copy()
        print(f"Pairs with both {metric.upper()} > {metric_threshold}: {len(valid)}")
    else:
        valid = merged[(merged[metric_1] > metric_threshold) | (merged[metric_2] > metric_threshold)].copy()
        print(f"Pairs with at least one {metric.upper()} > {metric_threshold}: {len(valid)}")

    # Enforce minimum metric difference
    valid['metric_diff'] = (valid[metric_1] - valid[metric_2]).abs()
    valid = valid[valid['metric_diff'] >= min_metric_diff]
    print(f"After filtering for |Δ{metric}| ≥ {min_metric_diff}: {len(valid)}")

    # Build DPO pairs
    dpo_samples = []
    skipped = 0

    for _, row in valid.iterrows():
        # Parse system + user (from prompt_1 / question_1)
        try:
            prompt_messages = None
            try:
                prompt_messages = json.loads(row['prompt_1'])
            except json.JSONDecodeError:
                prompt_messages = ast.literal_eval(row['prompt_1'])
            system_message = prompt_messages[0]['content'] if (prompt_messages and len(prompt_messages) > 0) else ""
        except Exception:
            print(f"⚠️ Could not parse prompt for sample_idx={row['sample_idx']}, skipping")
            skipped += 1
            continue

        user_message = row.get('question_1', "")

        # Clean responses
        r1 = clean_reponse(row.get('response_1', ""))
        r2 = clean_reponse(row.get('response_2', ""))

        # Choose by higher metric
        if row[metric_1] > row[metric_2]:
            chosen_response, rejected_response = r1, r2
            chosen_metric, rejected_metric = float(row[metric_1]), float(row[metric_2])
            chosen_baseline, rejected_baseline = row.get('baseline_1', None), row.get('baseline_2', None)
            chosen_expanded_query, rejected_expanded_query = row.get('expanded_query_1', None), row.get('expanded_query_2', None)
        else:
            chosen_response, rejected_response = r2, r1
            chosen_metric, rejected_metric = float(row[metric_2]), float(row[metric_1])
            chosen_baseline, rejected_baseline = row.get('baseline_2', None), row.get('baseline_1', None)
            chosen_expanded_query, rejected_expanded_query = row.get('expanded_query_2', None), row.get('expanded_query_1', None)

        dpo_sample = {
            'chosen': [
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': user_message},
                {'role': 'assistant', 'content': chosen_response}
            ],
            'rejected': [
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': user_message},
                {'role': 'assistant', 'content': rejected_response}
            ],
            # Generic metadata (metric-agnostic)
            'sample_idx': int(row['sample_idx']),
            'metric_name': metric,
            'chosen_metric': chosen_metric,
            'rejected_metric': rejected_metric,
            'metric_diff': chosen_metric - rejected_metric,
            'chosen_baseline': chosen_baseline,
            'rejected_baseline': rejected_baseline,
            'chosen_expanded_query': chosen_expanded_query,
            'rejected_expanded_query': rejected_expanded_query,
            'target_items': row.get('target_items_1', None),
        }

        dpo_samples.append(dpo_sample)

    print(f"\nCreated {len(dpo_samples)} DPO samples (skipped {skipped})")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with open(output_path, 'w') as f:
            for sample in dpo_samples:
                f.write(json.dumps(sample) + '\n')
        print(f"Saved to {output_path} (JSONL)")
    elif format == "json":
        with open(output_path, 'w') as f:
            json.dump(dpo_samples, f, indent=2)
        print(f"✅ Saved to {output_path} (JSON)")

    # Simple stats
    print("\n" + "="*80)
    print("DPO Dataset Statistics")
    print("="*80)
    if dpo_samples:
        diffs = [s['metric_diff'] for s in dpo_samples]
        chosen_vals = [s['chosen_metric'] for s in dpo_samples]
        rejected_vals = [s['rejected_metric'] for s in dpo_samples]

        print(f"Mean |Δ{metric}|: {sum(abs(x) for x in diffs)/len(diffs):.4f}")
        print(f"Median |Δ{metric}|: {sorted(abs(x) for x in diffs)[len(diffs)//2]:.4f}")
        print(f"Max |Δ{metric}|: {max(abs(x) for x in diffs):.4f}")

        print(f"\nChosen {metric.upper()}: mean={sum(chosen_vals)/len(chosen_vals):.4f}, "
              f"min={min(chosen_vals):.4f}, max={max(chosen_vals):.4f}")
        print(f"Rejected {metric.upper()}: mean={sum(rejected_vals)/len(rejected_vals):.4f}, "
              f"min={min(rejected_vals):.4f}, max={max(rejected_vals):.4f}")

    return dpo_samples

def pick_two_csvs(input_dir: str):
    files = sorted(glob.glob(str(Path(input_dir) / "*.csv")), key=lambda p: Path(p).stat().st_mtime, reverse=True)
    if len(files) < 2:
        raise RuntimeError(f"Need at least 2 CSVs in {input_dir}, found {len(files)}")
    return files[0], files[1]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True,
                   help="Dir containing two CSV runs to pair (will pick the two newest *.csv).")
    p.add_argument("--metric", type=str, default="ndcg",
                   help="Ranking metric column name (e.g., ndcg, ap, mrr).")
    p.add_argument("--k", type=int, default=1000,
                   help="K used to compute the metric (informational; columns usually already aggregated).")
    p.add_argument("--sft_out", type=str, required=True)
    p.add_argument("--dpo_out", type=str, required=True)
    p.add_argument("--sft_threshold", type=float, default=0.3)
    p.add_argument("--dpo_threshold", type=float, default=0.1)
    p.add_argument("--min_diff", type=float, default=0.05)
    p.add_argument("--require_both", action="store_true",
                   help="Require both sides > dpo_threshold.")
    return p.parse_args()

 
if __name__ == "__main__":

    args = parse_args()
    d1, d2 = pick_two_csvs(args.input_dir)

    # SFT (simple high-quality filter by metric name)
    sft_samples_all = create_sft_dataset_simple(
        data1_path=d1,
        data2_path=d2,
        output_path=args.sft_out,
        ndcg_threshold=args.sft_threshold if args.metric.lower() == "ndcg" else args.dpo_threshold  # use 0.3 for NDCG; else reuse dpo_threshold if you want
    )

    # DPO (We use AP instead)
    dpo_samples = create_dpo_dataset(
        data1_path=d1,
        data2_path=d2,
        output_path=args.dpo_out,
        metric=args.metric.lower(),
        metric_threshold=args.dpo_threshold,
        min_metric_diff=args.min_diff,
        require_both_above_threshold=args.require_both,
        format="jsonl"
    )
    
  