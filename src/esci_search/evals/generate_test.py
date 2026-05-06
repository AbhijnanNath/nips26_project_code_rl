#!/usr/bin/env python3

# Code released for anonymous review. License: CC-BY-NC-4.0  

import os
import re
import json
import math
import time
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--checkpoint_base", type=str, default="/home/greenland-user/esci_search/src/Qwen")
    parser.add_argument("--dataset_path", type=str, default="/home/greenland-user/home/greenland-user/esci_search/esci_search_candidate_pools")
    parser.add_argument("--output_dir", type=str, default="/home/greenland-user/esci_search/test_generations")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--s3_bucket", type=str, default="")
    parser.add_argument("--checkpoint_priority", type=int, nargs="+", default=[2000, 1500, 1000])
    return parser.parse_args()


def get_baselines(checkpoint_base):
    return {
        
        "OSPO_rank": os.path.join(checkpoint_base, "Qwen2.5-7B-Instruct-OSPO-esci-search-rank_based_2"),
        "OSPO_prop": os.path.join(checkpoint_base, "Qwen2.5-7B-Instruct-OSPO-esci-search-owen_weights_2"),
        "GRPO": os.path.join(checkpoint_base, "Qwen2.5-7B-Instruct-GRPO-search-esci"),
    }

def find_best_checkpoint_dir(baseline_dir, checkpoint_priority):
    if not os.path.isdir(baseline_dir):
        return None
    ckpts = []
    for e in os.listdir(baseline_dir):
        m = re.match(r"checkpoint-(\d+)$", e)
        if m:
            p = os.path.join(baseline_dir, e)
            if os.path.isdir(p):
                ckpts.append(int(m.group(1)))
    if not ckpts:
        return None
    for p in checkpoint_priority:
        if p in ckpts:
            return os.path.join(baseline_dir, f"checkpoint-{p}")
    return os.path.join(baseline_dir, f"checkpoint-{max(ckpts)}")


def load_lora_model(base_model_name, lora_checkpoint_dir):
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    model = PeftModel.from_pretrained(base_model, lora_checkpoint_dir, local_files_only=True)
    model.eval()
    return model


def generate_responses_batch(model, tokenizer, prompt_messages_list, temperature, max_new_tokens, batch_size):
    all_responses = []
    total_batches = math.ceil(len(prompt_messages_list) / batch_size)
    print(f"\nGenerating responses for {len(prompt_messages_list)} prompts in {total_batches} batches (batch size = {batch_size})...")

    for i in tqdm(range(0, len(prompt_messages_list), batch_size), total=total_batches, desc="Generating", ncols=100):
        batch_prompts = prompt_messages_list[i:i + batch_size]
        formatted = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in batch_prompts]
        inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True).to(model.device)

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

        in_lens = inputs.attention_mask.sum(dim=1).tolist()
        for j, out in enumerate(outputs):
            gen_ids = out[int(in_lens[j]):]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            all_responses.append(text)

        if i == 0:
            print("\nSample prompt-completion pairs (first batch):")
            for k, (p, r) in enumerate(zip(batch_prompts[:3], all_responses[:3])):
                fp = "".join(f"[{t['role'].upper()}] {t['content'].strip()}\n" for t in p)
                print(f"\n=== Sample {k+1} ===")
                print("Prompt:\n", fp.strip())
                print("\nCompletion:\n", r[:400])
                print("-"*80)

    return all_responses


def main():
    args = parse_args()
    
    dataset = load_from_disk(args.dataset_path)
    test_ds = dataset["test"]
    print(f"Test size: {len(test_ds)}")
    print("Test fields:", test_ds.column_names)

    n = len(test_ds)
    sample_idxs = test_ds["sample_idx"] if "sample_idx" in test_ds.column_names else list(range(n))
    questions = test_ds["question"] if "question" in test_ds.column_names else [None] * n
    answers = test_ds["answer"] if "answer" in test_ds.column_names else [None] * n
    prompts = test_ds["prompt"] if "prompt" in test_ds.column_names else [None] * n

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, padding_side="left", truncation_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    run_id = time.strftime("%Y%m%d_%H%M%S")
    s3_prefix = f"test_generations/esci/{run_id}"
    baselines = get_baselines(args.checkpoint_base)
    manifest = {}
    all_csv_paths = []

    for baseline, baseline_dir in baselines.items():
        print(f"\n==== Baseline: {baseline} ====")
        best_ckpt = find_best_checkpoint_dir(baseline_dir, args.checkpoint_priority)
        if best_ckpt is None:
            print(f"No checkpoint found under {baseline_dir}, skipping.")
            continue

        print(f"Selected checkpoint: {best_ckpt}")
        model = load_lora_model(args.base_model, best_ckpt)
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

        responses = generate_responses_batch(model, tokenizer, prompts, args.temperature, args.max_new_tokens, args.batch_size)

        df = pd.DataFrame({
            "sample_idx": sample_idxs,
            "question": questions,
            "answer": answers,
            "prompt": [json.dumps(p) for p in prompts],
            "baseline": baseline,
            "checkpoint_dir": best_ckpt,
            "temperature": args.temperature,
            "response": responses,
        })

        ckpt_tag = os.path.basename(best_ckpt)
        csv_name = f"test_generations__{baseline}__{ckpt_tag}__temp{args.temperature}.csv"
        csv_path = os.path.join(args.output_dir, csv_name)
        df.to_csv(csv_path, index=False)
        all_csv_paths.append(csv_path)

        manifest[baseline] = {
            "selected_checkpoint": best_ckpt,
            "temperature": args.temperature,
            "csv_path": csv_path,
            "num_samples": len(df),
        }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest_path = os.path.join(args.output_dir, "manifest_baseline_checkpoints_and_outputs.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "base_model": args.base_model,
            "dataset_path": args.dataset_path,
            "output_dir": args.output_dir,
            "temperature": args.temperature,
            "run_id": run_id,
            "s3_prefix": s3_prefix,
            "baselines": manifest,
            "all_csv_paths": all_csv_paths
        }, f, indent=2)

    print("Manifest:", manifest_path)
    print("CSV files:")
    for p in all_csv_paths:
        print(" -", p)


if __name__ == "__main__":
    main()