 
"""
DPO Training Script for ESCI Search Task

Prerequisites:
1. Generate DPO dataset first:
   cd src/esci_search/data_processing
   ./run_sft_dpo_pipeline.sh
   This creates: data/esci/dpo_dataset.jsonl

2. (Optional) Train SFT model first for better initialization:
   python src/esci_search/trainers/train_sft.py --model qwen-7b
   This creates checkpoints in: sft_trained_models/qwen-7b/

Usage (run from PROJECT ROOT):
   # Basic DPO (from base model)
   python src/esci_search/trainers/train_dpo.py --model qwen-7b
   
   # DPO from SFT checkpoint (recommended)
   python src/esci_search/trainers/train_dpo.py \
       --model qwen-7b \
       --sft_checkpoint_path sft_trained_models/qwen-7b/checkpoint-150
   
   # With custom dataset
   python src/esci_search/trainers/train_dpo.py \
       --model qwen-7b \
       --dataset data/esci/custom_dpo.jsonl
   
   # With accelerate (multi-GPU)
   accelerate launch src/esci_search/trainers/train_dpo.py --model qwen-7b
 
Outputs:
   - Checkpoints: ./outputs/dpo_models/{model_key}/
   - Logs: wandb (run: wandb login)
"""

import os
import json
import torch
import argparse
import subprocess
from datetime import datetime
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, PeftModel
import gc
from transformers import TrainerCallback
import boto3
import shutil
import random
from botocore.exceptions import ClientError
class LocalCheckpointCallback(TrainerCallback):
    """Saves checkpoints locally and removes old ones to save disk space."""
    def __init__(self, keep_local_checkpoints=2):
        self.keep_local_checkpoints = keep_local_checkpoints

    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        if not os.path.exists(checkpoint_path):
            return
        print(f"Saved checkpoint-{state.global_step} to {checkpoint_path}")
        self._cleanup_old_checkpoints(args.output_dir, state.global_step)

    def _cleanup_old_checkpoints(self, output_dir, current_step):
        try:
            checkpoints = []
            for item in os.listdir(output_dir):
                if item.startswith("checkpoint-"):
                    try:
                        step = int(item.split("-")[1])
                        checkpoints.append(
                            (step, os.path.join(output_dir, item))
                        )
                    except (ValueError, IndexError):
                        continue
            checkpoints.sort(key=lambda x: x[0], reverse=True)
            for step, path in checkpoints[self.keep_local_checkpoints:]:
                if os.path.exists(path):
                    shutil.rmtree(path)
                    print(f"Deleted old checkpoint: checkpoint-{step}")
        except Exception as e:
            print(f"Error during cleanup: {e}")


class ESCISearchTrainerDPOTrainer:
    def __init__(self, dataset_path="data/esci/dpo_dataset.jsonl", sft_checkpoint_path = None):
         
        self.dataset_path = dataset_path
        self.base_output_dir = "./outputs/dpo_models" 
        self.sft_checkpoint_path = sft_checkpoint_path
        # DPO-specific model configurations
        self.model_configs = {
            "qwen-0.5b": {
                "model_name": "Qwen/Qwen2.5-0.5B-Instruct",  # Use instruct models for DPO
                "batch_size": 4,  # Smaller for DPO (needs both chosen/rejected)
                "gradient_accumulation": 2,
                "learning_rate": 5e-6,  # Lower LR for DPO
                "lora_r": 16,
                "lora_alpha": 32,
                "max_length": 1024,
                "beta": 0.1  # DPO-specific parameter
            },
            "qwen-1.5b": {
                "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                "batch_size": 2,
                "gradient_accumulation": 4,
                "learning_rate": 3e-6,
                "lora_r": 32,
                "lora_alpha": 64,
                "max_length": 1024,
                "beta": 0.1
            },
            "qwen-7b": {
                "model_name": "Qwen/Qwen2.5-7B-Instruct",
                "batch_size": 4,
                "gradient_accumulation": 4,
                "learning_rate": 5e-6,
                "lora_r": 32,
                "lora_alpha": 16,
                "max_length": 512,  #first run analyze_dataset_token_lengths to see token-length distribution. Then, fix this during training. 
                "beta": 0.1
            },
            "qwen-7b-ipo": {
                "model_name": "Qwen/Qwen2.5-7B-Instruct",
                "batch_size": 4,
                "gradient_accumulation": 4,
                "learning_rate": 5e-6,
                "lora_r": 32,
                "lora_alpha": 16,
                "max_length": 1400,  #first run analyze_dataset_token_lengths to see token-length distribution. Then, fix this during training. 
                "beta": 0.1,
                "loss_type": "ipo"
            }
        }

 
        self.load_dataset()
      

    def analyze_dataset_token_lengths(self, dataset, dataset_name, tokenizer):
        total_samples = len(dataset)
        
        # Collect all token lengths
        chosen_token_lengths = []
        rejected_token_lengths = []
        
        role_token_stats = {'chosen': {}, 'rejected': {}}
        
        for sample in dataset:
            # Analyze chosen
            chosen_total_tokens = 0
            for msg in sample['chosen']:
                role = msg['role']
                tokens = tokenizer.encode(msg['content'])
                token_len = len(tokens)
                chosen_total_tokens += token_len
                if role not in role_token_stats['chosen']:
                    role_token_stats['chosen'][role] = []
                role_token_stats['chosen'][role].append(token_len)
            chosen_token_lengths.append(chosen_total_tokens)
            
            # Analyze rejected
            rejected_total_tokens = 0
            for msg in sample['rejected']:
                role = msg['role']
                tokens = tokenizer.encode(msg['content'])
                token_len = len(tokens)
                rejected_total_tokens += token_len
                if role not in role_token_stats['rejected']:
                    role_token_stats['rejected'][role] = []
                role_token_stats['rejected'][role].append(token_len)
            rejected_token_lengths.append(rejected_total_tokens)
        
        print(f"\n=== {dataset_name.upper()} DATASET TOKEN LENGTH ANALYSIS ===")
        print(f"Total samples: {total_samples}")
        print(f"\nChosen token lengths - Min: {min(chosen_token_lengths)}, Max: {max(chosen_token_lengths)}, Avg: {sum(chosen_token_lengths)/len(chosen_token_lengths):.1f}")
        print(f"Rejected token lengths - Min: {min(rejected_token_lengths)}, Max: {max(rejected_token_lengths)}, Avg: {sum(rejected_token_lengths)/len(rejected_token_lengths):.1f}")
        
        # Role-wise token stats
        for conv_type in ['chosen', 'rejected']:
            print(f"\n{conv_type.capitalize()} role token breakdown:")
            for role, token_lengths in role_token_stats[conv_type].items():
                avg_tokens = sum(token_lengths) / len(token_lengths)
                print(f"  {role}: {len(token_lengths)} msgs, avg {avg_tokens:.1f} tokens")

     
    def load_dataset(self):
        """Load the dataset from JSONL and create train/eval split"""
        from datasets import Dataset  # Add this import at top
        
        print(f"Loading dataset from {self.dataset_path}")
        
        # Load JSONL
        samples = []
        with open(self.dataset_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                samples.append(sample)
        
        # Shuffle and split (90% train, 10% eval)
        random.seed(42)
        random.shuffle(samples)
        
        eval_size = int(len(samples) * 0.1)
        test_samples = samples[:eval_size]
        train_samples = samples[eval_size:]
        
        # Convert to HuggingFace Dataset format
        self.train_dataset = Dataset.from_list(train_samples)
        self.test_dataset = Dataset.from_list(test_samples)
        
        print(f"Train dataset: {len(self.train_dataset)} samples")
        print(f"Test dataset: {len(self.test_dataset)} samples")
        
        # Print one sample from each split
        print(f"\nTrain sample: {self.train_dataset[0]}")
        print(f"\nTest sample: {self.test_dataset[0]}")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        self.analyze_dataset_token_lengths(self.train_dataset, "train", tokenizer)
        self.analyze_dataset_token_lengths(self.test_dataset, "test", tokenizer)


    def get_model_size_gb(self, model_name):
        size_mapping = {
            "0.5B": 1,
            "1.5B": 3,
            "7B": 14,
            "14B": 28,
            "32B": 64
        }
        for size, gb in size_mapping.items():
            if size in model_name:
                return gb
        return 8
 

    def setup_model_and_tokenizer(self, config, checkpoint_path=None):
        model_name = config["model_name"]
        print(f"\nLoading model: {model_name}")
        
        model_size_gb = self.get_model_size_gb(model_name)
        torch_dtype = torch.bfloat16 if model_size_gb <= 14 else torch.float16
        if not os.path.exists(checkpoint_path):
            raise ValueError(
                f"Checkpoint path does not exist: {checkpoint_path}"
            )
 
            print(f"Loading LoRA checkpoint from: {checkpoint_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                use_cache=False
            )
            model = PeftModel.from_pretrained(base_model, checkpoint_path)  # Now uses local path
            ref_model = None
        else:
            # Load fresh models
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                use_cache=False
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
 
    
    def create_dpo_config(self, config, output_dir):
        """Create DPO-specific training configuration"""
        return DPOConfig(
            output_dir=output_dir,
            num_train_epochs=3,  # DPO typically needs fewer epochs
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation"],
            learning_rate=config["learning_rate"],
            weight_decay=0.001,
            bf16=True,
            max_grad_norm=0.3,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            save_steps=15,  # Save less frequently for DPO
            logging_steps=1,
            eval_steps=15,
            max_steps=30,  # Fewer steps for DPO
            eval_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            max_length=config["max_length"],
            max_prompt_length=config["max_length"] // 2,  # DPO specific
            beta=config["beta"],  # DPO regularization parameter
            report_to="wandb",
            run_name=f"qwen-dpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
    
    def create_lora_config(self, config):
        return LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )
    
    def train_model(self, model_key):
        if model_key not in self.model_configs:
            raise ValueError(f"Model {model_key} not found in configurations")
        
        config = self.model_configs[model_key]
        output_dir = os.path.join(self.base_output_dir, model_key)
        
        print(f"\n{'='*60}")
        print(f"Training DPO {model_key}: {config['model_name']}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")
        
        try:
            model, tokenizer = self.setup_model_and_tokenizer(config, self.sft_checkpoint_path)
            training_args = self.create_dpo_config(config, output_dir)
            lora_config = self.create_lora_config(config)
                # Create DPO trainer
            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.test_dataset,
                processing_class=tokenizer,
                peft_config=lora_config,
                # callbacks=[s3_callback]
            )
            
            trainer.train()
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)
            
            config_path = os.path.join(output_dir, "training_config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Successfully trained and saved DPO {model_key}")
            
            del trainer, model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"Error training DPO {model_key}: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Train a DPO model")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["qwen-0.5b", "qwen-1.5b", "qwen-7b"],
        required=True,
        help="Model to train"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="data/esci/dpo_dataset.jsonl",
        help="Path to DPO dataset"
    )

    parser.add_argument(
    "--sft_checkpoint_path", 
    type=str, 
    default=None,   
    help="Path to SFT checkpoints for DPO init (e.g., outputs/sft_models/qwen-7b/checkpoint-150)"
)
    parser.add_argument("--dataset", default="data/esci/dpo_dataset.jsonl")
    args = parser.parse_args()
    
    trainer = ESCISearchTrainerDPOTrainer(dataset_path=args.dataset, sft_checkpoint_path = args.sft_checkpoint_path)
    success = trainer.train_model(args.model)

if __name__ == "__main__":
    main()