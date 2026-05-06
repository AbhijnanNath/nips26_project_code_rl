"""
SFT Training Script 
Usage: python train_sft.py --model qwen-7b --dataset data/hnm/dpo_dataset.jsonl
or accelerate launch [xx] format. 
"""

import os
import json
import torch
from datetime import datetime
from datasets import load_from_disk, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
import random
from peft import LoraConfig
import gc
from transformers import TrainerCallback
 
 
class UserProfileSFTTrainer:
    def __init__(self, dataset_path="sft_all_high_quality.jsonl"):
        self.dataset_path = dataset_path
        self.base_output_dir = "./sft_trained_models"
        
        # Model configurations for different sizes
        self.model_configs = {
            "qwen-0.5b": {
                "model_name": "Qwen/Qwen2.5-0.5B",
                "batch_size": 8,
                "gradient_accumulation": 1,
                "learning_rate": 2e-4,
                "lora_r": 16,
                "lora_alpha": 32,
                "max_length": 2048
            },
            "qwen-1.5b": {
                "model_name": "Qwen/Qwen2.5-1.5B", 
                "batch_size": 4,
                "gradient_accumulation": 2,
                "learning_rate": 1e-4,
                "lora_r": 32,
                "lora_alpha": 64,
                "max_length": 2048
            },
            "qwen-7b": {
                "model_name": "Qwen/Qwen2.5-7B-Instruct",
                "batch_size": 6,
                "gradient_accumulation": 4,
                "learning_rate": 5e-5,
                "lora_r": 32,
                "lora_alpha": 16,
                "max_length": 512
            },
            "qwen-14b": {
                "model_name": "Qwen/Qwen2.5-14B",
                "batch_size": 4,
                "gradient_accumulation": 8,
                "learning_rate": 3e-5,
                "lora_r": 64,
                "lora_alpha": 128,
                "max_length": 4096
            },
            "qwen-32b": {
                "model_name": "Qwen/Qwen2.5-32B",
                "batch_size": 1,
                "gradient_accumulation": 16,
                "learning_rate": 2e-5,
                "lora_r": 64,
                "lora_alpha": 128,
                "max_length": 4096
            }
        }
        
        # Load dataset once
        self.load_dataset()
 
    
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

    def get_model_size_gb(self, model_name):
        """Estimate model size for memory management"""
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
        return 8  # default
    
    def setup_model_and_tokenizer(self, config):
        """Setup model and tokenizer with proper configurations"""
        model_name = config["model_name"]
        print(f"\nLoading model: {model_name}")
        
        # Determine dtype based on model size
        model_size_gb = self.get_model_size_gb(model_name)
        torch_dtype = torch.bfloat16 if model_size_gb <= 14 else torch.float16
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False  # Disable cache for training
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
 
        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    
    def create_training_config(self, config, output_dir):
        """Create training configuration"""
        return SFTConfig(
            output_dir=output_dir,
            num_train_epochs=1,  # Reduced for faster training across multiple models
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=config["gradient_accumulation"],
            learning_rate=config["learning_rate"],
            weight_decay=0.001,
            bf16=True,  # Use bfloat16 for stability
            max_grad_norm=0.3,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            save_steps=30,
            logging_steps=1,
            eval_steps=30,
            max_steps = 60, 
            eval_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            packing=False,  # Safer without flash attention if packing is false. 
            dataset_text_field="messages",
            max_length=config["max_length"],
            # assistant_only_loss = True, 
            report_to="wandb",
            run_name=f"qwen-sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
    
    def create_lora_config(self, config):
        """Create LoRA configuration"""
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
        """Train a single model"""
        if model_key not in self.model_configs:
            raise ValueError(f"Model {model_key} not found in configurations")
        
        config = self.model_configs[model_key]
        output_dir = os.path.join(self.base_output_dir, model_key)
        
        print(f"\n{'='*60}")
        print(f"Training {model_key}: {config['model_name']}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")
        
        try:
            model, tokenizer = self.setup_model_and_tokenizer(config)
            training_args = self.create_training_config(config, output_dir)
            lora_config = self.create_lora_config(config)
             
            # Create trainer
            trainer = SFTTrainer(
                model=model,
                # tokenizer=tokenizer,
                train_dataset=self.train_dataset,
                eval_dataset=self.test_dataset,
                args=training_args,
                peft_config=lora_config,
             
            )
            trainer.train()
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)
            
            # Save training config for reference
            config_path = os.path.join(output_dir, "training_config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Successfully trained and saved {model_key}")
            
            # Clean up memory
            del trainer, model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"Error training {model_key}: {str(e)}")
            return False
    
    def train_all_models(self, model_keys=None):
        """Train all models in sequence"""
        if model_keys is None:
            model_keys = list(self.model_configs.keys())
        
        results = {}
        
        print(f"Starting training for models: {', '.join(model_keys)}")
        
        for model_key in model_keys:
            success = self.train_model(model_key)
            results[model_key] = success
            
            if success:
                print(f"{model_key} completed successfully")
            else:
                print(f"{model_key} failed")
        
        # Print summary
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")
        for model_key, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            print(f"{model_key:15s} {status}")
        
        return results


def main():
    """Main training script"""
    # Initialize trainer
    trainer = UserProfileSFTTrainer()
    # Train the model with SFT training
    results = trainer.train_model("qwen-7b")

if __name__ == "__main__":
    main()