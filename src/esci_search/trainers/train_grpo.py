# Code released for anonymous review. License: CC-BY-NC-4.0  


import os
import re
import json, sys, traceback
import math
import time
import random
import logging
import pickle
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union
# Set NCCL timeout and monitoring before importing torch
os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '1800'  # 30 minutes
os.environ['TORCH_NCCL_ENABLE_MONITORING'] = '0'
os.environ['NCCL_DEBUG'] = 'INFO'
import torch
import torch.distributed as dist
from accelerate import logging
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from peft import LoraConfig, PeftConfig
import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    HfArgumentParser, 
    set_seed, 
    BitsAndBytesConfig
)
from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from dense_search.search import SearchRewardFunction, _unwrap_search_instance
logger = logging.get_logger(__name__)

script_dir = Path(__file__).resolve().parent  
project_root = script_dir.parent.parent.parent
DATASET_PATH = project_root / 'data' / 'esci' / 'rl_dataset'


@dataclass
class ScriptArguments:
    """
    The arguments for the GRPO training script.
    """
    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parametersQwen/Qwen2.5-7B-Instruct
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen2.5-1.5B-Instruct",  
        metadata={"help": "the location of the SFT model name or path"},
    )
    #data and index paths
    dataset_path: str = field(
        default="data/esci/rl_dataset",
        metadata={"help": "Path to RL dataset"}
    )
    faiss_index_path: str = field(
        default="data/esci/index/all-mpnet-base-v2_faiss.bin",
        metadata={"help": "Path to FAISS index"}
    )
    asin_mapping_path: str = field(
        default="data/esci/index/all-mpnet-base-v2_asin_mapping.json",
        metadata={"help": "Path to ASIN mapping JSON"}
    )
    metadata_path: str = field(
        default="data/esci/metadata/item_catalog.jsonl",
        metadata={"help": "Path to item catalog"}
    )
    embedding_model: str = field(
        default="sentence-transformers/all-mpnet-base-v2",
        metadata={"help": "Embedding model for search"}
    )
 
    learning_rate: Optional[float] = field(default=5.0e-7, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    
    kl_beta: Optional[float] = field(default=0.0, metadata={"help": "KL coefficient. If `0.0` (default), the reference model is not loaded, reducing memory usage and "
            "improving training speed."})
    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=2, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=2, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    use_peft: Optional[bool] = field(default=True, metadata={"help": "whether to use PEFT or Low-rank approx method"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=32, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=256, metadata={"help": "the maximum prompt length"})
    max_completion_length: Optional[int] = field(default=356, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=2000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    log_completions: Optional[bool] = field(default=True, metadata={"help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps."})
    save_steps: Optional[int] = field(default=500, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=200, metadata={"help": "the evaluation frequency"})
    eval_strategy: Optional[str] = field(default="steps", metadata={"help": "the evaluation strategy"})
    num_generations: Optional[int] = field(default=8, metadata={"help": "MCMC sampling for GRPO"})
    faiss_topk: Optional[int] = field(default=1000, metadata={"help": "FAISS index retrieval top before reward computation in GRPO"})
    output_dir: Optional[str] = field(default="Qwen2-0.5B-DPO", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    torch_dtype: Optional[str] = field(
    default="bfloat16", metadata={"help": "torch_dtype[fp16, bfloat16, float] for loading."}
)
    # instrumentation
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
  
def reward_num_unique_letters(completions, **kwargs):
    start = time.time()
    result = [float(len(set(completion[0]["content"]))) for completion in completions]
    print(f"Reward computation took: {time.time() - start:.2f}s for {len(completions)} completions")
    return result
 
def custom_get_quantization_config(script_args: ScriptArguments) -> Optional[BitsAndBytesConfig]:
    if script_args.load_in_4bit:
        # Map string to torch.dtype object
        if script_args.torch_dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        elif script_args.torch_dtype == "float16":
            compute_dtype = torch.float16
        else:
            compute_dtype = torch.float32 # Fallback or explicit float32

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,  
            bnb_4bit_quant_type="nf4",           
            bnb_4bit_use_double_quant=True,           
            # bnb_4bit_quant_storage should generally match compute_dtype for optimal performance
            bnb_4bit_quant_storage=compute_dtype, 
        )
    else:
        quantization_config = None

    return quantization_config

class FixedGRPOConfig(GRPOConfig):
    """
    Custom GRPO config that fixes the __post_init__ logic to handle
    generation_batch_size and steps_per_generation properly.
    """
    def __post_init__(self):
   
        if script_args.torch_dtype == "bf16": # Using script_args here, assuming it's available
             self.bf16 = True
             self.fp16 = False # Ensure fp16 is False if bf16 is True
        elif script_args.torch_dtype == "float16":
             self.fp16 = True
             self.bf16 = False
        else: # Default or float32
             self.fp16 = False
             self.bf16 = False

        # Call the parent's parent __post_init__ (skip TRL's GRPOConfig's buggy version)
        # This handles all the TrainingArguments validation
        
        transformers.TrainingArguments.__post_init__(self)
        
        # Now handle GRPO-specific logic correctly
        num_processes = self.world_size
        # Custom logic: Just ensure both values are set properly
        if self.generation_batch_size is None and self.steps_per_generation is None:
            # Default case: use gradient accumulation steps
            self.steps_per_generation = self.gradient_accumulation_steps
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
            
        elif self.generation_batch_size is not None and self.steps_per_generation is None:
            # User set generation_batch_size, calculate steps_per_generation
            global_batch_size = self.per_device_train_batch_size * num_processes
            if self.generation_batch_size % global_batch_size != 0:
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size "
                    f"({global_batch_size})."
                )
            self.steps_per_generation = self.generation_batch_size // global_batch_size
            
        elif self.generation_batch_size is None and self.steps_per_generation is not None:
            # User set steps_per_generation, calculate generation_batch_size
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
            
        else:
            # Both are set - instead of failing, use steps_per_generation and recalculate generation_batch_size
            # This fixes the mutual exclusion bug from TRL
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        
        # Handle evaluation batch size validation
        if self.do_eval and self.eval_strategy != "no":
            if (self.per_device_eval_batch_size * num_processes) % self.num_generations != 0:
                raise ValueError(
                    f"The global eval batch size ({self.per_device_eval_batch_size} * {num_processes}) must be "
                    f"divisible by num_generations ({self.num_generations})."
                )
        
        # Ensure generation_batch_size is divisible by num_generations
        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by num_generations "
                f"({self.num_generations})."
            )
            
        # Ensure minimum generations requirement
        if self.num_generations < 2:
            raise ValueError(
                "GRPO requires at least 2 generations per prompt to calculate the advantages. You provided "
                f"{self.num_generations}, which is less than the minimum required."
            )

# ==================== CUSTOM GRPO TRAINER WITH SEARCH ====================

class SearchTrainer(GRPOTrainer):
    """
    Custom RL trainer for search and retrieval based RL training. 
    Primarily supports GRPO advantage estimation. 
    Inherits from GRPOTrainer and adds task-specific reward functions and logging.
    """
    
    def __init__(self, *args, **kwargs):
        # Set default reward functions if not provided
        if 'reward_funcs' not in kwargs:
            kwargs['reward_funcs'] = [
                ndcg_at_k_reward_function,
                exact_xml_format_reward_function,
                valid_item_ids_reward_function,
                ranking_completeness_reward_function,
                reasoning_quality_reward_function,
                top_k_accuracy_reward_function
            ]
        
        # Set default reward weights if not provided
        # NDCG(60%), Format(5%), Valid IDs(10%), Completeness(10%), Reasoning(5%), Top-K(10%)
        if hasattr(kwargs.get('args', None), 'reward_weights'):
            pass  # Use provided weights
        else:
            # Will be set after super().__init__()
            self._default_reward_weights = [0.6, 0.05, 0.1, 0.1, 0.05, 0.1]
        
        super().__init__(*args, **kwargs)
        # Add retrieval logging
 
        self._logs["target_item"] = []
        self._logs["retrieved_items"] = []
        self._logs["expanded_queries"] = []
        self._logs["retrieval_ranks"] = []
        self._logs["all_rewards"] = []
        # Global accumulator for all data across steps
        self._global_results = []
        
        # Set default weights if not already set
        if hasattr(self, '_default_reward_weights'):
            import torch
            self.reward_weights = torch.tensor(self._default_reward_weights)
    
    def _generate_and_score_completions(self, inputs):
        """Override to capture retrieval info after generation"""
        self._logs["target_item"].clear()
        self._logs["retrieved_items"].clear()
        self._logs["expanded_queries"].clear()
        self._logs["retrieval_ranks"].clear()
        self._logs["all_rewards"].clear()
        # Extract target items from inputs BEFORE calling parent
        
        for input_item in inputs:
            target = input_item.get('target_item_id', input_item.get('answer', ''))
            self._logs["target_item"].append(target)
        # Call parent FIRST to generate completions and compute rewards
        output = super()._generate_and_score_completions(inputs)
        
        # Extract retrieval info after rewards are computed
        if hasattr(self, 'reward_funcs'):
            for reward_func in self.reward_funcs:
                if hasattr(reward_func, '__self__') and hasattr(reward_func.__self__, 'last_retrievals'):
                    for retrieval in reward_func.__self__.last_retrievals:
                        # Use 'retrieved_top10' to match your actual dict key
                        self._logs["retrieved_items"].append(retrieval.get('retrieved_items', retrieval.get('retrieved_items', [])))
                        self._logs["expanded_queries"].append(retrieval.get('expanded_query', ''))
                        self._logs["retrieval_ranks"].append(retrieval.get('rank'))
        
        return output

    def _parse_candidate_items(self, question_text: str) -> dict:
        """Parse 'From these candidate items,' section to extract ID->Title mapping."""
        pattern = r'From these candidate items,.*?:\n(.*?)(?=\n\n|\Z)'
        match = re.search(pattern, question_text, re.DOTALL)
        
        if not match:
            return {}
        
        items_section = match.group(1)
        item_mapping = {}
        
        # Parse each line: "1. [753737001]: Twenty HW tapered - Trousers in Black {price: $0.04}"
        for line in items_section.split('\n'):
            if line.strip():
                item_match = re.match(r'\d+\.\s*\[(\w+)\]:\s*([^{]+)', line.strip())
                if item_match:
                    item_id = item_match.group(1)
                    title = item_match.group(2).strip()
                    item_mapping[item_id] = title
        
        return item_mapping


    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """Override to add custom tracked metrics logging and global data storage."""
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}
        
        # Add eval prefix if needed
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        logs = {**logs, **metrics}
        
        # Store current batch data for global accumulation
        if self._logs["prompt"]:  # Only if we have data
            current_batch_data = {
                'step': self.state.global_step,
                'mode': mode,
                'prompts': list(self._logs["prompt"]),
                'completions': list(self._logs["completion"]),
                'expanded_queries': list(self._logs["expanded_queries"]),
                'rewards': {name: list(rewards) for name, rewards in self._logs["rewards"].items()},
                'advantages': list(self._logs["advantages"]),
                'target_items': list(self._logs["target_item"]),
                'retrieved_items': list(self._logs["retrieved_items"]),
                'retrieval_ranks': list(self._logs["retrieval_ranks"]),
                'timestamp': __import__('time').time()
            }
            
            self._global_results.append(current_batch_data)
            
            if self.state.global_step % self.args.save_steps == 0 or self.state.global_step >= self.args.max_steps - 1:
                self._save_global_results()
                # self._print_sample_retrievals()
        
        # Call parent log method
        super().log(logs, start_time)
        
        # Custom wandb logging for search task
        if self.accelerator.is_main_process and self.log_completions:
            if self.args.report_to and "wandb" in self.args.report_to:
                try:
                    import wandb
                    if wandb.run is not None:
                        import pandas as pd
                        table = {
                            "step": [str(self.state.global_step)] * len(self._logs["prompt"]),
                            "prompt": list(self._logs["prompt"]),
                            "completion": list(self._logs["completion"]),
                            "expanded_query": list(self._logs["expanded_queries"]),
                            **{name: list(rewards) for name, rewards in self._logs["rewards"].items()},
                            "advantage": list(self._logs["advantages"]),
                            "target_item": list(self._logs["target_item"]),
                            "retrieved_top10": [str(items) for items in self._logs["retrieved_items"]],
                            "rank": list(self._logs["retrieval_ranks"]),
                        }
                        
                        df = pd.DataFrame(table)
                        if self.wandb_log_unique_prompts:
                            df = df.drop_duplicates(subset=["prompt"])
                        wandb.log({"search_completions": wandb.Table(dataframe=df)})
                except ImportError:
                    logger.warning("wandb not available for custom logging")

    def _print_sample_retrievals(self):
        if self._logs["prompt"] and self._logs["retrieved_items"]:
            idx = 0
            print(f"\n{'='*80}")
            print(f"SAMPLE RETRIEVAL AT STEP {self.state.global_step}")
            print(f"{'='*80}")
            print(f"Original Query: {self._logs['prompt'][idx][:150]}...")
            print(f"\nExpanded Query: {self._logs['expanded_queries'][idx][:200]}...")
            print(f"\nTarget Item: {self._logs['target_item'][idx]}")  # Fixed: ground_truth → target_item
            print(f"Top 5 Retrieved: {self._logs['retrieved_items'][idx][:5]}")  # Added [:5] to limit
            print(f"Rank: {self._logs['retrieval_ranks'][idx]}")
            
            # Handle rewards dict structure
            reward_val = self._logs['rewards']['search_ndcg_reward'][idx] if 'search_ndcg_reward' in self._logs['rewards'] else 0.0
            print(f"Reward: {reward_val:.4f}")
            print(f"All rewards: {self._logs['all_rewards'][idx]}")
            print(f"{'='*80}\n")
    
    def _save_global_results(self):
        """Save global results to pickle file."""
        if self.accelerator.is_main_process and self._global_results:
            output_path = f"{self.args.output_dir}/global_results_step_{self.state.global_step}.pkl"
            try:
                with open(output_path, 'wb') as f:
                    pickle.dump(self._global_results, f)
                logger.info(f"Saved global results to {output_path}")
            except Exception as e:
                logger.warning(f"Failed to save global results: {e}")
    
 
# ==================== UTILITY FUNCTIONS ====================

def extract_xml_content(text: str, tag: str) -> str:
    """Extract content from XML tags."""
    pattern = f'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_ranking_list(text: str) -> List[str]:
    """Extract comma-separated ranking from <ranking> tags."""
    ranking_content = extract_xml_content(text, "ranking")
    if not ranking_content:
        return []
    
    # Split by comma and clean up
    items = [item.strip() for item in ranking_content.split(',')]
    return [item for item in items if item]

def score_ndcg_at_k(pred_ranking: List[str], gt_ranking: List[str], k: int) -> float:
    """NDCG@k calculation with string normalization and duplicate filtering."""
    if not pred_ranking or not gt_ranking or k <= 0:
        return 0.0

    # Normalize to strings
    gt_ranking = [str(x) for x in gt_ranking]
    pred_ranking = [str(x) for x in pred_ranking]

    # Relevance: higher for earlier GT positions
    rel = {item: (len(gt_ranking) - i) for i, item in enumerate(gt_ranking)}

    def dcg(order):
        val = 0.0
        for i, iid in enumerate(order, start=1):
            if iid in rel:
                val += (2**rel[iid] - 1) / math.log2(i + 1)
        return val

    # Remove duplicates while preserving order
    seen = set()
    pred_unique = []
    for x in pred_ranking:
        if x in rel and x not in seen:
            pred_unique.append(x)
            seen.add(x)

    if not pred_unique:
        return 0.0

    # Apply cutoff k
    pred_k = pred_unique[:k]
    idcg_k = dcg(gt_ranking[:k])
    if idcg_k == 0:
        return 0.0

    return dcg(pred_k) / idcg_k

def search_ndcg_reward(prompts, completions, target_item_id, **kwargs):
        return search_reward_instance(prompts, completions, target_item_id, **kwargs)

def load_data(dataset_path):
    dataset = DatasetDict.load_from_disk(dataset_path)
    dummy_test_indices = random.sample(range(len(dataset['train'])), 10)
    eval_dataset = dataset['train'].select(dummy_test_indices)
    train_dataset = dataset['train'].select([i for i in range(len(dataset['train'])) if i not in dummy_test_indices])
    return train_dataset, eval_dataset

if __name__ == "__main__":

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
    if "--config" in sys.argv:
        cfg_path = Path(sys.argv[sys.argv.index("--config") + 1]).expanduser()
        
        if not cfg_path.is_absolute() and not cfg_path.exists():
            cwd_path = Path.cwd() / cfg_path
            if cwd_path.exists():
                cfg_path = cwd_path
            else:
                script_dir = Path(__file__).resolve().parent
                esci_search_dir = script_dir.parent
                cfg_path = (esci_search_dir / cfg_path).resolve()
        
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
            
        print(f"Loading config from: {cfg_path}")
        script_args = parser.parse_yaml_file(str(cfg_path))[0]
    else:
        script_args = parser.parse_args_into_dataclasses()[0]
    
    # Project root is: LLM-Seq-Shapley-Owen-PO/
    script_dir = Path(__file__).resolve().parent  # trainers/
    esci_search_dir = script_dir.parent  # esci_search/
    src_dir = esci_search_dir.parent  # src/
    project_root = src_dir.parent  # LLM-Seq-Shapley-Owen-PO/
    
    # Convert relative paths to absolute (relative to project root)
    dataset_path = project_root / script_args.dataset_path
    faiss_path = project_root / script_args.faiss_index_path
    asin_mapping_path = project_root / script_args.asin_mapping_path
    print(f"Project root: {project_root}")
    print(f"Dataset path: {dataset_path}")
    print(f"Dataset exists: {dataset_path.exists()}")
    print("CONFIG LOADED output_dir:", script_args.output_dir)
    print("CONFIG LOADED dataset_path:", dataset_path)
    print("CONFIG LOADED faiss_index_path:", faiss_path)
    print("CONFIG LOADED model_name_or_path:", script_args.model_name_or_path)
    print("CONFIG LOADED output_dir:", script_args.output_dir)
    # set seed
    set_seed(script_args.seed)
    # Determine the actual torch_dtype object
    if script_args.torch_dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        actual_torch_dtype = torch.bfloat16
    elif script_args.torch_dtype == "float16":
        actual_torch_dtype = torch.float16
    else:
        actual_torch_dtype = torch.float32 # Fallback
  
    # 2. Get quantization config (for model loading)
    # Using the custom_get_quantization_config to ensure correct dtype is passed
    quantization_config = custom_get_quantization_config(script_args)
    print(f"Quantization config: {quantization_config}")

    #get the ModelConfig
    model_config_for_peft = ModelConfig(
    model_name_or_path=script_args.model_name_or_path,
    use_peft=script_args.use_peft,
    lora_r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    load_in_4bit=script_args.load_in_4bit,
    torch_dtype="bfloat16",  
    bnb_4bit_quant_type="nf4",           
    use_bnb_nested_quant=True,           
)
    # Get the PEFT config using TRL's utility
    peft_config = get_peft_config(model_config_for_peft)
    print(f"PEFT config: {peft_config}")
    print(f"actual_torch_dtype config: {actual_torch_dtype}") 
     
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,  
        torch_dtype=actual_torch_dtype, # Use the determined actual_torch_dtype
        quantization_config=quantization_config,   
        device_map=get_kbit_device_map(),          
        trust_remote_code=True,  
        # attn_implementation="flash_attention_2" if flash attention is installed. 
    )
    # Initialize the GRPOConfig properly, ensuring bf16/fp16 flags are set, Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-7B-Instruct-GRPO
    training_args = FixedGRPOConfig(
        output_dir="Qwen/Qwen2.5-7B-Instruct-GRPO-search-esci-pooled", 
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        num_generations=script_args.num_generations,
        logging_steps=script_args.logging_steps,
        learning_rate=script_args.learning_rate,
        gradient_checkpointing=script_args.gradient_checkpointing,
        ddp_find_unused_parameters=False,  
        bf16= True,
        fp16= False,
        # GRPO specific parameters
        generation_batch_size=64, 
        steps_per_generation=4,    
        max_prompt_length=script_args.max_prompt_length,
        max_completion_length=script_args.max_completion_length,
        beta=script_args.kl_beta,
        remove_unused_columns = False,
        eval_steps=script_args.eval_steps, 
        eval_strategy=script_args.eval_strategy,
        max_steps=script_args.max_steps,
        save_steps=script_args.save_steps,
 

    )
    print("training args AFTER change", training_args)
    # load train and test data
    train_dataset, eval_dataset = load_data(str(dataset_path)) 
    print("Shallow print of train set", train_dataset)
    print("Shallow print of eval set", eval_dataset)
    # Extract candidate pools from dataset for filtered NDCG computation. 
    train_pools = {
        row['sample_idx']: row['candidate_pool'] 
        for row in train_dataset
    }
    eval_pools = {row['sample_idx']: row['candidate_pool'] for row in eval_dataset}
    all_pools = {**train_pools, **eval_pools}
    search_reward_instance = SearchRewardFunction(
                faiss_index_path=str(faiss_path),
                mapping_path=str(asin_mapping_path),
                model_name=script_args.embedding_model,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                top_k=script_args.faiss_topk,
                candidate_pools=all_pools
            )
   
    trainer = SearchTrainer(
        model=model,
        reward_funcs=[search_ndcg_reward],
        reward_processing_classes=[None],  
        # reward_func_names=['search_ndcg'],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
  
    )
    try:
        trainer.train()
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

 