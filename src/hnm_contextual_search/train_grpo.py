import os
import re
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
from peft import PeftConfig, PeftModel
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
# from dense_search.search import SearchRewardFunction 
from dense_search.search_with_logging import SearchRewardFunction, _unwrap_search_instance
logger = logging.get_logger(__name__)
 
@dataclass
class ScriptArguments:
    """
    The arguments for the GRPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parametersQwen/Qwen2.5-7B-Instruct
    # model_name_or_path: Optional[str] = field(
    #     default="Qwen/Qwen2.5-7B-Instruct",  
    #     metadata={"help": "the location of the SFT model name or path"},
    # )

    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen2.5-1.5B-Instruct",  
        metadata={"help": "the location of the SFT model name or path"},
    )
    lora_model_name_or_path: Optional[str] = field(
        default="sft_trained_models_hnm/qwen-1.5b/checkpoint-222",  
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_path: Optional[str] = field(default="/home/greenland-user/data_bundles/Oct12_HNMs/hnm_rl_datasets_with_pools_sampleidx_unique_2_generic", metadata={"help": "the path to the RL dataset"})
    eval_dataset_path: Optional[str] = field(default="hnm_data_for_icml/contextual_search_test_with_pools_itemcontext_recreated", metadata={"help": "the path to the RL dataset"})
        # /home/greenland-user/home/greenland-user/hnm_context_search/src/hnm_rl_datasets_with_pools
    # /home/greenland-user/home/greenland-user/hnm_context_search/src/contextual_search_grpo_with_history
    # /home/greenland-user/home/greenland-user/hnm_context_search/src/hnm_rl_datasets_with_pools_sampleidx

    learning_rate: Optional[float] = field(default=5.0e-6, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    
    kl_beta: Optional[float] = field(default=0.0, metadata={"help": "KL coefficient. If `0.0` (default), the reference model is not loaded, reducing memory usage and "
            "improving training speed."})
    per_device_train_batch_size: Optional[int] = field(default=8, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=8, metadata={"help": "eval batch size per device"})
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

    max_prompt_length: Optional[int] = field(default=356, metadata={"help": "the maximum prompt length"})
    max_completion_length: Optional[int] = field(default=786, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=2000, metadata={"help": "max number of training steps"})
    #search parameters
    faiss_topk: Optional[int] = field(default=1000, metadata={"help": "FAISS index retrieval top before reward computation in OSPO"})
    reward_topk: Optional[int] = field(default=100, metadata={"help": "Additional candidate filtering after FAISS retrieval for metrics"}) #not used in experiments, since pool is used,. 

    logging_steps: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    log_completions: Optional[bool] = field(default=True, metadata={"help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps."})
    save_steps: Optional[int] = field(default=500, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=200, metadata={"help": "the evaluation frequency"})
    eval_strategy: Optional[str] = field(default="steps", metadata={"help": "the evaluation strategy"})
    num_generations: Optional[int] = field(default=8, metadata={"help": "MCMC sampling for GRPO"})
    use_candidate_pools: Optional[bool] = field(default=True, metadata={"help": "whether to use candidate pools in retrieval NDCG computation"})
     
    output_dir: Optional[str] = field(default="Qwen2-0.5B-DPO", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    train_data_sanity_check_size: Optional[int] = field(default=100, metadata={"help": "Smaller train sizes for test runs"})
    eval_data_sanity_check_size: Optional[int] = field(default=200, metadata={"help": "Smaller eval sizes for test runs"})
    num_generations_eval: Optional[int] = field(default=1, metadata={"help": "num_generations_eval for evals vs train where num generations is used"})
     
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

    RETRIEVAL_LOG_KEYS = [
            "retrieved_items", "expanded_queries", "retrieval_ranks", "target_item",
            "pool_size", "n_raw", "n_filt",
            "overlap_uniq", "overlap_rate",
            "target_in_faiss_top1000",
            "first_hit_rank_raw", "first_hit_rank_post",
            "zero_reward",
            "ndcg", "ap", "mrr", "recall",
            "ndcg@pool", "recall@pool", "mrr@pool", "ap@pool",
            "post_first10",
            # if you use it:
            "all_rewards",
        ]
    
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
        self._logs = getattr(self, "_logs", {})
        for k in self.RETRIEVAL_LOG_KEYS:
            self._logs.setdefault(k, [])
        
        # Global accumulator for all data across steps
        self._global_results = []
        
        # Set default weights if not already set
        if hasattr(self, '_default_reward_weights'):
            import torch
            self.reward_weights = torch.tensor(self._default_reward_weights)
    
    def clear_retrieval_logs(self):
        # Use self.RETRIEVAL_LOG_KEYS
        for k in self.RETRIEVAL_LOG_KEYS:
            lst = self._logs.get(k)
            if isinstance(lst, list):
                lst.clear()


    def _generate_and_score_completions(self, inputs):
        # Initialize keys
        retrieval_keys = [
            "target_in_pool", "target_in_faiss_top1000", "zero_reward",
            "ndcg", "ap", "mrr", "recall",
            "ndcg@pool", "recall@pool", "mrr@pool", "ap@pool",
            "first_hit_rank_raw", "first_hit_rank_post",
            "pool_size", "n_raw", "n_filt", "overlap_uniq", "overlap_rate"
        ]
        for key in retrieval_keys:
            if key not in self._logs:
                self._logs[key] = []
        
        # Clear and capture targets
        self.clear_retrieval_logs()
        for input_item in inputs:
            target = input_item.get('target_item_id', input_item.get('answer', ''))
            self._logs["target_item"].append(target)
        
        # Call parent
        output = super()._generate_and_score_completions(inputs)
        
        # FIRST: Extract from reward functions (POPULATE LOGS)
        if hasattr(self, 'reward_funcs'):
            for rf in self.reward_funcs:
                logs_src = None

                # Case 1: rf itself is an instance with last_retrievals
                if hasattr(rf, 'last_retrievals'):
                    logs_src = rf.last_retrievals

                if self.accelerator.is_main_process:
                    for r in logs_src:
                        self._logs["target_in_pool"].append(bool(r.get("target_in_pool", False)))
                        self._logs["target_in_faiss_top1000"].append(bool(r.get("target_in_faiss_top1000", False)))
                        self._logs["zero_reward"].append(bool(r.get("zero_reward", False)))

                        for metric in ["ndcg", "ap", "mrr", "recall"]:
                            self._logs[metric].append(r.get(metric, 0.0))  # default to 0.0

                        for metric in ["ndcg@pool", "recall@pool", "mrr@pool", "ap@pool"]:
                            self._logs[metric].append(r.get(metric, 0.0))

        
        # SECOND: Aggregate (READ FROM LOGS)
        def _safe_mean(seq):
            seq = [x for x in seq if x is not None]
            return float(sum(seq) / len(seq)) if seq else float('nan')

        def _safe_rate(bools):
            vals = [1.0 if bool(x) else 0.0 for x in bools]
            return _safe_mean(vals)
        
        mode = "train" if self.model.training else "eval"
        L = self._logs
        M = self._metrics[mode]
        
        M.setdefault("retrieval/targets_in_faiss_top1000_rate", []).append(
            _safe_rate(L.get("target_in_faiss_top1000", []))
        )
        M.setdefault("retrieval/target_in_pool_rate", []).append(
            _safe_rate(L.get("target_in_pool", [])) if "target_in_pool" in L else float('nan')
        )
        
        # Effectiveness metrics
        for key_src, key_dst in [
            ("ndcg@pool",   "ndcg_post@pool_mean"),
            ("mrr@pool",    "mrr_post@pool_mean"),
            ("recall@pool", "recall_post@pool_mean"),
            ("ap@pool",     "ap_post@pool_mean"),
            ("ndcg",        "ndcg_post@post_mean"),
            ("mrr",         "mrr_post@post_mean"),
            ("recall",      "recall_post@post_mean"),
            ("ap",          "ap_post@post_mean"),
        ]:
            if key_src in L:
                M.setdefault(f"retrieval/{key_dst}", []).append(_safe_mean(L[key_src]))
        
        # Zero reward rate
        M.setdefault("retrieval/zero_reward_rate", []).append(
            _safe_rate(L.get("zero_reward", []))
        )
        # SECOND: Aggregate (READ FROM LOGS)
        def _safe_mean(seq):
            seq = [x for x in seq if x is not None]
            return float(sum(seq) / len(seq)) if seq else float('nan')

        def _safe_rate(bools):
            vals = [1.0 if bool(x) else 0.0 for x in bools]
            return _safe_mean(vals)
        
        mode = "train" if self.model.training else "eval"
        L = self._logs
        M = self._metrics[mode]
        
        M.setdefault("retrieval/targets_in_faiss_top1000_rate", []).append(
            _safe_rate(L.get("target_in_faiss_top1000", []))
        )
        M.setdefault("retrieval/target_in_pool_rate", []).append(
            _safe_rate(L.get("target_in_pool", [])) if "target_in_pool" in L else float('nan')
        )
        
        # Effectiveness metrics
        for key_src, key_dst in [
            ("ndcg@pool",   "ndcg_post@pool_mean"),
            ("mrr@pool",    "mrr_post@pool_mean"),
            ("recall@pool", "recall_post@pool_mean"),
            ("ap@pool",     "ap_post@pool_mean"),
            ("ndcg",        "ndcg_post@post_mean"),
            ("mrr",         "mrr_post@post_mean"),
            ("recall",      "recall_post@post_mean"),
            ("ap",          "ap_post@post_mean"),
        ]:
            if key_src in L:
                M.setdefault(f"retrieval/{key_dst}", []).append(_safe_mean(L[key_src]))
        
        # Zero reward rate
        M.setdefault("retrieval/zero_reward_rate", []).append(
            _safe_rate(L.get("zero_reward", []))
        )

        return output

   
    def _extract_and_store_metadata(self, inputs: list[dict[str, Union[torch.Tensor, Any]]]):
        """Extract item titles and candidate info from prompts."""
        for input_item in inputs:
            # Direct access to preprocessed fields
            customer_id = input_item.get('customer_id', '')
            ground_truth_with_titles = input_item.get('ground_truth_with_titles', [])
            answer = input_item.get('answer', '')  # Raw comma-separated IDs
            
            # Store in logs
            self._logs["customer_id"].append(customer_id)
            self._logs["ground_truth_with_titles"].append(ground_truth_with_titles)
            self._logs["ground_truth"].append(answer)
    
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
                 'retrieval': {
                    key.replace('retrieval/', ''): val[-1] if val else None
                    for key, val in self._metrics[mode].items()
                    if key.startswith('retrieval/')
                },
                
                'timestamp': __import__('time').time()
            }
            
            self._global_results.append(current_batch_data)
            
            if self.state.global_step % self.args.save_steps == 0 or self.state.global_step >= self.args.max_steps - 1:
                self._save_global_results()
                self._print_sample_retrievals()
        
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
                            "target_item": list(self._logs["ground_truth"]),
                            "retrieved_top10": [str(items) for items in self._logs["retrieved_items"]],
                            "rank": list(self._logs["retrieval_ranks"]),
                        }
                        
                        df = pd.DataFrame(table)
                        if self.wandb_log_unique_prompts:
                            df = df.drop_duplicates(subset=["prompt"])
                        wandb.log({"search_completions": wandb.Table(dataframe=df)})
                except ImportError:
                    logger.warning("wandb not available for custom logging")

    def _print_sample_retrievals(self, n_samples=3):
        """Print N sample retrievals for debugging"""
        if not self._logs.get("prompt") or not self._logs.get("retrieved_items"):
            print("No retrieval logs available")
            return
        
        # Determine how many samples to print
        num_available = len(self._logs["prompt"])
        n_samples = min(n_samples, num_available)
        
        print(f"\n{'='*80}")
        print(f"SAMPLE RETRIEVALS AT STEP {self.state.global_step} ({n_samples} samples)")
        print(f"{'='*80}\n")
        
        for i in range(n_samples):
            print(f"--- Sample {i+1} ---")
            
            # Original query
            if i < len(self._logs.get('prompt', [])):
                print(f"Original Query: {self._logs['prompt'][i][:150]}...")
            else:
                print(f"Original Query: Not available")
            
            # Expanded query
            if 'expanded_queries' in self._logs and i < len(self._logs['expanded_queries']):
                print(f"\nExpanded Query: {self._logs['expanded_queries'][i][:200]}...")
            else:
                print(f"\nExpanded Query: Not available")
            
            # Target item
            if 'target_item_id' in self._logs and i < len(self._logs['target_item_id']):
                print(f"\nTarget Item: {self._logs['target_item_id'][i]}")
            else:
                print(f"\nTarget Item: Not available")
            
            # Retrieved items
            if 'retrieved_items' in self._logs and i < len(self._logs['retrieved_items']):
                retrieved = self._logs['retrieved_items'][i]
                print(f"Top 5 Retrieved: {retrieved[:5] if retrieved else 'None'}")
            else:
                print(f"Top 5 Retrieved: Not available")
            
            # Rank
            if 'retrieval_ranks' in self._logs and i < len(self._logs['retrieval_ranks']):
                print(f"Rank: {self._logs['retrieval_ranks'][i]}")
            else:
                print(f"Rank: Not available")
            
            # Reward
            if 'rewards' in self._logs and 'search_ndcg_reward' in self._logs['rewards'] and i < len(self._logs['rewards']['search_ndcg_reward']):
                print(f"Reward: {self._logs['rewards']['search_ndcg_reward'][i]:.4f}")
            else:
                print(f"Reward: Not available")
            
            print()  # Blank line between samples
        
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

# ==================== REWARD FUNCTIONS ====================

def ndcg_at_k_reward_function(prompts, completions, answer, k=5, **kwargs):
    """NDCG@k reward function for ranking quality."""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for i, response in enumerate(responses):
        try:
            pred_ranking = extract_ranking_list(response)
            gt_ranking_str = answer[i] if i < len(answer) else ""
            gt_ranking = [item.strip() for item in gt_ranking_str.split(',') if item.strip()]
            
            ndcg_score = score_ndcg_at_k(pred_ranking, gt_ranking, k)
            rewards.append(ndcg_score)
            
        except Exception as e:
            logger.warning(f"Error calculating NDCG for sample {i}: {e}")
            rewards.append(0.0)
    
    return rewards

def exact_xml_format_reward_function(prompts, completions, answer, **kwargs):
    """Reward for exact XML format: <thinking>...</thinking><ranking>...</ranking>"""
    responses = [completion[0]['content'] for completion in completions]
    # Strict pattern matching the exact format
    pattern = r'^<thinking>\n.*?\n</thinking>\n<ranking>\n.*?\n</ranking>$'
    rewards = []
    
    for response in responses:
        try:
            match = re.match(pattern, response.strip(), re.DOTALL)
            rewards.append(1.0 if match else 0.0)
        except Exception as e:
            logger.warning(f"Error checking XML format: {e}")
            rewards.append(0.0)
    
    return rewards

def valid_item_ids_reward_function(prompts, completions, answer, **kwargs):
    """Reward for using only valid item IDs from candidates."""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for i, response in enumerate(responses):
        try:
            pred_ranking = extract_ranking_list(response)
            
            # Extract candidate IDs from ground truth
            gt_ranking_str = answer[i] if i < len(answer) else ""
            valid_ids = set(item.strip() for item in gt_ranking_str.split(',') if item.strip())
            
            if not pred_ranking or not valid_ids:
                rewards.append(0.0)
                continue
            
            # Check how many predicted IDs are valid
            valid_preds = [item for item in pred_ranking if item in valid_ids]
            reward = len(valid_preds) / len(pred_ranking) if pred_ranking else 0.0
            rewards.append(reward)
            
        except Exception as e:
            logger.warning(f"Error validating item IDs for sample {i}: {e}")
            rewards.append(0.0)
    
    return rewards

def ranking_completeness_reward_function(prompts, completions, answer, **kwargs):
    """Reward for including all available candidates."""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for i, response in enumerate(responses):
        try:
            pred_ranking = extract_ranking_list(response)
            
            # Get expected number of items from ground truth
            gt_ranking_str = answer[i] if i < len(answer) else ""
            gt_ranking = [item.strip() for item in gt_ranking_str.split(',') if item.strip()]
            
            if not gt_ranking:
                rewards.append(0.0)
                continue
            
            expected_count = len(gt_ranking)
            actual_count = len(pred_ranking)
            
            # Full reward if all items ranked, proportional otherwise
            reward = min(1.0, actual_count / expected_count)
            rewards.append(reward)
            
        except Exception as e:
            logger.warning(f"Error calculating completeness for sample {i}: {e}")
            rewards.append(0.0)
    
    return rewards

def reasoning_quality_reward_function(prompts, completions, answer, min_length=100, **kwargs):
    """Reward for longer, more detailed reasoning in <thinking> tags."""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for response in responses:
        try:
            thinking_content = extract_xml_content(response, "thinking")
            
            if not thinking_content:
                rewards.append(0.0)
                continue
            
            # Reward based on length of reasoning
            reasoning_length = len(thinking_content.strip())
            
            if reasoning_length >= min_length:
                reward = 1.0
            else:
                # Proportional reward for shorter reasoning
                reward = reasoning_length / min_length
            
            rewards.append(min(reward, 1.0))  # Cap at 1.0
            
        except Exception as e:
            logger.warning(f"Error evaluating reasoning quality: {e}")
            rewards.append(0.0)
    
    return rewards

def top_k_accuracy_reward_function(prompts, completions, answer, k=3, **kwargs):
    """Reward for top-k accuracy (hit@k)."""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for i, response in enumerate(responses):
        try:
            pred_ranking = extract_ranking_list(response)
            
            gt_ranking_str = answer[i] if i < len(answer) else ""
            gt_ranking = [item.strip() for item in gt_ranking_str.split(',') if item.strip()]
            
            if not gt_ranking or not pred_ranking:
                rewards.append(0.0)
                continue
            
            # Check if GT top-1 is in predicted top-k
            gt_top_1 = gt_ranking[0]
            pred_top_k = pred_ranking[:k]
            
            reward = 1.0 if gt_top_1 in pred_top_k else 0.0
            rewards.append(reward)
            
        except Exception as e:
            logger.warning(f"Error calculating top-{k} accuracy for sample {i}: {e}")
            rewards.append(0.0)
    
    return rewards

def search_ndcg_reward(prompts, completions, target_item_id, **kwargs):
        return search_reward_instance(prompts, completions, target_item_id, **kwargs)

def load_data(dataset_path):
    dataset = DatasetDict.load_from_disk(dataset_path)
    print(dataset['train'][0]['prompt'][-1]['content'])
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    print("train_dataset", len(train_dataset))
    print("eval_dataset", len(eval_dataset))
    
    return train_dataset, eval_dataset

if __name__ == "__main__":

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
    # load rl dataset. 
   
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
        dtype=actual_torch_dtype, # Use the determined actual_torch_dtype
        quantization_config=quantization_config,   
        device_map=get_kbit_device_map(),          
        trust_remote_code=True,  
        # attn_implementation="flash_attention_2" if flash attention is installed. 
    )


     # attach LoRA weights if provided
    if hasattr(script_args, "lora_model_name_or_path") and script_args.lora_model_name_or_path:
        print(f"  Loading LoRA weights from: {script_args.lora_model_name_or_path}")
        model = PeftModel.from_pretrained(
            model,
            script_args.lora_model_name_or_path,
            is_trainable=True,   # keep trainable for further fine-tuning
        )
        print(f"  ✓ LoRA attached — trainable params: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    else:
        print("  No LoRA path provided — training from base model")

            #add lora lora_model_name_or_path


    # Initialize the GRPOConfig properly, ensuring bf16/fp16 flags are set, Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-7B-Instruct-GRPO
    training_args = FixedGRPOConfig(
        output_dir=f"Qwen/Qwen2.5-7B-Instruct-GRPO-hnm-search_{script_args.output_dir}", 
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
        # num_generations_eval = script_args.num_generations_eval
        # reward_weights=[0.9, 0.1]  #testing with sem sim reward from SemanticSimilarityReward; can remove. 
 
 

    )
    print("training args AFTER change", training_args)
    
    # train_dataset, eval_dataset = load_data(script_args.dataset_path) 
    train_dataset, _ = load_data(script_args.dataset_path) 
    eval_dataset = load_from_disk(script_args.eval_dataset_path) 


    print("Train dataset loaded:", train_dataset) 
    print("Eval dataset loaded:", eval_dataset)

    eval_dataset = eval_dataset.select(range(script_args.eval_data_sanity_check_size))
    print("final len train with train_data_sanity_check_size", train_dataset)
    print("final len eval_dataset with eval_data_sanity_check_size ", eval_dataset)


    def check_coverage(dsplit):
        total = len(dsplit)
        misses = []
        for ex in dsplit:
            tgt  = str(ex["target_item_id"]).strip()
            pool = [str(x).strip() for x in ex["candidate_pool"]]
            if tgt not in pool:
                misses.append((ex["global_idx"], tgt, pool[:12]))
        print(f"[COVERAGE] split={dsplit}: {total-len(misses)}/{total} "
            f"({(total-len(misses))*100/total:.4f}%) have target in pool")
        if misses:
            print(f"[SAMPLE MISS] count={len(misses)}  first_miss={misses[0]}")
        return misses
    train_misses = check_coverage(train_dataset)
    test_misses  = check_coverage(eval_dataset)
    if script_args.use_candidate_pools:
        print("USING  use_candidate_pools")
        # Extract candidate pools from dataset for filtered NDCG computation. 
        train_pools = {
            row['sample_idx']: row['candidate_pool'] 
            for row in train_dataset
        }
        eval_pools = {row['sample_idx']: row['candidate_pool'] for row in eval_dataset}
        all_pools = {**train_pools, **eval_pools}
    else:
        all_pools = None
        print("NOT using use_candidate_pools")

    # search_reward_instance = SearchRewardFunction(
    #         faiss_index_path='/home/greenland-user/hnm_context_search/faiss_hnsw_index.bin',
    #         mapping_path='/home/greenland-user/hnm_context_search/hm_article_mapping.json',
    #         model_name='princeton-nlp/sup-simcse-roberta-large',
    #         device='cuda', top_k=script_args.faiss_topk, candidate_pools=all_pools, debug = False,  debug_max_items=2000
    #     )

    search_reward_instance = SearchRewardFunction(
            faiss_index_path="data/hnm/index/simcse_large_faiss.bin",
            mapping_path="data/hnm/index/simcse_large_article_mapping.json",
            model_name="princeton-nlp/sup-simcse-roberta-large",
            device="cuda:0",
            top_k=script_args.faiss_topk,
            debug=False,
            candidate_pools=all_pools, 
        )
    search_reward_instance.__name__ = 'search_ndcg_reward' 
       
    # semantic_reward_instance = SemanticSimilarityReward(
    #             model_name='princeton-nlp/sup-simcse-roberta-large',
    #             device='cuda',
    #             reward_weight=1.0,  # Configurable,  weighting is done inside OSPO trainer. 
    #             debug=False  # Set to True for detailed logging
    #         )
 

    def search_ndcg_reward(prompts, completions, target_item_id, **kwargs):
                return search_reward_instance(prompts, completions, target_item_id, **kwargs)

    def semantic_similarity_reward(prompts, completions, completion_ids, **kwargs):
        # completion_ids is ignored but needed for signature compatibility
        return semantic_reward_instance(prompts, completions, **kwargs)

    # train_dataset = train_dataset.select(range(2000))
    # eval_dataset = eval_dataset.select(range(500))

    trainer = SearchTrainer(
        model=model,
        reward_funcs=[search_reward_instance], #
        # reward_funcs=[search_reward_instance, semantic_reward_instance],
        # reward_processing_classes=[None],  
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

# CUDA_VISIBLE_DEVICES=1,4,6 accelerate launch --num_processes=3 train_grpo.py


# # ── 3B first (longest, run overnight) ────────────────────────────────────────
# CUDA_VISIBLE_DEVICES=0 accelerate launch train_grpo.py \
#     --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
#     --lora_model_name_or_path "sft_trained_models_hnm/qwen-3b/checkpoint-222" \
#     --output_dir outputs/grpo_3b_sft \
#     --dataset_path hnm_data_for_icml/hnm_rl_datasets_with_pools_no_item_context \
#     --learning_rate 5e-6 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 2 \
#     --max_prompt_length 356 \
#     --max_completion_length 1024 \
#     --max_steps 500 \
#     --eval_steps 100 \
#     --eval_strategy steps \
#     --faiss_topk 1000 \
#     --save_steps 100 \
#     --logging_steps 5 \
#     --num_generations 8 \
#     --kl_beta 0.1 \
#     --use_candidate_pools true \
#     --eval_data_sanity_check_size 200 \
 
