import os
# os.environ["HF_HUB_OFFLINE"] = "1"
import re
import os, json, sys, traceback
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import gc
import time
import random
import logging
import pickle
import copy
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union
# Set NCCL timeout and monitoring before importing torch
os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '1800'  # 30 minutes
os.environ['TORCH_NCCL_ENABLE_MONITORING'] = '0'
os.environ['NCCL_DEBUG'] = 'INFO'
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from accelerate import logging
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from peft import LoraConfig, PeftConfig
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    HfArgumentParser, 
    set_seed, 
    BitsAndBytesConfig,
    Trainer
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
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from generation_utils import truncate_with_protected_tokens, nanmax, nanmin, nanstd, pad
from trl.models import prepare_deepspeed, prepare_fsdp, prepare_peft_model, unwrap_model_for_generation
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_liger_kernel_available, is_vllm_available
from trl.models.utils import _ForwardRedirection
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from reward_utils import ndcg_at_k_reward_function, valid_item_ids_reward_function
# from owen_utils import segment_completion_by_constituency, segment_completion_by_sentences, compute_owen_shap_rewards_mc
from ospo_utils import compute_search_owen_rewards 
# from dense_search.search import SearchRewardFunction, _unwrap_search_instance
from dense_search.search_with_logging import SearchRewardFunction, _unwrap_search_instance
from generation_utils import (
    disable_dropout_in_model,
    entropy_from_logits,
    generate_model_card,
    get_comet_experiment_url,
    identity,
    nanmax,
    nanmin,
    nanstd,
    pad,
    print_prompt_completions_sample,

)
logger = logging.get_logger(__name__)
from peft import PeftConfig, PeftModel
import wandb

@dataclass
class ScriptArguments:
    """
    The arguments for the OSPO training script.
    """
    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    # training parametersQwen/Qwen2.5-7B-Instruct
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen2.5-1.5B-Instruct",  
        metadata={"help": "the location of the SFT model name or path"},
    )
    lora_model_name_or_path: Optional[str] = field(
        default="sft_trained_models_hnm/qwen-1.5b/checkpoint-222",  
        metadata={"help": "the location of the SFT model name or path"},
    )


    output_dir: Optional[str] = field(default="Qwen/Qwen2.5-7B-Instruct", metadata={"help": "the output directory"})
    dataset_path: Optional[str] = field(default="hnm_data_for_icml/hnm_rl_datasets_with_pools_sampleidx_unique_2_generic", metadata={"help": "the path to the RL dataset"})
    eval_dataset_path: Optional[str] = field(default="hnm_data_for_icml/contextual_search_test_with_pools_itemcontext_recreated", metadata={"help": "the path to the RL dataset"})
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
    #Owen_shapley parameters; 
    owen_max_width: Optional[int] = field(default=32, metadata={"help": "maximum width of spans permitted for contiguous segments like phrases."})
    owen_max_permutations: Optional[int] = field(default=64, metadata={"help": "total permutations to be kept after coalition formation. "})
    owen_ablation_name: Optional[str] = field(default="coalitions_w2_m64", metadata={"help": "Contiguous coalitions: width=2, perms=64 (local, low variance)"})
    #search parameters
    faiss_topk: Optional[int] = field(default=1000, metadata={"help": "FAISS index retrieval top before reward computation in OSPO"})
    reward_topk: Optional[int] = field(default=100, metadata={"help": "Additional candidate filtering after FAISS retrieval for metrics"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    log_completions: Optional[bool] = field(default=True, metadata={"help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps."})
    save_steps: Optional[int] = field(default=500, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=200, metadata={"help": "the evaluation frequency"})
    eval_strategy: Optional[str] = field(default="steps", metadata={"help": "the evaluation strategy"})
    num_generations: Optional[int] = field(default=8, metadata={"help": "MCMC sampling for GRPO"})
    redistribution_mode: Optional[str] = field(default="owen_weights", metadata={"help": "Terminal advantage redistribution mode"})
    clip_ospo_advantages: Optional[bool] = field(default=False, metadata={"help": "Whether to clip redistributed advanges from OSPO"})
    use_candidate_pools: Optional[bool] = field(default=True, metadata={"help": "whether to use candidate pools in retrieval NDCG computation"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    torch_dtype: Optional[str] = field(default="bfloat16", metadata={"help": "torch_dtype[fp16, bfloat16, float] for loading."})
    train_data_sanity_check_size: Optional[int] = field(default=100, metadata={"help": "Smaller train sizes for test runs"})
    eval_data_sanity_check_size: Optional[int] = field(default=200, metadata={"help": "Smaller eval sizes for test runs"})
    # instrumentation
    report_to: Optional[str] = field(
        default="none",
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


class OSPOConfig(GRPOConfig):
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
        import transformers
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




# ==================== CUSTOM OSPO TRAINER ====================

class OwenShapleyTrainer(GRPOTrainer):
    """
    Custom RL trainer for chronological sequential ranking tasks.
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
        self.use_shapley_owen = True
        self.use_os_rewards_as_dense = self.args.use_os_rewards_as_dense
        self.use_os_rewards_as_search = self.args.use_os_rewards_as_search
        print("self.use_os_rewards_as_search", self.use_os_rewards_as_search)
        self.owen_max_width = self.args.owen_max_width
        self.owen_max_permutations = self.args.owen_max_permutations
     
        # Set default weights if not already set 
        if hasattr(self, '_default_reward_weights'):
            import torch
            self.reward_weights = torch.tensor(self._default_reward_weights)

    
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

    def clear_retrieval_logs(self):
        # Use self.RETRIEVAL_LOG_KEYS
        for k in self.RETRIEVAL_LOG_KEYS:
            lst = self._logs.get(k)
            if isinstance(lst, list):
                lst.clear()
        
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
                            "target_item": list(self._logs["target_item_id"]),
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
    
    def normalize_rewards_minmax_stable(self, rewards, global_min=-5.0, global_max=10.0):
        """
        Stable normalization with fixed bounds for consistent scaling across training
        """
        normalized = (rewards - global_min) / (global_max - global_min)
        return torch.clamp(normalized, 0.0, 1.0)

    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        reward_kwargs["trainer_state"] = self.state
 
        # STEP 1: Compute Owen-Shapley rewards ONCE before the loop
        token_level_owen_shap_rewards = None
        search_reward_func = None
        
        for func, name in zip(self.reward_funcs, self.reward_func_names):
            if name == 'search_ndcg_reward':
                search_reward_func = _unwrap_search_instance(func)
                break
        
        if search_reward_func is not None and self.model.training:
            try:
                token_level_owen_shap_rewards = compute_search_owen_rewards(
                    prompts=prompts,
                    completions=completions,
                    completion_ids_list=completion_ids_list,
                    search_reward_func=search_reward_func,
                    main_tokenizer=self.tokenizer,
                    max_permutations=self.owen_max_permutations,
                    device=device,
                    max_width = self.owen_max_width,
                    **reward_kwargs
                )

            except Exception as e:
                traceback.print_exc()
        
        # STEP 2: Now loop through ALL reward functions (both NN and non-NN)
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            print("running reward_func and reward_func_name in _calculate_rewards", reward_func, reward_func_name)
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):
                    # NN-based rewards
                    if self.use_os_rewards_as_dense:
                        token_level_owen_shap_rewards, debug_info = compute_owen_shap_rewards_mc(
                            prompts=prompts,
                            completions=completions,
                            completion_ids_list=completion_ids_list,
                            reward_func=reward_func,
                            reward_tokenizer=reward_processing_class,
                            main_tokenizer=self.tokenizer,
                            is_conversational=is_conversational(inputs[0]),
                            apply_chat_template=apply_chat_template,
                            prepare_inputs=lambda x: Trainer._prepare_inputs(self, x),
                            max_permutations=self.max_shapley_perms if hasattr(self, "max_shapley_perms") else 64,
                            device=device,
                        )
                        continue
                    else:
                        if is_conversational(inputs[0]):
                            messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                            texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                        else:
                            texts = [p + c for p, c in zip(prompts, completions)]
                        reward_inputs = reward_processing_class(
                            text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                        )
                        reward_inputs = Trainer._prepare_inputs(self, reward_inputs)
                        with torch.inference_mode():
                            output = reward_func(**reward_inputs)
                            raw_rewards = output.logits[:, 0]
                            normalized_rewards = self.normalize_rewards_minmax_stable(raw_rewards)
                            print(f"Normalized rewards range: [{torch.min(normalized_rewards):.4f}, {torch.max(normalized_rewards):.4f}]")
                            print(f"Normalized rewards: {normalized_rewards}")
                            rewards_per_func[:, i] = normalized_rewards
                            print(f"rewards_per_func shape after assignment: {rewards_per_func.shape}")
                
                else:
                    # Non-NN rewards (SearchRewardFunction, SemanticSimilarityReward, etc.)
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
    
         # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        # However, gathering token_level_owen_shap_rewards might be too expensive (added token dim) so we'd just use it locally later. 
        rewards_per_func = gather(rewards_per_func) #shape, B, G, N (total reward functions) HINT:  #shape, (B*num_gpus, N_reward_functions)
        
        # Handle Owen-Shapley rewards with different sequence lengths
        if 'token_level_owen_shap_rewards' in locals() and token_level_owen_shap_rewards is not None:
            # Get the maximum sequence length across all GPUs  
            local_seq_len = torch.tensor(token_level_owen_shap_rewards.size(1), device=device)
            all_seq_lens = self.accelerator.gather(local_seq_len)  # Gather seq lengths from all GPUs
            max_seq_len_global = all_seq_lens.max().item()
            
            # print(f"Local seq len: {local_seq_len.item()}, Global max seq len: {max_seq_len_global}")
            
            # Pad current tensor to global max length if needed: this is crucial to do here since later we'd apply
            # these token attributions to GRPO rewards to do PBRS, where we'd need zeros rightmost side tokens
            current_seq_len = token_level_owen_shap_rewards.size(1)
            if current_seq_len < max_seq_len_global:
                padding_size = max_seq_len_global - current_seq_len
                padding = torch.zeros(
                    token_level_owen_shap_rewards.size(0), 
                    padding_size, 
                    device=token_level_owen_shap_rewards.device,
                    dtype=token_level_owen_shap_rewards.dtype
                )
                token_level_owen_shap_rewards = torch.cat([token_level_owen_shap_rewards, padding], dim=1)
                # print(f"Padded Owen rewards from {current_seq_len} to {max_seq_len_global}")
        else:
            token_level_owen_shap_rewards = None
  
        return rewards_per_func, token_level_owen_shap_rewards 


    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # self._extract_and_store_metadata(inputs)
        device =  self.accelerator.device
        mode = "train" if self.model.training else "eval"
        # print("len inputs ", len(inputs))
        prompts = [x["prompt"] for x in inputs]
        # print("inuts on generate and score",inputs[0])
        original_prompts = copy.deepcopy(prompts)
        kwargs = {}
        has_images = "image" in inputs[0]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **kwargs,
        )
        # prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in prompt_inputs.items()}
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        if self.max_prompt_length is not None:
            # If max_prompt_length is set, we trim the prompt to keep only the last `max_prompt_length` tokens.
            # Then we decode those tokens back into text. We manually remove leading pad tokens from the decoded text,
            # because we can't use `skip_special_tokens=True` (some special tokens are still needed for generation).
            protected = [self.image_token_id, self.vision_start_token_id, self.vision_end_token_id]
            protected = [token for token in protected if token is not None]
            prompt_ids, prompt_mask = truncate_with_protected_tokens(
                prompt_ids, prompt_mask, self.max_prompt_length, protected
            )

            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            prompts_text = [re.sub(rf"^({re.escape(self.pad_token)})+", "", text) for text in prompts_text]
 
 
        with (
            profiling_context(self, "transformers.generate"),
            unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
            prompt_inputs["input_ids"], prompt_inputs["attention_mask"] = prompt_ids, prompt_mask
            prompt_completion_ids = unwrapped_model.generate(
                **prompt_inputs, generation_config=self.generation_config, disable_compile=True  
            )
        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [row[mask_row].tolist() for row, mask_row in zip(completion_ids, completion_mask.bool())]

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        num_items_in_batch = agg_completion_lengths.sum()  # this is required for the DAPO loss

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        image_split_sizes = None
        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction #this is the torch no grad for the denominator implementation
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    pixel_values=prompt_inputs.get("pixel_values"),
                    image_grid_thw=prompt_inputs.get("image_grid_thw"),
                    pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                    image_sizes=prompt_inputs.get("image_sizes"),
                    # image_split_sizes=None,
                )
            else:
                old_per_token_logps = None
            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        pixel_values=prompt_inputs.get("pixel_values"),
                        image_grid_thw=prompt_inputs.get("image_grid_thw"),
                        pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                        image_sizes=prompt_inputs.get("image_sizes"),
                        # image_split_sizes=None,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            pixel_values=prompt_inputs.get("pixel_values"),
                            image_grid_thw=prompt_inputs.get("image_grid_thw"),
                            pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                            image_sizes=prompt_inputs.get("image_sizes"),
                            # image_split_sizes=None,
                        )
            else:
                ref_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.

        # Without gather - WRONG normalization
        # Process 0 only sees rewards [r0, r1, r2] 
        # It would normalize: (r0 - mean([r0,r1,r2])) / std([r0,r1,r2])
            # With gather - CORRECT normalization  
        # All processes see rewards [r0, r1, r2, r3, r4, r5, r6, r7, r8]
        # They normalize: (r0 - mean(all_rewards)) / std(all_rewards)
        self.clear_retrieval_logs()
        for input_item in inputs:
            target = input_item.get('target_item_id', input_item.get('answer', ''))
            self._logs["target_item"].append(target)
        rewards_per_func, token_level_owen_shap_rewards = self._calculate_rewards(inputs, original_prompts, completions, completion_ids_list)
        if hasattr(self, 'reward_funcs'):
            for reward_func in self.reward_funcs:
                search_instance = _unwrap_search_instance(reward_func)
                if search_instance and hasattr(search_instance, 'last_retrievals'):
                    # optional: all_retrievals = gather_object(search_instance.last_retrievals)
                    if self.accelerator.is_main_process:
                        for r in search_instance.last_retrievals:
                            self._logs["retrieved_items"].append(r.get('retrieved_items', []))
                            self._logs["expanded_queries"].append(r.get('expanded_query', ''))
                            self._logs["retrieval_ranks"].append(r.get('rank', None))
                            self._logs["target_item"].append(r.get('target', ''))
                            # --- NEW, richer diagnostics ---
                            self._logs["pool_size"].append(r.get("pool_size", None))
                            self._logs["n_raw"].append(r.get("n_raw", None))
                            self._logs["n_filt"].append(r.get("n_filt", None))

                            self._logs["overlap_uniq"].append(r.get("overlap_uniq", None))
                            self._logs["overlap_rate"].append(r.get("overlap_rate", None))

                            self._logs["target_in_faiss_top1000"].append(
                                bool(r.get("target_in_faiss_top1000", False))
                            )
                            self._logs["first_hit_rank_raw"].append(r.get("first_hit_rank_raw", None))
                            self._logs["first_hit_rank_post"].append(r.get("first_hit_rank_post", None))
                            self._logs["zero_reward"].append(bool(r.get("zero_reward", False)))

                            # core metrics (post-filter, per-sample)
                            self._logs["ndcg"].append(r.get("ndcg", None))
                            self._logs["ap"].append(r.get("ap", None))
                            self._logs["mrr"].append(r.get("mrr", None))
                            self._logs["recall"].append(r.get("recall", None))

                            # @pool variants
                            self._logs["ndcg@pool"].append(r.get("ndcg@pool", None))
                            self._logs["recall@pool"].append(r.get("recall@pool", None))
                            self._logs["mrr@pool"].append(r.get("mrr@pool", None))
                            self._logs["ap@pool"].append(r.get("ap@pool", None))

                            # optional preview for quick eyeballing
                            self._logs["post_first10"].append(r.get("retrieved_items", [])[:10])


                            #add the faiss and other retrieval metrics for pooling and filtering





        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1) #this is where the standard terminal reward will be computed. 
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards in ["group", "none"]:
            # If self.scale_rewards = "none", we'll still log group level std
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            # Compute global std
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )
        
        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        max_seq_len = completion_ids.size(1)
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)
            # Apply same normalization to hybrid advantages
            std_rewards_expanded = std_rewards.unsqueeze(1).expand(-1, max_seq_len) # same STD value but expanded in token  length dimension
            # hybrid_token_advantages = hybrid_token_advantages / (std_rewards_expanded + 1e-4)

        
        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        
        # print("max_seq_len", max_seq_len)
        # Handle Owen alignment and redistribution ONLY during training
        if mode == "train" and token_level_owen_shap_rewards is not None:
            # Alignment logic
            if token_level_owen_shap_rewards.size(1) != max_seq_len:
                if token_level_owen_shap_rewards.size(1) > max_seq_len:
                    token_level_owen_shap_rewards = token_level_owen_shap_rewards[:, :max_seq_len]
                else:
                    padding_size = max_seq_len - token_level_owen_shap_rewards.size(1)
                    padding = torch.zeros(
                        token_level_owen_shap_rewards.size(0), 
                        padding_size, 
                        device=token_level_owen_shap_rewards.device
                    )
                    token_level_owen_shap_rewards = torch.cat([token_level_owen_shap_rewards, padding], dim=1)
            
            # Prepare completion mask
            completion_mask_local = completion_mask
            if completion_mask_local.size(1) != max_seq_len:
                if completion_mask_local.size(1) > max_seq_len:
                    mask_local = completion_mask_local[:, :max_seq_len]
                else:
                    padding = torch.zeros(
                        completion_mask_local.size(0),
                        max_seq_len - completion_mask_local.size(1),
                        device=completion_mask_local.device,
                        dtype=completion_mask_local.dtype
                    )
                    mask_local = torch.cat([completion_mask_local, padding], dim=1)
            else:
                mask_local = completion_mask_local
            mask_local = mask_local.to(token_level_owen_shap_rewards.dtype)
            
            # Slice local rewards
            rewards_local = rewards[process_slice]
            adv_local = advantages[process_slice]

        
            # Compute Owen advantages
            hybrid_token_advantages = self.compute_owen_shap_advantages(
               token_level_owen_shap_rewards,
                completion_mask=mask_local,
                rewards=rewards_local,
                advantages=adv_local,
                redistribution_mode=self.args.redistribution_mode,
         
                eps=1e-4,
            )
            
            # Dynamic clipping
            if self.args.clip_ospo_advantages:
                # Dynamic clipping
                grpo_std = advantages.std()
                clip_multiplier = 2.0
                clip_bound = grpo_std * clip_multiplier
                
                # Store original for logging
                original_max = hybrid_token_advantages.abs().max().item()
                
                # Clip
                hybrid_token_advantages = torch.clamp(
                    hybrid_token_advantages, 
                    min=-clip_bound, 
                    max=clip_bound
                )
                
                # Clipping statistics
                clipped_count = ((hybrid_token_advantages.abs() >= clip_bound * 0.99).sum().item())
                if clipped_count > 0:
                    print(f"Clipped {clipped_count} extreme advantages (max was {original_max:.2f}, bound is {clip_bound:.2f})")

            # Store for logging
            self._last_redistributed_advantages = hybrid_token_advantages.clone()
            
            # Use Owen advantages
            all_process_advantages = advantages.clone()
            original_advantages = advantages[process_slice]
            advantages = hybrid_token_advantages
            
        else:
            # Eval mode: use standard GRPO advantages (no Owen redistribution)
            all_process_advantages = advantages.clone()
            original_advantages = advantages  # No process slicing needed
            # Expand sequence-level advantages to token-level for compute_loss
            # Standard GRPO: same advantage value repeated across all tokens
            advantages_expanded = advantages[process_slice].unsqueeze(1).expand(-1, max_seq_len)
            advantages = advantages_expanded

            # advantages stays as-is (sequence-level GRPO advantages)

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
    
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())
        self._metrics[mode]["advantages_grpo/mean"].append(all_process_advantages.mean().item())
        self._metrics[mode]["advantages_grpo/std"].append(all_process_advantages.std().item())
        self._metrics[mode]["advantages_grpo/min"].append(all_process_advantages.min().item())
        self._metrics[mode]["advantages_grpo/max"].append(all_process_advantages.max().item())
        self._metrics[mode]["advantages_grpo/abs_mean"].append(all_process_advantages.abs().mean().item())

        #log the
        # Add redistributed advantage logging:
        if mode == "train" and hasattr(self, '_last_redistributed_advantages'):
            # Get valid token advantages only (exclude padding)
            mask_to_use = mask_local if 'mask_local' in locals() else completion_mask
            redist_valid = self._last_redistributed_advantages[completion_mask > 0.5]
            self._metrics[mode]["advantages_redistributed/mean"].append(redist_valid.mean().item())
            self._metrics[mode]["advantages_redistributed/std"].append(redist_valid.std().item())
            self._metrics[mode]["advantages_redistributed/min"].append(redist_valid.min().item())
            self._metrics[mode]["advantages_redistributed/max"].append(redist_valid.max().item())
            self._metrics[mode]["advantages_redistributed/abs_mean"].append(redist_valid.abs().mean().item())


        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())


        def _safe_mean(seq):
            seq = [x for x in seq if x is not None]
            return float(sum(seq) / len(seq)) if seq else float('nan')

        def _safe_rate(bools):
            vals = [1.0 if bool(x) else 0.0 for x in bools]
            return _safe_mean(vals)
        #get aggregated metrics from retrieval logs.
        L = self._logs
        M = self._metrics[mode]
        M.setdefault("retrieval/targets_in_faiss_top1000_rate", []).append(
            _safe_rate(L.get("target_in_faiss_top1000", []))
        )

        # % where target was present in the candidate pool used for filtering
        # (log this as a bool per-sample when you create last_retrievals)
       
        M.setdefault("retrieval/target_in_pool_rate", []).append(
            _safe_rate(L.get("target_in_pool", [])) if "target_in_pool" in L else float('nan'))

        # 2) Effectiveness (post-filter)
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

        # First-hit rank (post-filter): store mean and a robust percentile view
        post_ranks = L.get("first_hit_rank_post", [])
        post_ranks = [r for r in post_ranks if r is not None]  # strip misses
        if post_ranks:
            post_ranks_sorted = sorted(post_ranks)
            p50 = post_ranks_sorted[len(post_ranks_sorted) // 2]
            p90 = post_ranks_sorted[int(0.9 * (len(post_ranks_sorted) - 1))]
       
        else:
            M.setdefault("retrieval/first_hit_rank_post_mean", []).append(float('nan'))
 
        # 3) Stability / health
        # zero_reward is a per-sample bool you can log when reward == 0.0
        M.setdefault("retrieval/zero_reward_rate", []).append(
            _safe_rate(L.get("zero_reward", []))
        )
        # If you want global means of the raw@1000 vs post@pool NDCG per batch:
        if "ndcg_raw@1000" in L and "ndcg@pool" in L:
            M.setdefault("retrieval/ndcg_raw1000_mean", []).append(_safe_mean(L["ndcg_raw@1000"]))
            M.setdefault("retrieval/ndcg_delta_post_minus_raw", []).append(
                _safe_mean(L["ndcg@pool"]) - _safe_mean(L["ndcg_raw@1000"])
            )

        # final return output after logging is done. 
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }

        if mode == "train":
            output["hybrid_token_advantages"] = hybrid_token_advantages
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps

 
        return output

    def compute_owen_shap_advantages(
        self,
        owen_values: torch.Tensor,  # (B, T) - raw Owen-Shapley token attributions [0.1, 0.9]
        completion_mask: torch.Tensor,  # (B, T)
        rewards: torch.Tensor,  # (B,) - terminal GRPO rewards
        advantages: torch.Tensor,  # (B,) - sequence-level GRPO advantages
        *,
        redistribution_mode: str = "owen_weights",  # "owen_weights", "rank_based", "hybrid_alpha"
        alpha: float = 0.7,  # For hybrid mode only
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        Redistribute sequence-level advantages to token-level using Owen-Shapley credits.
        
        Args:
            owen_values: Token attributions from Owen-Shapley, shape (B, T)
            completion_mask: Binary mask for valid tokens, shape (B, T)
            rewards: Terminal rewards per sequence, shape (B,)
            advantages: GRPO advantages per sequence, shape (B,)
            strategy: Redistribution strategy
            alpha: Interpolation weight for hybrid mode (0=uniform, 1=full Owen)
            eps: Small constant for numerical stability
            
        Returns:
            Token-level advantages, shape (B, T)
        """
        B, T = owen_values.shape
        device = owen_values.device
        mask = completion_mask.to(owen_values.dtype)
        num_tokens = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        # Step 1: PBRS - Scale Owen values to sum to terminal rewards
        # This ensures additivity: sum of token credits = total reward
        owen_sums = (owen_values * mask).sum(dim=1, keepdim=True).clamp_min(eps)
        pbrs_owen = owen_values * (rewards.unsqueeze(1) / owen_sums)
        
        if redistribution_mode == "owen_weights":
            # Use Owen values directly as proportional weights
            # Guarantees: sum(redistributed) = advantages, respects Owen credits
            # Normalizing to proportions decouples from the reward scale: so, we need owen_props from pbrs_owen
            # Since rewards and advantages are in different scaled. Once owens are normalized, then we can apply to any advantage scale
            owen_props = pbrs_owen / (pbrs_owen * mask).sum(dim=1, keepdim=True).clamp_min(eps)
            #why multiply by num_tokens? To reduce bias towards longer responses. 
            redistributed = advantages.unsqueeze(1) * owen_props * num_tokens
            
        elif redistribution_mode == "rank_based":
            # Convert Owen credits to ranks to avoid scale sensitivity
            # Higher credit → higher rank → more advantage
            owen_ranks = torch.argsort(torch.argsort(pbrs_owen, dim=1, descending=True), dim=1).float()
            owen_ranks = owen_ranks * mask
            # Normalize ranks to proportions
            rank_sum = owen_ranks.sum(dim=1, keepdim=True).clamp_min(eps)
            rank_props = owen_ranks / rank_sum
            # Redistribute advantages
            redistributed = advantages.unsqueeze(1) * rank_props
            #Scale back up by sequence length
            redistributed = redistributed * num_tokens

        elif redistribution_mode == "hybrid_alpha":
            # Blend between uniform (safe) and Owen-based (informative)
            # alpha controls trust in Owen values
            uniform_props = mask / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            owen_props = pbrs_owen / (pbrs_owen * mask).sum(dim=1, keepdim=True).clamp_min(eps)
            
            final_props = alpha * owen_props + (1 - alpha) * uniform_props
            redistributed = advantages.unsqueeze(1) * final_props

        # elif redistribution_mode = "direct_owen":
        #     redistributed = advantages.unsqueeze(1) * pbrs_owen * num_token
        else:
            raise ValueError(f"Unknown strategy: {redistribution_mode}. Choose from: owen_weights, rank_based, hybrid_alpha")
        
        # Verification
        redist_sum = (redistributed * mask).sum(dim=1)
        conservation_error = (redist_sum - advantages).abs().max().item()
        valid_redist = redistributed[mask > 0.5]
        return redistributed
        
    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.use_liger_loss:
            # Compute the loss using the liger grpo loss
            unwrapped_model = self.accelerator.unwrap_model(model)
            return self._forward_redirection(model, unwrapped_model, self.compute_liger_loss, unwrapped_model, inputs)
        else:
            return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0 
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            # image_split_sizes=inputs.get("image_split_sizes"),
        )
        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else: 
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )
        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)
      
        assert advantages.shape == coef_1.shape, f"Shape mismatch: advantages {advantages.shape} vs coef_1 {coef_1.shape}"
        
        # per_token_loss1 = coef_1 * advantages.unsqueeze(1) #these are B, G (now unsqueeze is not needed since already token level)
      
        per_token_loss1 = coef_1 * advantages  # (B,T) * (B,T) = (B,T) instsead of B, T * B in original GRPO. 
        per_token_loss2 = coef_2 * advantages  # (B,T) * (B,T) = (B,T)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None


def load_data(dataset_path):
    dataset = DatasetDict.load_from_disk(dataset_path)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    print("train_dataset", len(train_dataset))
    print("eval_dataset", len(eval_dataset))
    return train_dataset, eval_dataset



def create_ablation_training_config(base_script_args, experiment_config, experiment_name):
    """
    Create training configuration for a specific ablation experiment.
    
    Args:
        base_script_args: Base script arguments from HfArgumentParser
        experiment_config: Dictionary with ablation-specific boolean flags
        experiment_name: String identifier for the experiment
        
    Returns:
        OSPOConfig: Training configuration with ablation flags set
    """
    training_args = OSPOConfig(
        output_dir=f"Qwen/Qwen2.5-7B-Instruct-OSPO-hnm-search-{experiment_name}_{base_script_args.output_dir}_3", 
        per_device_train_batch_size=base_script_args.per_device_train_batch_size,
        per_device_eval_batch_size=base_script_args.per_device_eval_batch_size,
        gradient_accumulation_steps=base_script_args.gradient_accumulation_steps,
        num_generations=base_script_args.num_generations,
        logging_steps=base_script_args.logging_steps,
        learning_rate=base_script_args.learning_rate,
        gradient_checkpointing=base_script_args.gradient_checkpointing,
        ddp_find_unused_parameters=False,  
        bf16=True,
        fp16=False,
        # GRPO specific parameters
        generation_batch_size=64, 
        steps_per_generation=4,    
        max_prompt_length=base_script_args.max_prompt_length,
        max_completion_length=base_script_args.max_completion_length,
        beta=base_script_args.kl_beta,
        remove_unused_columns=False,
        eval_steps=base_script_args.eval_steps, 
        eval_strategy=base_script_args.eval_strategy,
        max_steps=base_script_args.max_steps,
        save_steps=base_script_args.save_steps, 
        # reward_weights=[0.9, 0.1]  #testing with sem sim reward from SemanticSimilarityReward; can remove. 
    )
    
    # Set baseline Owen-Shapley  
    training_args.use_shapley_owen = True
    training_args.use_os_rewards_as_search = True
    training_args.use_os_rewards_as_dense = False
    # Set redistribution mode for the ablation
    training_args.redistribution_mode = experiment_config["redistribution_mode"]
    # Add experiment metadata for logging and identification
    training_args.experiment_name = experiment_name
    training_args.clip_ospo_advantages = base_script_args.clip_ospo_advantages
    training_args.owen_max_width = base_script_args.owen_max_width
    training_args.owen_max_permutations = base_script_args.owen_max_permutations
    print("running experimnent with owen_max_width", training_args.owen_max_width)
    print("running experimnent with owen_max_permutations", training_args.owen_max_permutations)
    return training_args

def run_ablation_experiments(script_args, train_dataset, eval_dataset, 
                           actual_torch_dtype, quantization_config, peft_config):
    """
    Execute all ablation experiments sequentially.
    
    Args:
        script_args: Parsed script arguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        actual_torch_dtype: Torch dtype for model loading
        quantization_config: Model quantization configuration
        peft_config: PEFT configuration for parameter-efficient fine-tuning
        
    Returns:
        Dict: Summary of results from all experiments
    """

    ABLATION_EXPERIMENTS = [
    {
        "name": "owen_weights",
        "description": "Owen-Shapley proportional redistribution with advantage conservation",
        "config": {
            "redistribution_mode": "owen_weights"
        }
    },
    {
        "name": "rank_based", 
        "description": "Rank-based redistribution with sequence length scaling",
        "config": {
            "redistribution_mode": "rank_based"
        }
    },
    {
        "name": "hybrid_alpha",
        "description": "Hybrid Owen-uniform blend (alpha=0.7)",
        "config": {
            "redistribution_mode": "hybrid_alpha"
        }
    },
]

    def search_ndcg_reward(prompts, completions, target_item_id, **kwargs):
        return search_reward_instance(prompts, completions, target_item_id, **kwargs)

    results_summary = {}
    # Define reward functions for all experiments
    # reward_funcs=[search_ndcg_reward]
    current_redistribution_mode = script_args.redistribution_mode
    ABLATION_EXPERIMENTS = [
        exp for exp in ABLATION_EXPERIMENTS 
        if exp['config']['redistribution_mode'] == current_redistribution_mode]

    if not ABLATION_EXPERIMENTS:
        print(f"No experiment found for redistribution_mode: {current_redistribution_mode}")
        return {}

    print(f"Running ablation for: {current_redistribution_mode}")
    for experiment in ABLATION_EXPERIMENTS:

        exp_name = experiment["name"]
        exp_config = experiment["config"]
        exp_description = experiment["description"]
        
        print(f"\n{'-'*60}")
        print(f"Starting experiment: {exp_name}")
        print(f"Description: {exp_description}")
        print(f"Configuration: {exp_config}")
        print(f"Redistribution mode: {exp_config['redistribution_mode']}")
        print(f"{'-'*60}")
        
        try:
            print(f"\n{'='*60}")
            print(f"Environment check for experiment: {exp_name}")
            print(f"  CUDA memory allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            print(f"  Process group initialized: {torch.distributed.is_initialized()}")
            if torch.distributed.is_initialized():
                print(f"  World size: {torch.distributed.get_world_size()}")
            print(f"{'='*60}\n")
            # Load fresh model for each experiment to ensure clean state
            # model = AutoModelForCausalLM.from_pretrained(
            #     script_args.model_name_or_path,  
            #     torch_dtype=actual_torch_dtype,
            #     quantization_config=quantization_config,   
            #     device_map=get_kbit_device_map(),          
            #     trust_remote_code=True,  
            # )
            model = AutoModelForCausalLM.from_pretrained(
                script_args.model_name_or_path,  
                dtype=actual_torch_dtype,
                quantization_config=quantization_config,   
                device_map=get_kbit_device_map(),
                trust_remote_code=True,  
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
            
            # Create experiment-specific training configuration
            training_args = create_ablation_training_config(script_args, exp_config, exp_name)
            
            print(f"Ablation flags for {exp_name}:")
            # print(f"  use_pbrs: {training_args.use_pbrs}")
            print(f"  redistribution_mode: {training_args.redistribution_mode}")
            print(f"  experiment_name: {training_args.experiment_name}")
            print(f"  output_dir: {training_args.output_dir}")

            # Add this RIGHT BEFORE creating the trainer
            print(f"\nPre-trainer checks for {exp_name}:")
            print(f"  Output dir: {training_args.output_dir}")
            print(f"  Output dir exists: {os.path.exists(training_args.output_dir)}")
            print(f"  Contents: {os.listdir(training_args.output_dir) if os.path.exists(training_args.output_dir) else 'N/A'}")
            print(f"  save_steps: {training_args.save_steps}")
            print(f"  resume_from_checkpoint: {training_args.resume_from_checkpoint}")

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

        #     search_reward_instance = SearchRewardFunction(
        #     faiss_index_path='/home/greenland-user/hnm_context_search/faiss_hnsw_index.bin',
        #     mapping_path='/home/greenland-user/hnm_context_search/hm_article_mapping.json',
        #     model_name='princeton-nlp/sup-simcse-roberta-large',
        #     device='cuda', top_k=script_args.faiss_topk, candidate_pools=all_pools, debug = False,  debug_max_items=2000
        # )


            # hnm_dataset_path      = "hnm_data_for_icml/hnm_rl_datasets_with_pools_sampleidx_unique_2_generic"
            # hm_item_metadata_path = "data/hnm/hm_item_metadata.jsonl"

            # print("Loading metadata...")
            # with open(hm_item_metadata_path) as f:
            #     hm_item_metadata = [json.loads(l) for l in f if l.strip()]

            search_reward_instance = SearchRewardFunction(
                faiss_index_path="data/hnm/index/simcse_large_faiss.bin",
                mapping_path="data/hnm/index/simcse_large_article_mapping.json",
                model_name="princeton-nlp/sup-simcse-roberta-large",
                device="cuda:0",
                top_k=1000,
                debug=False,
                candidate_pools=all_pools, 
            )
            
            search_reward_instance.validate_catalog_alignment(n=2000)
            print("search_reward_instance", search_reward_instance)

            def search_ndcg_reward(prompts, completions, target_item_id, **kwargs):
                return search_reward_instance(prompts, completions, target_item_id, **kwargs)

            def semantic_similarity_reward(prompts, completions, completion_ids, **kwargs):
                # completion_ids is ignored but needed for signature compatibility
                return semantic_reward_instance(prompts, completions, **kwargs)

            trainer = OwenShapleyTrainer(
                model=model,
                reward_funcs=[search_ndcg_reward], #can add the semantic sim reward here as another name in the list. 
                # reward_funcs=[search_ndcg_reward, semantic_similarity_reward],
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                peft_config=peft_config,
            )
            print(f"Post-trainer init for {exp_name}")
            print(f"  Trainer save_steps: {trainer.args.save_steps}")
            print(f"  Trainer global_step at start: {trainer.state.global_step}")
            
            
            # Execute training
            print(f"Beginning training for experiment: {exp_name}")
            train_result = trainer.train()
            
            # Collect final metrics from training
            final_metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
            results_summary[exp_name] = {
                "description": exp_description,
                "config": exp_config,
                "final_metrics": final_metrics,
                "train_result": train_result,
                "status": "completed"
            }
            
            print(f"Successfully completed experiment: {exp_name}")
            
        except Exception as e:
            print(f"Experiment {exp_name} failed with error: {str(e)}")
            logger.error(f"Training failed for {exp_name}: {e}")
            
            results_summary[exp_name] = {
                "description": exp_description, 
                "config": exp_config,
                "error": str(e),
                "status": "failed"
            }
        
        finally:
            # Clean up resources to prevent memory accumulation
            if 'trainer' in locals():
                del trainer
            if 'model' in locals():
                del model

            # Force reset distributed backend
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
                
                import time
                time.sleep(5) 
            torch.cuda.empty_cache()
            gc.collect()
            
        print(f"Completed processing for experiment: {exp_name}")
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("ABLATION EXPERIMENTS SUMMARY")
    print(f"{'='*60}")
    
    successful_experiments = 0
    failed_experiments = 0
    
    for exp_name, results in results_summary.items():
        status_indicator = "SUCCESS" if results["status"] == "completed" else "FAILED"
        print(f"{exp_name}: {status_indicator}")
        
        if results["status"] == "completed":
            successful_experiments += 1
        else:
            failed_experiments += 1
            print(f"  Error: {results.get('error', 'Unknown error')}")
        
    
    print(f"\nTotal experiments: {len(ABLATION_EXPERIMENTS)}")
    print(f"Successful: {successful_experiments}")
    print(f"Failed: {failed_experiments}")
    
    return results_summary

# def search_ndcg_reward(prompts, completions, target_item_id, **kwargs):
#     return search_reward_instance(prompts, completions, target_item_id, **kwargs)

# def semantic_similarity_reward(prompts, completions, target_item_id, **kwargs):
#     return semantic_reward_instance(prompts, completions, target_item_id, **kwargs)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
    
    # Load and prepare datasets
    train_dataset, _ = load_data(script_args.dataset_path) 
    eval_dataset = load_from_disk(script_args.eval_dataset_path) 
 
    print("Train dataset loaded:", train_dataset) 
    print("Eval dataset loaded:", eval_dataset)
    set_seed(script_args.seed)
    
    # Determine torch dtype for model loading
    if script_args.torch_dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        actual_torch_dtype = torch.bfloat16
    elif script_args.torch_dtype == "float16":
        actual_torch_dtype = torch.float16
    else:
        actual_torch_dtype = torch.float32
  
    # Configure quantization settings
    quantization_config = custom_get_quantization_config(script_args)
    print(f"Quantization configuration: {quantization_config}")

    # Configure PEFT settings
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
    peft_config = get_peft_config(model_config_for_peft)
    print(f"PEFT configuration: {peft_config}")
    # train_dataset = train_dataset.select(range(script_args.train_data_sanity_check_size))
    eval_dataset = eval_dataset.select(range(script_args.eval_data_sanity_check_size))

    
    print("final len train with train_data_sanity_check_size", train_dataset)
    print("final len eval_dataset with eval_data_sanity_check_size ", eval_dataset)
    # Execute all ablation experiments
    print("Starting ablation experiment suite...")
    redistribution_mode = script_args.redistribution_mode  
    results = run_ablation_experiments(
        script_args, train_dataset, eval_dataset, 
        actual_torch_dtype, quantization_config, peft_config
    )
    
    print("Ablation experiments completed.")
    print(f"Results summary available with {len(results)} experiments.")

    # # Plot the results
    # print("\nGenerating comparison plots...")
    # plot_ablation_results(results_dir="./cleaned_files/Qwen")


    # CUDA_VISIBLE_DEVICES=0 accelerate launch train_ospo.py \
    # --model_name_or_path "Qwen/Qwen2.5-1.5B-Instruct" \
    # --output_dir outputs/ospo_prop_no_clip_with_sft \
    # --dataset_path hnm_data_for_icml/hnm_rl_datasets_with_pools_sampleidx_unique_2_generic \
    # --learning_rate 5e-6 \
    # --per_device_train_batch_size 8 \
    # --per_device_eval_batch_size 8 \
    # --gradient_accumulation_steps 2 \
    # --max_prompt_length 356 \
    # --max_completion_length 1024 \
    # --max_steps 1000 \
    # --eval_steps 100 \
    # --eval_strategy steps \
    # --owen_max_width 8 \
    # --owen_max_permutations 96 \
    # --redistribution_mode owen_weights \ 
    # --clip_ospo_advantages False \
    # --faiss_topk 1000 \
    # --save_steps 200 \
    # --logging_steps 5 \
    # --num_generations 8 \
    # --kl_beta 0.0 \
    # --use_candidate_pools true 


# ── 3B first (longest, run overnight) ────────────────────────────────────────
# CUDA_VISIBLE_DEVICES=0 accelerate launch train_ospo.py \
#     --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
#     --lora_model_name_or_path "sft_trained_models_hnm/qwen-3b/checkpoint-222" \
#     --output_dir outputs/ospo_prop_clipped_3b_sft \
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
#     --owen_max_width 8 \
#     --owen_max_permutations 96 \
#     --redistribution_mode owen_weights \
#     --clip_ospo_advantages True \
#     --faiss_topk 1000 \
#     --save_steps 100 \
#     --logging_steps 5 \
#     --num_generations 8 \
#     --kl_beta 0.1 \
#     --use_candidate_pools true