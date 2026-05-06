

# ==================== UTILITY FUNCTIONS ====================

import os
os.environ["HF_HUB_OFFLINE"] = "1"
import json
import re
import math
from typing import List, Dict, Any, Optional
from accelerate import logging

logger = logging.get_logger(__name__)

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
            
            #  Use ground truth length as k (minimum of 1)
            sample_k = max(1, len(gt_ranking))
            ndcg_score = score_ndcg_at_k(pred_ranking, gt_ranking, sample_k)
            rewards.append(ndcg_score)
            logger.debug(f"Sample {i}: GT length={len(gt_ranking)}, k={sample_k}, NDCG={ndcg_score:.4f}")
            
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

