# Code released for anonymous review. License: CC-BY-NC-4.0  


import torch
import torch.nn.functional as F
import numpy as np
import math
import random
import logging
import re
import nltk
import spacy
from typing import List, Dict, Tuple, Optional, Any, Callable
from itertools import combinations
from dense_search.search import SearchRewardFunction
from transformers import PreTrainedTokenizer
logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)
try:
    nltk.download('punkt_tab', quiet=True)
except:
    pass  # Already downloaded or use fallback
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

nlp = spacy.load("en_core_web_sm")
def break_query_into_phrases(texts, batch_size=64):
    """
    Return phrases in ORIGINAL TEXT ORDER.
    Each phrase is collected with (start_char, end_char, text), sorted by start.
    Overlap suppression prefers longer spans and checks against ALL kept spans.
    """
    docs = list(nlp.pipe(texts, batch_size=batch_size))
    out_phrases = []

    for doc in docs:
        spans = []

        # 1) Noun phrases
        for chunk in doc.noun_chunks:
            words = chunk.text.strip().split()
            if 2 <= len(words) <= 5:
                spans.append((chunk.start_char, chunk.end_char, chunk.text))

        # 2) Adjective–noun pairs
        for tok in doc:
            if tok.pos_ == "ADJ" and tok.head.pos_ == "NOUN" and tok.head.i > tok.i:
                span = doc[tok.i: tok.head.i + 1]
                words = span.text.strip().split()
                if 2 <= len(words) <= 5:
                    spans.append((span.start_char, span.end_char, span.text))

        # 3) Verb-led preference phrases
        preference_verbs = {'looking', 'need', 'want', 'prefer', 'replace', 'update', 'wear', 'pair'}
        for tok in doc:
            if tok.lemma_ in preference_verbs:
                end = min(tok.i + 6, len(doc))
                span = doc[tok.i:end]
                words = span.text.strip().split()
                if 3 <= len(words) <= 6:
                    spans.append((span.start_char, span.end_char, span.text))

        # Sort by start
        spans.sort(key=lambda x: x[0])

        # ---- FULL non-max suppression across ALL kept spans ----
        kept = []
        for sL, sR, sText in spans:
            keep_me = True
            for idx, (kL, kR, kText) in enumerate(kept):
                # overlap if not disjoint
                if not (sL >= kR or sR <= kL):
                    # prefer longer
                    if (sR - sL) > (kR - kL):
                        kept[idx] = (sL, sR, sText)  # replace shorter kept
                    keep_me = False
                    break
            if keep_me:
                kept.append((sL, sR, sText))

        # De-dupe exact strings but retain first occurrence (order preserved)
        seen = set()
        ordered_phrases = []
        for _, _, t in kept:
            lt = t.strip().lower()
            if lt not in seen:
                seen.add(lt)
                ordered_phrases.append(t.strip())

        out_phrases.append(ordered_phrases)

    return out_phrases

def compute_search_owen_rewards(
    prompts: List[str],
    completions: List[str],
    completion_ids_list: List[List[int]],
    search_reward_func: "SearchRewardFunction",
    main_tokenizer: PreTrainedTokenizer,
    max_permutations: int = 32,
    device: Optional[torch.device] = None,
    max_width: int = 8,
    **kwargs
) -> torch.Tensor:
    """
    Compute token-level attributions using Owen-Shapley values for search query generation.
    
    This function decomposes LLM-generated search queries into linguistic phrases, evaluates
    coalitions of phrases using search rewards (e.g., NDCG), computes phrase-level Owen values
    based on marginal contributions, and maps these attributions back to individual tokens.
    
    Args:
        prompts: List of input prompts (not directly used in current implementation)
        completions: List of LLM-generated completions containing search queries
        completion_ids_list: List of token ID sequences for each completion
        search_reward_func: Search reward function that evaluates query quality (e.g., via FAISS retrieval)
        main_tokenizer: Tokenizer for mapping between text and tokens
        max_permutations: Maximum number of coalition permutations to evaluate (sampled if exceeded)
        device: PyTorch device for tensor operations
        max_width: Maximum width of contiguous phrase coalitions to generate
        **kwargs: Additional arguments including:
            - sample_idx: Global indices for candidate pool tracking in reward computation
            - answer: Ground truth target items (list of lists for multiple matches)
            - target_item_id: Fallback single target item per sample
    
    Returns:
        torch.Tensor: Padded tensor of shape (batch_size, max_seq_len) containing normalized
                     token-level attribution scores in [0, 1]. Tokens from queries with single
                     phrases receive uniform attribution of 1/num_tokens.
    
    Process:
        1. Extract queries from completions (text within <answer> tags)
        2. Break queries into linguistic phrases using spaCy
        3. Generate contiguous phrase coalitions up to max_width
        4. Evaluate search rewards for each coalition using the reward function
        5. Compute Owen values for phrases based on marginal contributions
        6. Map phrase attributions to tokens using character-overlap weighting
        7. Return padded Owen values after min-max
        8. Pad sequences to uniform length and return as tensor with padding. 
    """
    
    global_indices = kwargs.get('sample_idx', None)  # List: needed for candidate pool tracking for reward computation (NDCG)
    # Step 1: extract queries and then break queries into linguistic phrases using spacy pipe. 
    completion_texts = [_extract_query(c) for c in completions]
    batch_phrases = break_query_into_phrases(completion_texts, batch_size = 50) #only the query generated by the LLM within the <answer> tokens take part in the Owen-value comptations. 
 
    # Step 2: Loop for coalition generation and evaluation 
    batch_token_attributions = []
    for i, (completion_text, phrases) in enumerate(zip(completion_texts, batch_phrases)):
        if len(phrases) <= 1:
            enc = main_tokenizer(completion_text, add_special_tokens=False)
            num_tokens = len(enc["input_ids"])
            batch_token_attributions.append(torch.ones(num_tokens, device=device) / num_tokens)

            continue

        coalitions = generate_coalition_combinations(len(phrases), max_width=max_width, include_empty=True)
        
        if len(coalitions) > max_permutations:
            # Keep empty and full, sample the rest
            sampled = [coalitions[0], coalitions[-1]]
            mid = coalitions[1:-1]
            np.random.shuffle(mid)
            sampled.extend(mid[:max_permutations-2])
            coalitions = sampled
        
        sample_idx = global_indices[i] if global_indices is not None else None

        target_item = None
        if "answer" in kwargs and i < len(kwargs["answer"]): # ESCI data contains multiple exact matches, so our preference is to try to use all ground-truth items. 
            target_item = kwargs["answer"][i]  # List of all ground truth items
        elif "target_item_id" in kwargs and i < len(kwargs["target_item_id"]):
            target_item = kwargs["target_item_id"][i]  # Fallback to single
        coalition_rewards = evaluate_search_coalition_rewards(completion_text,
            phrases, coalitions, search_reward_func,
            target_item=target_item, device = device, sample_idx = sample_idx
        )
        # Coalition evaluation for getting marginal contribution of indivudual phrases. 
        phrase_attributions = compute_sentence_owen_values(coalition_rewards, coalitions, len(phrases))
        phrase_attributions = phrase_attributions.to(device)
        # Map to tokens
        W, token_ids = phrase_token_mapping(completion_text, phrases, main_tokenizer, device=device)
        token_attributions = W.T @ phrase_attributions
        # Apply min-max normalization
        min_attr = token_attributions.min()
        max_attr = token_attributions.max()

        if max_attr > min_attr:
            normalized = (token_attributions - min_attr) / (max_attr - min_attr)
            token_attributions = normalized 
        else:
            token_attributions = torch.ones_like(token_attributions, device=device) * 0.5

        batch_token_attributions.append(token_attributions)
    
    if not batch_token_attributions:
        return torch.zeros(len(completions), 1, device=device)  # Fallback
    # Pad and return
    max_len = max(t.size(0) for t in batch_token_attributions)
 
    return torch.stack([F.pad(t, (0, max_len - t.size(0))) for t in batch_token_attributions]).to(device)

def phrase_token_mapping(
    completion_text: str,
    phrases: List[str],
    main_tokenizer, device = None
) -> Tuple[torch.Tensor, List[int]]:
    """
    Map phrases to tokens using word overlap.
    Returns W: (P, T) phrase->token mapping where P=num phrases, T=num tokens
    """
    # Tokenize full completion
    enc = main_tokenizer(completion_text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = enc["input_ids"]
    token_offsets = enc["offset_mapping"]
    
    P, T = len(phrases), len(token_ids)
    W = torch.zeros(P, T, dtype=torch.float32, device =device)
    
    # For each phrase, find which tokens it spans
    for p_idx, phrase in enumerate(phrases):
        # Find phrase in completion text
        phrase_start = completion_text.lower().find(phrase.lower())
        if phrase_start == -1:
            continue
        phrase_end = phrase_start + len(phrase)
        
        # Find overlapping tokens
        for t_idx, (t_start, t_end) in enumerate(token_offsets):
            if t_end <= t_start:
                continue
            # Calculate character overlap
            overlap = max(0, min(phrase_end, t_end) - max(phrase_start, t_start))
            if overlap > 0:
                W[p_idx, t_idx] = overlap / (t_end - t_start)  # Fractional overlap
    
    # Normalize per token (each token assigned to one phrase)
    col_sums = W.sum(dim=0, keepdim=True)
    W = W / torch.clamp(col_sums, min=1e-8)
    
    return W, token_ids

def coalition_text_from_spans(doc_text, phrase_spans, coalition_idxs):
    """
    phrase_spans: [(start_char, end_char, text), ...] sorted by start
    coalition_idxs: list of indices (contiguous in the phrase list)
    Returns the contiguous slice from the min start to max end, preserving intervening tokens.
    """
    if not coalition_idxs:
        return ""
    sL = min(phrase_spans[i][0] for i in coalition_idxs)
    sR = max(phrase_spans[i][1] for i in coalition_idxs)
    return doc_text[sL:sR].strip()


def evaluate_search_coalition_rewards(completion_text,
    phrases: List[str],
    coalitions: List[List[int]],
    search_reward_func: "SearchRewardFunction",
    target_item: str, device = None, sample_idx = None
) -> torch.Tensor:
    """Evaluate search rewards for phrase coalitions with debug prints."""

    coalition_rewards = []
    for coalition in coalitions:
        if len(coalition) == 0:
            reward = 0.0
            # print(f"Coalition [] -> reward={reward}")
        else:
            coalition_query = " ".join([phrases[i] for i in coalition]) 

            if hasattr(search_reward_func, "_evaluate_single_query"):
                reward = search_reward_func._evaluate_single_query(coalition_query, target_item, sample_idx = sample_idx)
            else:
                reward = 0.0  # fallback

            # Print coalition indices, query string, and reward
            coalition_indices = [phrases[i] for i in coalition]
            # print(f"Coalition {coalition_indices} -> query='{coalition_query}' -> reward={reward}")

        coalition_rewards.append(reward)

    return torch.tensor(coalition_rewards, dtype=torch.float32, device=device)

def generate_coalition_combinations(num_phrases: int,
                                    max_width: int = 4,
                                    include_empty: bool = True,
                                    include_full: bool = True):
    combos = []
    if include_empty:
        combos.append([])

    # contiguous windows up to max_width
    for w in range(1, min(max_width, num_phrases) + 1):
        for start in range(0, num_phrases - w + 1):
            combos.append(list(range(start, start + w)))

    if include_full and num_phrases > 0 and list(range(num_phrases)) not in combos:
        combos.append(list(range(num_phrases)))
    return combos

def compute_sentence_owen_values(
    coalition_rewards: torch.Tensor,
    coalition_combinations: List[List[int]], 
    num_sentences: int
) -> torch.Tensor:
    """Compute Owen values for each phrase (or sentences) using coalition rewards."""
    sentence_attributions = torch.zeros(num_sentences, device=coalition_rewards.device)
    # Create mapping from coalition to reward
    coalition_to_reward = {}
    for i, coalition in enumerate(coalition_combinations):
        coalition_key = tuple(sorted(coalition))
        coalition_to_reward[coalition_key] = coalition_rewards[i]
    
    # Compute marginal contributions for each sentence
    for sentence_idx in range(num_sentences):
        marginal_contributions = []
        
        for coalition in coalition_combinations:
            if sentence_idx not in coalition:
                # Coalition without sentence
                coalition_without = tuple(sorted(coalition))
                # Coalition with sentence  
                coalition_with = tuple(sorted(coalition + [sentence_idx]))
                
                if coalition_without in coalition_to_reward and coalition_with in coalition_to_reward:
                    # Marginal contribution = V(S ∪ {i}) - V(S)
                    marginal = coalition_to_reward[coalition_with] - coalition_to_reward[coalition_without]
                    
                    # Owen value weight: 1/(n * C(n-1, |S|))
                    coalition_size = len(coalition)
                    if coalition_size < num_sentences:
                        weight = 1.0 / (num_sentences * math.comb(num_sentences - 1, coalition_size))
                        marginal_contributions.append(marginal * weight)
        
        # Sum weighted marginal contributions
        if marginal_contributions:
            sentence_attributions[sentence_idx] = sum(marginal_contributions) 
            
    
    return sentence_attributions

def _extract_query(completion):
        """Extract expanded query from <answer> tags"""
        # Handle chat format if completion is a list
        if isinstance(completion, list):
            text = completion[0].get('content', '') if completion else ''
        else:
            text = str(completion)
        
        # Extract from <answer> tags
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Fallback: remove <think> tags and use rest
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        return text.strip()