import faiss
import json
from collections import deque
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import torch.nn.functional as F
import re
import math
from collections import OrderedDict
import logging; logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)


class SearchRewardFunction:
    """
    Computes NDCG-based rewards for search query generation tasks.
    
    Encodes generated queries using a sentence transformer, retrieves items via FAISS,
    optionally filters by candidate pools, and computes ranking metrics (NDCG, MRR, AP, Recall).
    
    Args:
        faiss_index_path: Path to pre-built FAISS index for item embeddings
        mapping_path: JSON mapping FAISS indices to item IDs (ASINs)
        model_name: HuggingFace model name for query encoding (e.g., 'sentence-transformers/all-mpnet-base-v2')
        device: Device for embedding model ('cuda' or 'cpu')
        top_k: Number of items to retrieve from FAISS before filtering
        candidate_pools: Optional dict {sample_idx: [valid_item_ids]} to restrict retrieval
        debug: If True, logs detailed diagnostics per query
        debug_max_items: Maximum debug records to retain in memory
    
    Key Methods:
        __call__(prompts, completions, target_item_id, **kwargs):
            Computes rewards for a batch of generated queries.
            Returns list of scalar rewards (NDCG-based).
        
        dump_debug(path): Saves debug logs to pickle file
        validate_catalog_alignment(n): Checks overlap between pools and catalog
    
    Reward Components:
        - Primary: NDCG@k (default weight: 1.0)
        - Auxiliary (logged but not used): Recall, MRR, AP
    
    Diagnostics:
        Tracks pre/post-filter metrics, target coverage, rank positions, and pool overlap.
        Use debug=True and dump_debug() for detailed query-level analysis.
    
    Example:
        >>> search_fn = SearchRewardFunction(
        ...     faiss_index_path='index.bin',
        ...     mapping_path='mapping.json',
        ...     model_name='sentence-transformers/all-mpnet-base-v2',
        ...     candidate_pools={0: ['item1', 'item2']}
        ... )
        >>> rewards = search_fn(prompts, completions, target_ids, sample_idx=[0])
    """
    def __init__(self, faiss_index_path, mapping_path, model_name, device='cuda', top_k = 500, candidate_pools=None, debug = False, debug_max_items=500):
        self.faiss_index = faiss.read_index(faiss_index_path)
        self.__name__ = 'SearchRewardFunction'
        with open(mapping_path, 'r') as f:
            self.id_mapping = json.load(f)
        # Use transformers directly with safetensors
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, use_safetensors=True)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.top_k = top_k
        self.debug = debug
        self.device_param = device # Store device parameter but will use model's actual device
        self.last_retrievals = []  # Store last batch's retrievals
        self.search_reward_weights = {
            "ndcg": 1.0,         
            "soft_recall": 0.3, # extra dense signal
            "ap": 0.4,
            'hit_rate': 0.5    
        }
        self.soft_recall_tau = 10.0  # decay for exp(-rank/tau); tune per collection
        canon = lambda x: str(x).strip()
        self.id_mapping = {canon(k): canon(v) for k, v in self.id_mapping.items()}
        self.candidate_pools = candidate_pools  # Dict: {sample_idx: [list of valid item IDs]}
        if candidate_pools is not None:
            self.candidate_pools = {
                int(k): [canon(v) for v in vs]
                for k, vs in candidate_pools.items()
            }
        self.debug_counter = 0
        self.debug_logs = deque(maxlen=debug_max_items)  # each entry = one query’s debug dict
        print("SearchRewardFunction initialized!")

    def _log_debug(self, record: dict):
        if not self.debug:
            return
        self.debug_counter += 1
        self.debug_logs.append(record)
    def dump_debug(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(list(self.debug_logs), f)
        print(f"[DEBUG] wrote {len(self.debug_logs)} records → {path}")
    
    def validate_catalog_alignment(self, n=1000):
        canon = lambda x: str(x).strip()
        catalog = set(self.id_mapping.values())
        pools = []
        if self.candidate_pools:
            for _, p in list(self.candidate_pools.items())[:n]:
                pools.extend(p)
        pools = [canon(x) for x in pools]
        cov = sum(1 for x in pools if x in catalog)
        print(f"[CHECK] pool-in-catalog: {cov}/{len(pools)} ({100*cov/max(1,len(pools)):.1f}%)")

    def encode(self, texts):
        # Get actual device from model (handles DDP correctly)
        device = next(self.model.parameters()).device
         
        encoded = self.tokenizer(texts, padding=True, truncation=True, 
                                return_tensors='pt', max_length=512)
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = self.model(**encoded)
        # Mean pooling
        token_embeddings = output[0]
        attention_mask = encoded['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def __call__(self, prompts, completions, target_item_id, **kwargs):
        rewards = []
        self.last_retrievals = []
        canon = lambda x: str(x).strip()
        def _trunc(xs, k):
            return (xs or [])[:k]
        
        global_indices = kwargs.get('sample_idx', None) #sample index in the RL dataset for accessing candidate pools. 
 
        for i, completion in enumerate(completions):
            try:
                # Handle both single item (str) and multiple items (list)
                target_items = target_item_id[i] if i < len(target_item_id) else None
                
                # Convert single string to list for uniform handling
                if isinstance(target_items, str):  # Only converts if it's a string, skips if already a list
                    target_items = [target_items] if target_items != "NONE" else []
                target_items = [canon(t) for t in (target_items or [])]
                target_set   = set(target_items)
                if not target_items or all(t == "NONE" for t in target_items):
                    rewards.append(0.0)
                    self.last_retrievals.append({
                        'target': target_items,
                        'retrieved_items': [],
                        'expanded_query': None,
                        'reward': 0.0
                    })
                    continue
                
                expanded_query = self._extract_query(completion)
           
                
                if not expanded_query:
                    rewards.append(0.0)
                    self.last_retrievals.append({
                        'target': target_items,
                        'retrieved_items': [],
                        'expanded_query': None,
                        'reward': 0.0
                    })
                    continue
                
                # Encode and search (same)
                query_embedding = self.encode([expanded_query])
                faiss.normalize_L2(query_embedding)
                scores, indices = self.faiss_index.search(query_embedding, self.top_k)
                
                # Map indices to IDs (same)
                retrieved_items = []
                for idx in indices[0]:
                    if str(idx) in self.id_mapping:
                        retrieved_items.append(str(self.id_mapping[str(idx)]))
                
                # ---- DIAGNOSTICS (PRE-FILTER) ----
                retrieved_raw = [str(x).strip() for x in retrieved_items]  # BEFORE filtering
                raw_n = len(retrieved_raw)
                raw_unique = len(set(retrieved_raw))

                # Where are targets in raw list?
                target_set = set(map(str, target_items))
                raw_ranks_all = [retrieved_raw.index(t) + 1 for t in target_set if t in retrieved_raw]  # 1-based
                raw_hits = len(raw_ranks_all)
                raw_top100_ranks = [r for r in raw_ranks_all if r <= 100]
                target_in_faiss = raw_hits > 0

                # Optional raw metrics for comparison (k=100 and k=1000)
                def _ndcg_at_k(list_ids, targets, k):
                    return self._calculate_ndcg(list_ids, targets, k=k)
                ndcg_raw_100  = _ndcg_at_k(retrieved_raw, target_items, k=100)
                ndcg_raw_1000 = _ndcg_at_k(retrieved_raw, target_items, k=min(1000, len(retrieved_raw)))

                # --- Candidate pool info (before applying it) ---
                pool = []
                pool_size = None
                if self.candidate_pools and global_indices is not None:
                    pool = list(self.candidate_pools.get(global_indices[i], []))
                    pool = [str(x).strip() for x in pool]
                    pool_size = len(pool)
                    target_in_pool = any(t in pool for t in target_set)
                else:
                    target_in_pool = False

                if getattr(self, "debug", True):
                    print(f"[PRE] faiss_n={raw_n} uniq={raw_unique} "
                        f"| targets_in_faiss={raw_hits} ranks={sorted(raw_ranks_all)[:5]} "
                        f"| ndcg_raw@100={ndcg_raw_100:.4f} ndcg_raw@1000={ndcg_raw_1000:.4f} "
                        f"| pool_size={pool_size} targets_in_pool={target_in_pool}")
                    print("[PRE] raw first 10:", retrieved_raw[:10])
                    raw_hits = len(raw_ranks_all)  # 0 if target not in FAISS raw list
                    if raw_hits > 0 and not target_in_pool:
                        # pick a target that appeared in raw list for local context
                        t0 = next((t for t in target_set if t in retrieved_raw), None)
                        t_rk = retrieved_raw.index(t0) + 1 if t0 else None  # 1-based

                        # small raw window around the rank
                        if t0 is not None:
                            s = max(0, (t_rk - 1) - 5)
                            e = min(len(retrieved_raw), (t_rk - 1) + 6)
                            raw_ctx = retrieved_raw[s:e]
                        else:
                            raw_ctx = []

                        # short strings for readability
                        ans_short = (expanded_query[:200] + "…") if expanded_query and len(expanded_query) > 200 else expanded_query
                      

                        print("\n[WHY-NOT-IN-POOL]")
                        print(f"  sample_idx:   {int(global_indices[i]) if global_indices is not None else None}")
                        print(f"  targets:      {list(target_set)[:5]}")
                        print(f"  faiss_ranks:  {sorted(raw_ranks_all)[:5]}  (1-based)")
                        if t0 is not None:
                            print(f"  raw context @{t_rk}: {raw_ctx}")
                        print(f"  pool_size:    {pool_size}")
                        print(f"  pool_head:    {pool[:12]}")
                        if ans_short:
                            print(f"  answer:       {ans_short}")
                        if ctx_short:
                            print(f"  context:      {ctx_short}")
                        print("[/WHY-NOT-IN-POOL]\n")
                
                # ---- APPLY CANDIDATE-POOL FILTERING ----
                if self.candidate_pools and global_indices is not None:
                    valid_items = set(pool)
                    overlap_uniq = len(set(retrieved_raw) & valid_items)
                    retrieved_items = [x for x in retrieved_raw if x in valid_items]  # keep order
                else:
                    overlap_uniq = None
                    retrieved_items = retrieved_raw[:]
       
                if self.debug:
                    logger.info(f"sample={i} expanded_query='{expanded_query[:50] if expanded_query else None}' target={target_items[:3] if len(target_items) > 3 else target_items} pool_size={pool_size} retrieved_before_filter={len(retrieved_raw)} retrieved_after_filter={len(retrieved_items)}")

                # ---- DIAGNOSTICS (POST-FILTER) ----
                filt_n = len(retrieved_items)
                filt_unique = len(set(retrieved_items))
                post_ranks = [retrieved_items.index(t) + 1 for t in target_set if t in retrieved_items]
                target_in_post = len(post_ranks) > 0

                # Metrics at effective lengths
                k_post = filt_n if filt_n > 0 else 1
                ndcg_post = self._calculate_ndcg(retrieved_items, target_items, k=k_post)
                k_pool = pool_size or k_post
                ndcg_at_pool = self._calculate_ndcg(retrieved_items[:k_pool], target_items, k=k_pool)

                if getattr(self, "debug", True):
                    print(f"[POST] post_n={filt_n} uniq={filt_unique} "
                        f"| overlap_uniq={overlap_uniq} "
                        f"| targets_in_post={len(post_ranks)} ranks={sorted(post_ranks)[:5]} "
                        f"| ndcg_post@post={ndcg_post:.4f} ndcg_post@pool={ndcg_at_pool:.4f}")
                    print("[POST] first 10:", retrieved_items[:10])

                # --- derive hit/coverage + ranks we’ll log ---
                raw_hit = (len(raw_ranks_all) > 0)
                first_hit_rank_raw = (min(raw_ranks_all) if raw_hit else None)

                post_hit = (len(post_ranks) > 0)
                first_hit_rank_post = (min(post_ranks) if post_hit else None)

                # catalog coverage: target appeared in raw FAISS top-k (pre-pool)
                target_in_faiss_top1000 = raw_hit  # since retrieved_raw is your raw@top_k list

                # zero-reward flag (for quick health checks)
                zero_reward = (ndcg_at_pool == 0.0)

                # overlap rate between raw and pool (unique)
                overlap_rate = (overlap_uniq / float(filt_n)) if (overlap_uniq is not None and filt_n > 0) else 0.0

                # after computing raw/post metrics & ranks
                self._log_debug({
                    "expanded_query": expanded_query,
                    "sample_idx": int(global_indices[i]) if global_indices is not None else None,
                    "targets": list(target_items),
                    "pool_size": pool_size,
                    "overlap_uniq": overlap_uniq,
                    "overlap_rate": overlap_rate,                        
                    "faiss_raw_n": raw_n,
                    "faiss_raw_unique": raw_unique,
                    "target_in_faiss_top1000": target_in_faiss_top1000,  
                    "target_in_pool": target_in_pool, 
                    "raw_target_ranks": sorted(raw_ranks_all)[:5],
                    "first_hit_rank_raw": first_hit_rank_raw,           
                    "post_n": len(retrieved_items),
                    "post_target_ranks": sorted(post_ranks)[:5],
                    "first_hit_rank_post": first_hit_rank_post,          
                    "ndcg_raw@100": ndcg_raw_100,
                    "ndcg_raw@1000": ndcg_raw_1000,
                    "ndcg_post@post": ndcg_post,
                    "ndcg_post@pool": ndcg_at_pool,
                    # "mrr@pool": mrr_at_pool,                            
                    "zero_reward": zero_reward,                        
                    "post_first10": retrieved_items[:10],
                })

                # Ensure comparable string IDs (post-filter list)
                retrieved_items = [str(x).strip() for x in retrieved_items]
                if isinstance(target_items, str):
                    target_items = [target_items]
                target_items = [str(t).strip() for t in (target_items or [])]

                # Calculate NDCG on filtered results
                # k_for_ndcg = len(retrieved_items) if retrieved_items else 1
                k_for_metrics = len(retrieved_items) if retrieved_items else 1
                # pool-aware K
                k_pool = pool_size  # use the candidate-pool size for this sample
                # --- Metrics ---
                ndcg_reward = self._calculate_ndcg(retrieved_items, target_items, k=k_for_metrics)

                found = [t for t in target_items if t in retrieved_items]
                ranks = [retrieved_items.index(t) + 1 for t in found]  # 1-based
                # print(f"[NDCG DEBUG] post={len(retrieved_items)} targets={len(target_items)} "
                #     f"hits={len(found)} ranks={ranks} ndcg={ndcg_reward:.4f}")

                ap_reward   = self._average_precision(retrieved_items, target_items, k=k_for_metrics)
                mrr_reward  = self._mrr(retrieved_items, target_items, k=k_for_metrics)
                recall_k    = self._calculate_recall_at_k(retrieved_items, target_items, k=k_for_metrics)
                

                # compute @pool metrics on the post-filter 
                ndcg_at_pool   = self._calculate_ndcg(_trunc(retrieved_items, k_pool), target_items, k=k_pool)
                recall_at_pool = self._calculate_recall_at_k(_trunc(retrieved_items, k_pool), target_items, k=k_pool)
                mrr_at_pool    = self._mrr(_trunc(retrieved_items, k_pool), target_items, k=k_pool)
                ap_at_pool     = self._average_precision(_trunc(retrieved_items, k_pool), target_items, k=k_pool)
                
                
                # ndcg_reward = self._calculate_ndcg(retrieved_items, target_items, k=k_for_ndcg)
                w = self.search_reward_weights
                reward = w["ndcg"] * ndcg_reward
                rewards.append(reward)
                # Store for logging
                
                self.last_retrievals.append({
                    "target": target_items,
                    "retrieved_items": retrieved_items,
                    "expanded_query": expanded_query,
                    "reward": reward,
                    "pool_size": pool_size,
                    "n_raw": raw_n,
                    "n_filt": len(retrieved_items),
                    "overlap_uniq": overlap_uniq,       
                    "overlap_rate": overlap_rate,       
                    "target_in_faiss_top1000": target_in_faiss_top1000, 
                    "first_hit_rank_raw": first_hit_rank_raw,             
                    "first_hit_rank_post": first_hit_rank_post,           
                    "zero_reward": zero_reward,                           
                    "ndcg": ndcg_reward,
                    "ap": ap_reward,
                    "mrr": mrr_reward,
                    "recall": recall_k,
                    "ndcg@pool": ndcg_at_pool,
                    "recall@pool": recall_at_pool,
                    "mrr@pool": mrr_at_pool,
                    "ap@pool": ap_at_pool,
                    "rank": [retrieved_items.index(t) + 1 for t in target_items if t in retrieved_items],
                })
 
                
            except Exception as e:
                print(f"Error computing reward for sample {i}: {e}")
                rewards.append(0.0)
                self.last_retrievals.append({
                    'target': target_items if 'target_items' in locals() else None,
                    'retrieved_items': [],
                    'expanded_query': None,
                    'reward': 0.0,
                    'error': str(e)
                })
        
        return rewards
        
    def _extract_query(self, completion):
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

    def _calculate_ndcg(self, retrieved_items, target_items, k=None):
        """Multi-item binary NDCG (works for single item too)"""
        if k is None:
            k = self.top_k
        
        try:
            # Handle both single string and list
            if isinstance(target_items, str):
                target_items = [target_items]
            
            target_set = set(str(x) for x in target_items)
            
            # DCG
            dcg = 0.0
            for i, item in enumerate(retrieved_items[:k]):
                if str(item) in target_set:
                    dcg += 1.0 / math.log2(i + 2)
            
            # IDCG
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(target_items), k)))
            
            return dcg / idcg if idcg > 0 else 0.0

        except (ValueError, ZeroDivisionError):
            return 0.0

    def _evaluate_single_query(self, query: str, target_items, sample_idx=None, k: int = None) -> float:
        """Lightweight single-query scorer used by Owen/coalitions - supports multi-item targets."""
        if not query:
            return 0.0
        
        # Handle both single item and list of items
        if isinstance(target_items, str):
            if target_items == "NONE":
                return 0.0
            target_items = [target_items]
        
        if not target_items or all(t == "NONE" for t in target_items):
            return 0.0
        
        try:
            # Encode & search
            q = self.encode([query])
            faiss.normalize_L2(q)
            scores, idxs = self.faiss_index.search(q, self.top_k)
            
            # Map to IDs (ASINs for ESCI, article_ids for H&M)
            retrieved = [str(self.id_mapping.get(str(i), "")) for i in idxs[0]]
            # Apply candidate pool filtering
            if self.candidate_pools and sample_idx is not None and sample_idx in self.candidate_pools:
                valid_items = set(self.candidate_pools[sample_idx])

                # DEBUG: Check if any targets are in the pool
                targets_in_pool = [t for t in target_items if t in valid_items]
                # print(f"DEBUG: Targets in pool: {len(targets_in_pool)}/{len(target_items)}")
                
                retrieved = [item for item in retrieved if item in valid_items]
                # print(f"DEBUG: Retrieved after filtering: {len(retrieved)} items")
                retrieved = [item for item in retrieved if item in valid_items]

                if retrieved:
                    targets_retrieved = [item for item in retrieved if item in target_items]
                    # print(f"DEBUG: Retrieved targets: {targets_retrieved}")
                    # print(f"DEBUG: All retrieved are targets: {len(targets_retrieved) == len(retrieved)}")
            
            # Calculate NDCG on filtered results
            k_for_ndcg = len(retrieved) if retrieved else 1
            return self._calculate_ndcg(retrieved, target_items, k=k_for_ndcg)
            
        except Exception as e:
            print(f"[SearchRewardFunction] _evaluate_single_query error: {type(e).__name__}: {e}", flush=True)
            return 0.0

    def _calculate_hit_rate_at_k(self, retrieved_items, target_items, k=None):
        """
        Hit rate at k: 1 if any target appears in top k, 0 otherwise.
        For multiple targets, returns the fraction of targets found in top k.
        Range [0,1].
        
        Args:
            retrieved_items: List of retrieved item IDs
            target_items: Single target item or list of target items
            k: Number of top items to consider (defaults to self.top_k)
        
        Returns:
            float: For single target: 1.0 if found, 0.0 if not
                For multiple targets: fraction of targets found in top k
        """
        if k is None:
            k = self.top_k
        if isinstance(target_items, str):
            target_items = [target_items]
        
        retrieved_slice = retrieved_items[:k]
        retrieved_set = set(retrieved_slice)
        
        hits = sum(1 for t in target_items if t in retrieved_set)
        return float(hits) / float(len(target_items)) if target_items else 0.0


    def _calculate_soft_recall(self, retrieved_items, target_items, k=None, tau=20.0):
        """
        Rank-shaped recall: average of g(rank) over targets, with g decreasing smoothly with rank.
        Missing targets contribute 0. Range ~[0,1].
        g(rank) here uses exp(-rank/tau); use tau ~ 10-50 depending on top_k scale.
        """
        if k is None: k = self.top_k
        if isinstance(target_items, str): target_items = [target_items]
        retrieved_slice = retrieved_items[:k]

        gains = []
        for t in target_items:
            try:
                r = retrieved_slice.index(t)  # 0-based
                gains.append(math.exp(-float(r) / float(tau)))
            except ValueError:
                gains.append(0.0)
        return float(sum(gains)) / float(len(target_items)) if target_items else 0.0


    def _average_precision(self, retrieved_items, target_items, k=None):
        """
        Average Precision@k (AP): average of precision@r over each hit r.
        Smooth and granular for multi-target sets. Range [0,1].
        """
        if k is None: k = self.top_k
        if isinstance(target_items, str): target_items = [target_items]
        target_set = set(str(x) for x in target_items)
        if not target_set: return 0.0

        hits = 0
        precisions = []
        for i, item in enumerate(retrieved_items[:k]):
            if item in target_set:
                hits += 1
                precisions.append(hits / float(i + 1))
        if not precisions: return 0.0
        # normalize by number of relevant items (|target_set|) to keep in [0,1]
        return sum(precisions) / float(len(target_set))

    def _mrr(self, retrieved_items, target_items, k=None):
        if k is None: k = self.top_k
        if isinstance(target_items, str): target_items = [target_items]
        retrieved_slice = retrieved_items[:k]
        best = None
        for t in target_items:
            try: best = min(best, retrieved_slice.index(t)) if best is not None else retrieved_slice.index(t)
            except ValueError: pass
        return 1.0 / (best + 1) if best is not None else 0.0

    def _calculate_recall_at_k(self, retrieved_items, target_items, k=None):
        """Recall@k - proportion of targets found"""
        if k is None:
            k = self.top_k
        
        try:
            if isinstance(target_items, str):
                target_items = [target_items]
            
            target_set = set(str(x) for x in target_items)
            retrieved_set = set(str(x) for x in retrieved_items[:k])
            
            found = len(target_set & retrieved_set)
            return found / len(target_items) if target_items else 0.0
        
        except Exception:
            return 0.0

def _unwrap_search_instance(reward_func):
    # Case A: the instance itself was passed
    if hasattr(reward_func, "faiss_index"):
        return reward_func
    # Case B: a bound method
    if hasattr(reward_func, "__self__") and hasattr(reward_func.__self__, "faiss_index"):
        return reward_func.__self__
    # Case C: a closure capturing the instance (your case)
    clos = getattr(reward_func, "__closure__", None)
    if clos:
        for cell in clos:
            try:
                obj = cell.cell_contents
            except Exception:
                continue
            if hasattr(obj, "faiss_index"):
                return obj
    return None


 

