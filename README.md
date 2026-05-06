# Owen-Shapley Policy Optimization (OSPO)

This repository contains implementation code for the submission, "Owen-Shapley Policy Optimization: A Principled RL Algorithm for Generative Search LLMs". 

---

## Installation

### Requirements
- Python 3.10+
- CUDA-capable GPU(s)  
 

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd repo_name
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Key dependencies include:
- `transformers`, `accelerate`, `trl` (LLM training)
- `sentence-transformers` (embedding generation)
- `faiss-gpu` (vector search)
- `datasets` (HuggingFace datasets)
- `torch`, `numpy`, `pandas`

---


## Project Structure
```
project_code/
├── data/                          # Datasets and indices
│   └── esci/                      # ESCI product search data
│       ├── metadata/
│       │   └── item_catalog.jsonl          # ASIN→metadata mapping (~3M items, 1.8GB)
│       ├── embeddings/
│       │   ├── all-mpnet-base-v2.npy       # Dense embeddings (3M × 768, 10GB)
│       │   └── all-mpnet-base-v2_asin_mapping.json
│       ├── index/
│       │   ├── all-mpnet-base-v2_faiss.bin # FAISS HNSW index (10GB)
│       │   ├── all-mpnet-base-v2_asin_mapping.json
│       │   └── all-mpnet-base-v2_faiss_metadata.json
│       ├── rl_dataset/            # RL training dataset
│       ├── sft_dataset.jsonl      # Generated SFT training data
│       └── dpo_dataset.jsonl      # Generated DPO preference pairs
│
├── src/esci_search/               # Main source code
│   ├── configs/                   # Training configurations
│   │   ├── grpo_config.yaml       # GRPO baseline config
│   │   ├── ospo_prop_config.yaml  # OSPO proportional
│   │   ├── ospo_rank_config.yaml  # OSPO rank-based
│   │   └── ospo_prop_no_clip.yaml # OSPO without gradient clipping
│   │
│   ├── data_processing/           # Data preparation pipeline
│   │   ├── build_index.py         # Build FAISS search index
│   │   ├── create_item_metadata.py
│   │   ├── generate_embeddings.py
│   │   ├── generate_rl_data.py    # Create RL training dataset
│   │   ├── generate_dpo_sft_data.py   # Generate model trajectories
│   │   ├── process_sft_dpo_data.py    # Process into SFT/DPO format
│   │   ├── run_sft_dpo_pipeline.sh    # End-to-end dataset generation
│   │   └── sample_queries.py
│   │
│   ├── evals/                     # Evaluation scripts
│   │   ├── generate_test.py       # Run inference on test set
│   │   ├── score_search_only.py   # Compute retrieval metrics
│   │   └── test_generations_sft_dpo/  # Generated trajectory CSVs
│   │
│   └── trainers/                  # Training implementations
│       ├── dense_search/          # Dense retrieval (FAISS)
│       │   └── search.py
│       ├── train_sft.py           # Supervised fine-tuning
│       ├── train_dpo.py           # Direct preference optimization
│       ├── train_grpo.py          # Group relative policy optimization
│       ├── train_ospo.py          # Owen-Shapley policy optimization
│       ├── ospo_utils.py          # OSPO-specific utilities
│       ├── reward_utils.py        # Reward computation
│       └── generation_utils.py    # Text generation helpers
│
├── outputs/                       # Training outputs (created at runtime)
│   ├── ospo_ablations_esci/       # OSPO/GRPO checkpoints
│   ├── sft_models/                # SFT checkpoints
│   └── dpo_models/                # DPO checkpoints
 
├── requirements.txt
 
```

### Directory Overview

- **`data/esci/`**: ESCI product search dataset and artifacts
  - **`metadata/`**: Product catalog with ASIN metadata (~3M items, 1.8GB)
  - **`embeddings/`**: Pre-computed dense embeddings (all-mpnet-base-v2, 10GB)
  - **`index/`**: FAISS HNSW search index for dense retrieval (10GB)
  - **`rl_dataset/`**: Prepared dataset for RL training (queries + candidate pools)
  - **`sft_dataset.jsonl`**: High-quality samples for supervised fine-tuning
  - **`dpo_dataset.jsonl`**: Preference pairs for direct preference optimization
  
- **`src/esci_search/`**: Core codebase for the ESCI product search task
  - **`configs/`**: YAML configurations for different training methods (OSPO, GRPO)
  - **`data_processing/`**: Scripts to prepare datasets, indices, embeddings, and SFT/DPO data
  - **`evals/`**: Inference and evaluation utilities, trajectory generation outputs
  - **`trainers/`**: Training implementations (SFT, DPO, GRPO, OSPO) and search utilities
  
- **`outputs/`**: All model checkpoints organized by training method (git-ignored)
  - **`ospo_ablations_esci/`**: OSPO and GRPO model checkpoints
  - **`sft_models/`**: Supervised fine-tuning checkpoints
  - **`dpo_models/`**: Direct preference optimization checkpoints

---

## Data Processing Pipeline

The pipeline prepares the ESCI Shopping Queries dataset for OSPO training by creating item metadata, dense embeddings, search indices, ground truth queries, and RL-formatted training data. A similar set of steps can be followed to generate the H&M training and evaluation data for the contextualized product search tasks. 


## What does the data look like?
Datasets for both tasks are provided in `data/esci/rl_dataset/` 
(ESCI: `esci_sample_train_500.pkl`, `esci_sample_test_100.pkl`) and 
`data/hnm/rl_dataset/` (H\&M: `hnm_sample_train_500.pkl`, 
`hnm_sample_test_100.pkl`)


## Dataset Fields for ESCI Product Search

- **sample_idx**: Original sample index for reproducibility and cross-referencing with the full ESCI dataset
- **question**: Raw customer search query as submitted to the search engine
- **prompt**: Full chat-formatted prompt with system instructions and user query, ready for model inference
- **answer**: List of ground truth ASIN(s) that are exact matches for this query — used for NDCG computation
- **target_item_id**: Primary target ASIN selected as the retrieval target for this sample
- **candidate_pool**: List of ASINs (target + negatives sampled from same/related categories) used for candidate-filtered NDCG evaluation with FAISS retrieval

## Dataset Fields for H&M Product Search

- **question**: Combined customer query and purchase history formatted as the model input question
- **prompt**: Full chat-formatted prompt with system instructions and user message, ready for model inference
- **answer**: Target item ID (article number) the model should retrieve — used as ground truth for NDCG computation
- **target_item_id**: Same as answer — the H&M article ID of the correct target item for this query
- **item_context**: Expert-generated natural language description of the target item used during data generation
- **global_idx**: Global index of this sample in the original H&M dataset before subsampling
- **purchase_history**: Pipe-separated list of the customer's past H&M purchases (product name, type, color)
- **query**: Raw customer search query before any processing or prompt formatting
- **candidate_pool**: List of 100 item IDs (1 target + 99 negatives) used for NDCG evaluation with FAISS retrieval
- **sample_idx**: Original sample index used for reproducibility and cross-referencing with full dataset



### Option A: Full Pipeline (Recommended)

Run all 5 steps sequentially from the data processing directory:

```bash
cd src/esci_search/data_processing
bash data_process.sh
```

This executes all stages in order and stops automatically if any step fails.

---

### Option B: Step-by-Step Pipeline

Run each step individually for debugging or customization:

#### Step 1: Create Item Metadata

**Script:** `create_item_metadata.py`

Extracts product metadata (titles + descriptions) from Amazon Reviews dataset for all ASINs in ESCI.

**What it does:**
- Downloads ESCI dataset from HuggingFace
- Filters for US locale, exact matches, small version
- Samples 50 negatives per query from same category
- Streams metadata from Amazon Reviews JSONL files
- Creates ASIN→metadata mapping

**Output:** `data/esci/metadata/item_catalog.jsonl` (~3M items, 1.8GB)

```bash
python create_item_metadata.py
```

---

#### Step 2: Generate Embeddings

**Script:** `generate_embeddings.py`

Creates dense vector embeddings for all products using sentence-transformers.

**What it does:**
- Loads `item_catalog.jsonl`
- Encodes metadata using `all-mpnet-base-v2` (768-dim)
- Multi-GPU acceleration (distributes across available GPUs)
- Saves embeddings as binary numpy array
- Creates index→ASIN mapping for FAISS lookups

**Output:**
- `data/esci/embeddings/all-mpnet-base-v2.npy` 
- `data/esci/embeddings/all-mpnet-base-v2_asin_mapping.json`  

**Runtime:** ~2-3 hours for 3M items on 2× RTX A6000

```bash
python generate_embeddings.py \
    --model_name sentence-transformers/all-mpnet-base-v2 \
    --batch_size 1024
```

---

#### Step 3: Build FAISS Index

**Script:** `build_index.py`

Builds HNSW index for fast approximate nearest neighbor search.

**What it does:**
- Loads embeddings from Step 2
- Normalizes vectors for cosine similarity
- Builds FAISS HNSW index (M=32, ef_construction=200)
- Copies ASIN mapping to index directory
- Saves index metadata (dimensions, build time, etc.)

**Output:**
- `data/esci/index/all-mpnet-base-v2_faiss.bin` (160MB)
- `data/esci/index/all-mpnet-base-v2_asin_mapping.json` (1.2MB)
- `data/esci/index/all-mpnet-base-v2_faiss_metadata.json` (152B)

**Runtime:** ~2-5 minutes

```bash
python build_index.py
```

---

#### Step 4: Sample Ground Truth Queries

**Script:** `sample_queries.py`

Curates high-quality queries with multiple verified exact matches for evaluation.

**What it does:**
- Processes ESCI to group exact matches by query
- Filters for queries with 3-10 exact matches
- Validates semantic relevance (≥30% token overlap with titles)
- Samples 40,000 queries with fixed random seed
- Deduplicates product matches per query

**Output:** `data/esci/ground_truth/query_ground_truth_40000.json`

```bash
python sample_queries.py \
    --num_queries 40000 \
    --min_matches 3 \
    --max_matches 10 \
    --seed 42
```

---

#### Step 5: Generate RL Training Dataset

**Script:** `generate_rl_data.py`

Creates TRL-compatible RL dataset with candidate pools for OSPO/GRPO training.

**What it does:**
- Loads ground truth queries from Step 4
- Loads ASIN mapping from index to filter valid items
- Samples diverse negatives per query:
  - 50 from same category (hard negatives)
  - 30 from related categories (medium)
  - 20 random from catalog (easy)
- Creates candidate pools (1 target + 99 negatives)
- Formats prompts with query expansion system instructions
- Splits 90/10 into train/test

**Output:** `data/esci/rl_dataset/` (HuggingFace DatasetDict)
- Train: 9,339 samples
- Test: 1,038 samples

```bash
python generate_rl_data.py
```

---

## Model Training

We provide training scripts for OSPO and GRPO baselines. All experiments use **Qwen/Qwen2.5-7B-Instruct** as the base model with LoRA fine-tuning.

### Configuration Files

Training configs are located in `src/esci_search/configs/`:
- `ospo_prop_config.yaml` - OSPO with proportional redistribution
- `ospo_rank_config.yaml` - OSPO with rank-based redistribution
- `ospo_prop_no_clip.yaml` - OSPO without advantage clipping
- `grpo_config.yaml` - GRPO baseline

### Key Hyperparameters

```yaml
# Model Parameters
model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"
use_peft: true
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
load_in_4bit: true
torch_dtype: "bfloat16"

# Training Hyperparameters
learning_rate: 5.0e-6
lr_scheduler_type: "cosine"
warmup_steps: 100
weight_decay: 0.05
optimizer_type: "paged_adamw_32bit"
max_steps: 2000
seed: 0

# Batch & Distribution
per_device_train_batch_size: 8
per_device_eval_batch_size: 8
gradient_accumulation_steps: 2
gradient_checkpointing: true
num_generations: 8  # MCMC samples for OSPO
max_prompt_length: 256  
max_completion_length: 356  

#search reward specific
faiss_topk: 1000

# Data Paths
dataset_path: "data/esci/rl_dataset"
faiss_index_path: "data/esci/index/all-mpnet-base-v2_faiss.bin"
asin_mapping_path: "data/esci/index/all-mpnet-base-v2_asin_mapping.json"
metadata_path: "data/esci/metadata/item_catalog.jsonl"
embedding_model: "sentence-transformers/all-mpnet-base-v2"
 

# OSPO / Search Specific
owen_max_width: 32
owen_max_permutations: 64
redistribution_mode: "owen_weights"  
clip_ospo_advantages: true
faiss_topk: 1000
reward_topk: 100

# Logging & Eval
output_dir: "./outputs/ospo_ablations_esci"
logging_steps: 1
save_steps: 500
eval_steps: 200
eval_strategy: "steps"
report_to: "none"
log_completions: true
sanity_check: false
```

**Note:** Token length parameters (256 for prompts, 356 for completions) are shorter than H&M dataset since ESCI prompts are based on conventional query rewrites which are naturally shorter. Verify these values with your tokenizer capture >99% of the dataset.


### Training Pipeline

#### 1. Generate SFT/DPO Datasets
First, generate training datasets from model trajectories:
```bash
cd src/esci_search/data_processing
./run_sft_dpo_pipeline.sh
```
This creates:
- `data/esci/sft_dataset.jsonl` - High-quality samples (NDCG > 0.3)
- `data/esci/dpo_dataset.jsonl` - Preference pairs (chosen vs rejected)
- Generation CSVs in `src/esci_search/evals/test_generations_sft_dpo/`

#### 2. SFT Training (Optional but Recommended)
Train with supervised fine-tuning for better initialization:
```bash
# Basic SFT
python src/esci_search/trainers/train_sft.py --model qwen-7b

# With accelerate
accelerate launch src/esci_search/trainers/train_sft.py --model qwen-7b
```
Checkpoints saved to: `./sft_trained_models/qwen-7b/checkpoint-{step}/`

#### 3. DPO Training
Train with preference optimization (optionally from SFT checkpoint):
```bash
# DPO from base model
python src/esci_search/trainers/train_dpo.py --model qwen-7b

# DPO from SFT checkpoint (recommended)
python src/esci_search/trainers/train_dpo.py \
    --model qwen-7b \
    --sft_checkpoint_path sft_trained_models/qwen-7b/checkpoint-150

# With accelerate
accelerate launch src/esci_search/trainers/train_dpo.py --model qwen-7b
```
Checkpoints saved to: `./trained_dpo_models/qwen-7b/checkpoint-{step}/`

#### 4. OSPO/GRPO Training

##### OSPO Variants
```bash
# OSPO Proportional
accelerate launch src/esci_search/trainers/train_ospo.py \
    --config src/esci_search/configs/ospo_prop_config.yaml

# OSPO Rank-based
accelerate launch src/esci_search/trainers/train_ospo.py \
    --config src/esci_search/configs/ospo_rank_config.yaml

# OSPO without clipping
accelerate launch src/esci_search/trainers/train_ospo.py \
    --config src/esci_search/configs/ospo_prop_no_clip.yaml
```

##### GRPO Baseline
```bash
accelerate launch src/esci_search/trainers/train_grpo.py \
    --config src/esci_search/configs/grpo_config.yaml
```
##### GRPO-λ Baseline
```bash
accelerate launch src/esci_search/trainers/train_ospo.py \
    --config src/esci_search/configs/grpo_lambda_config.yaml
```
> **Note:** GRPO-λ uses `train_ospo.py` (not `train_grpo.py`) since the eligibility trace implementation is integrated into the OSPO trainer via `use_grpo_lambda: True` in the config.

### Checkpoint Structure

**SFT/DPO Checkpoints:**
```
./sft_trained_models/qwen-7b/
├── checkpoint-50/
├── checkpoint-100/
└── checkpoint-150/

./trained_dpo_models/qwen-7b/
├── checkpoint-15/
└── checkpoint-30/
```

**OSPO/GRPO Checkpoints:**
```
./outputs/ospo_ablations_esci/<MODEL_TAG>/<RUN_NAME>/
├── checkpoint-500/
├── checkpoint-1000/
├── checkpoint-1500/
└── checkpoint-2000/
```

---

## Evaluation

### Step 1: Generate Model Outputs (Inference)
Run inference over the evaluation dataset using stored checkpoints:

**For OSPO/GRPO models:**
```bash
cd src/esci_search/evals
python generate_test.py \
  --base_model "Qwen/Qwen2.5-7B-Instruct" \
  --checkpoint_base "<PATH_TO_RUN_DIR_WITH_CHECKPOINTS>" \
  --dataset_path "../../data/esci/rl_dataset" \
  --output_dir "./test_generations" \
  --temperature 1.0 \
  --batch_size 1024 \
  --max_new_tokens 512
```

**For SFT/DPO models:**
```bash
python generate_test.py \
  --base_model "Qwen/Qwen2.5-7B-Instruct" \
  --checkpoint_base "sft_trained_models/qwen-7b" \
  --dataset_path "../../data/esci/rl_dataset" \
  --output_dir "./test_generations_sft" \
  --temperature 1.0 \
  --batch_size 1024
```

**For base model baseline:**
```bash
python generate_test.py \
  --base_model "Qwen/Qwen2.5-7B-Instruct" \
  --dataset_path "../../data/esci/rl_dataset" \
  --output_dir "./test_generations_base" \
  --temperature 1.0 \
  --batch_size 1024
```

**Notes:**
- `--dataset_path` must contain a "test" split with messages in chat format
- `--checkpoint_base` should contain `checkpoint-*` subdirectories
- Reduce `--batch_size` to 256/128 if OOM occurs
 
 

### Step 2: Score Generations (Retrieval Metrics)

We compute retrieval-based metrics (NDCG, AP, MRR) using FAISS-based evaluation. We report pooled evaluation for the paper's experiments. 

#### (A) Candidate-Filtered Evaluation (With Pools)

This variant restricts scoring to each query's candidate pool (recommended for pooled evaluation).

```bash
python score_search_only.py \
  --input_csv_glob "./test_generations/*.csv" \
  --output_dir "./test_generations_with_pools" \
  --fixed_k 10,20,50,100,1000 \
  --text_cols "expanded_query,response" \
  --batch_size 512 \
  --top_k 1000 \
  --device cuda \
  --faiss_index_path "../../data/esci/index/all-mpnet-base-v2_faiss.bin" \
  --mapping_path "../../data/esci/index/all-mpnet-base-v2_asin_mapping.json" \
  --use_pools true \
  --model_name "sentence-transformers/all-mpnet-base-v2"
```

#### (B) Raw Evaluation (No Pools)

This variant scores against raw FAISS top-K retrieval without candidate filtering.

```bash
python score_search_only.py \
  --input_csv_glob "./test_generations/*.csv" \
  --output_dir "./test_generations_without_pools" \
  --fixed_k 10,20,50,100,1000 \
  --text_cols "expanded_query,response" \
  --batch_size 512 \
  --top_k 1000 \
  --device cuda \
  --faiss_index_path "../../data/esci/index/all-mpnet-base-v2_faiss.bin" \
  --mapping_path "../../data/esci/index/all-mpnet-base-v2_asin_mapping.json" \
  --use_pools false \
  --model_name "sentence-transformers/all-mpnet-base-v2"
```

---

## Reproducing the OSPO vs. SCAR Complexity Plot (Figure 9 in paper)
 
### Requirements
```bash
pip install shap
```

### Run
From the project root directory:
```bash
cd src/hnm_contextual_search
python owen_computation_analysis.py
```

The script reads from `../../data/owen_computation_data/` and saves
`complexity_shap_v4.pdf` and `complexity_shap_v4.png` to the same directory.

## Directory Navigation

All commands assume you're in the appropriate subdirectory:

- **Data processing:** `cd src/esci_search/data_processing`
- **Training:** `cd src/esci_search/trainers`
- **Evaluation:** `cd src/esci_search/evals`

---
 
 

 
