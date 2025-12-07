# MICE for CATs: Model-Internal Confidence Estimators

Implementation of Model-Internal Confidence Estimators (MICE) for calibrating tool-calling agents, as described in the paper "MICE for CATs: Model-Internal Confidence Estimation for Calibrating Agents with Tools".

## Overview

MICE extracts features from intermediate layers of transformer language models using logit lens decoding, computes BERTScore similarities between layer outputs and the final output, and trains a classifier to predict confidence in tool calls.

## Installation

Using UV:

```bash
uv sync
```

## Usage

### Basic Usage

**Run on full dataset** (recommended for production):

```bash
uv run python -m mice.train \
    --data_path ../data/simulated-trial-and-error/STE/saved_results/gpt35.json \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir outputs
```

**Run on subset** (for testing/debugging):

```bash
uv run python -m mice.train \
    --data_path ../data/simulated-trial-and-error/STE/saved_results/gpt35.json \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir outputs \
    --max_examples 1000
```

### Command-line Arguments

- `data_path`: Path to STE dataset JSON file (required)
- `model_name`: HuggingFace model name (default: `meta-llama/Meta-Llama-3-8B-Instruct`)
- `output_dir`: Directory to save outputs (default: `outputs`)
- `max_examples`: Maximum examples to process (default: None = all)
- `train_ratio`: Proportion for training set (default: 0.6)
- `val_ratio`: Proportion for validation set (default: 0.2)
- `bertscore_model`: BERTScore model name (default: `microsoft/deberta-xlarge-mnli`)
- `device`: Device to use (`cuda` or `cpu`, default: auto-detect)
- `use_cache`: Whether to use cached features if available (default: True)

### Example

```python
from mice.train import train_mice

train_mice(
    data_path="../data/simulated-trial-and-error/STE/saved_results/gpt35.json",
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    output_dir="outputs",
    max_examples=1000,
)
```

## Components

- `mice/feature_extraction.py`: Logit lens decoding and feature extraction
- `mice/confidence.py`: Raw confidence calculation
- `mice/models.py`: MICE classifier implementations
- `mice/metrics.py`: Smooth ECE and ETCU metrics
- `mice/data.py`: Data loading and processing
- `mice/train.py`: Training script

