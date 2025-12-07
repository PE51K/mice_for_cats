# MICE for CATs Reproduction

Reproduction of "MICE for CATs: Model-Internal Confidence Estimation for Calibrating Agents with Tools"

## Setup

```bash
# Navigate to project
cd tools_uncertainty/mice_for_cats

# Install dependencies
uv sync

# Setup HuggingFace authentication (required for Llama models)
# 1. Get token from https://huggingface.co/settings/tokens
# 2. Accept Llama license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
cp env.template .env
# Edit .env and add your HF_TOKEN
```

## Run Experiments

### Standard Evaluation (Paper Table 1)

```bash
# Run with Llama-3.1-8B-Instruct (default, 32 layers)
uv run python scripts/run_pipeline.py

# Run with Llama-3.2-3B-Instruct (28 layers)
uv run python scripts/run_pipeline.py --model meta-llama/Llama-3.2-3B-Instruct

# Debug run with limited samples
uv run python scripts/run_pipeline.py --max_samples 50

# Skip generation and use cached features
uv run python scripts/run_pipeline.py --skip_generation
```

### Zero-Shot Evaluation (Paper Section 5)

For true zero-shot evaluation with leave-one-API-out:

```bash
# Run zero-shot evaluation for all 50 APIs (takes ~50x longer)
uv run python scripts/run_zero_shot.py

# Run for specific APIs only (faster)
uv run python scripts/run_zero_shot.py --apis WeatherAPI NewsAPI StockAPI

# With different model
uv run python scripts/run_zero_shot.py --model meta-llama/Llama-3.2-3B-Instruct
```

**Note**: Zero-shot evaluation trains 50 separate models (one per held-out API) and can take significantly longer than standard evaluation.

## Results

Results are saved to:
- `results/` directory for standard evaluation
- `results_zero_shot/` directory for zero-shot evaluation

Each run creates a JSON file with:
- smECE (smooth Expected Calibration Error)
- ETCU at τ=0.1, 0.5, 0.9 (Expected Tool-Calling Utility)
- ETCU AUC (area under utility curve)
- Classification metrics (ROC AUC, F1, etc.)
