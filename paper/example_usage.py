"""Example usage of MICE implementation."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mice.train import train_mice


if __name__ == "__main__":
    # Example: Train MICE on STE dataset
    data_path = "../data/simulated-trial-and-error/STE/saved_results/gpt35.json"
    
    train_mice(
        data_path=data_path,
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        output_dir="outputs",
        max_examples=100,  # Limit for testing
        device="cuda",  # or "cpu"
    )

