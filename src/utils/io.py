"""I/O utilities for saving and loading results."""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any


def save_results(
    results: dict[str, Any], output_dir: Path, model_name: str, prefix: str = "results"
) -> Path:
    """
    Save results to JSON file with timestamp.

    Args:
        results: Dictionary of results to save
        output_dir: Directory to save to
        model_name: Model name for filename
        prefix: Filename prefix

    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean model name for filename
    clean_name = model_name.replace("/", "_").replace("-", "_")

    filename = f"{prefix}_{clean_name}_{timestamp}.json"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {filepath}")
    return filepath


def load_results(filepath: Path) -> dict[str, Any]:
    """Load results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def save_features(features: dict[str, Any], output_dir: Path, model_name: str, split: str) -> Path:
    """
    Save extracted features to pickle file for reuse.

    Args:
        features: Dictionary containing features, labels, etc.
        output_dir: Directory to save to
        model_name: Model name for filename
        split: Dataset split name (train/val/test)

    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_name = model_name.replace("/", "_").replace("-", "_")
    filename = f"features_{clean_name}_{split}.pkl"
    filepath = output_dir / filename

    with open(filepath, "wb") as f:
        pickle.dump(features, f)

    print(f"Features saved to {filepath}")
    return filepath


def load_features(filepath: Path) -> dict[str, Any]:
    """Load features from pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)
