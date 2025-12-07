"""I/O utilities for saving and loading results."""

import json
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
