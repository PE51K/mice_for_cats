#!/usr/bin/env python
"""
Full MICE for CATs reproduction pipeline.

This script runs the complete experiment:
1. Load STE dataset and create splits
2. Generate tool calls and extract MICE features
3. Train all confidence estimators
4. Evaluate on test set with all metrics

Requires HF_TOKEN environment variable for Llama model access.

Usage:
    # Set HF token
    export HF_TOKEN=your_token_here
    # OR create .env file
    cp env.template .env && edit .env

    # Run with default model (Llama-3.1-8B-Instruct)
    uv run python scripts/run_pipeline.py

    # Run with specific model
    uv run python scripts/run_pipeline.py --model meta-llama/Llama-3.2-3B-Instruct
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path so we can import from src package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env before other imports that might need HF_TOKEN
from dotenv import load_dotenv  # noqa: E402

load_dotenv(project_root / ".env")

# Verify HF token
if not os.environ.get("HF_TOKEN"):
    print("=" * 60)
    print("ERROR: HF_TOKEN environment variable not set!")
    print("=" * 60)
    print("\nPlease either:")
    print("  1. Set HF_TOKEN environment variable:")
    print("     export HF_TOKEN=your_token")
    print("\n  2. Create a .env file:")
    print("     cp env.template .env")
    print("     # Then edit .env with your token")
    print("\nGet your token from: https://huggingface.co/settings/tokens")
    print("You must also accept the Llama model license on HuggingFace")
    print("=" * 60)
    sys.exit(1)

# Import from src package
from src.config import Config  # noqa: E402
from src.data.dataset import STEDataset  # noqa: E402
from src.data.demo_selector import DemoSelector  # noqa: E402
from src.estimators import (  # noqa: E402
    HistogramRegressionEstimator,
    MICELogisticRegression,
    MICERandomForest,
    NadarayaWatsonKernelRegressor,
    RawConfidenceEstimator,
)
from src.features.extractor import MICEFeatureExtractor  # noqa: E402
from src.metrics.classification import compute_classification_metrics  # noqa: E402
from src.metrics.etcu import compute_all_etcu_metrics  # noqa: E402
from src.metrics.smece import compute_smece  # noqa: E402
from src.models.llm_wrapper import LLMWrapper  # noqa: E402
from src.utils.io import save_results  # noqa: E402
from src.utils.seed import set_all_seeds  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MICE for CATs reproduction pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        choices=[
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
        ],
        help="LLM model to use",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples (for debugging)",
    )
    parser.add_argument(
        "--debug_examples",
        type=int,
        default=0,
        help="Print debug info for first N examples (0 to disable)",
    )
    return parser.parse_args()


def main():
    """Main pipeline function."""
    args = parse_args()

    print("=" * 60)
    print("MICE for CATs Reproduction Pipeline")
    print("=" * 60)

    # Set seeds for reproducibility
    set_all_seeds(args.seed)

    # Load config
    config = Config()
    config.model.model_name = args.model
    config.seed = args.seed

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get data directory (relative to script location)
    data_dir = Path(__file__).parent.parent / config.data_dir

    print("\nConfiguration:")
    print(f"  Model: {config.model.model_name}")
    print(f"  Seed: {config.seed}")
    print(f"  Data dir: {data_dir}")
    print(f"  Output dir: {output_dir}")

    # Load dataset
    print("\n" + "=" * 60)
    print("Loading STE Dataset")
    print("=" * 60)

    dataset = STEDataset(data_dir, seed=config.seed)
    demo_set, train_set, val_set, test_set = dataset.create_splits(
        demo_size=config.data.demo_set_size,
        samples_per_api_train=config.data.samples_per_api_train,
        samples_per_api_val=config.data.samples_per_api_val,
    )

    # Limit samples if debugging
    if args.max_samples:
        train_set = train_set[: args.max_samples]
        val_set = val_set[: min(args.max_samples // 2, len(val_set))]
        test_set = test_set[: min(args.max_samples // 2, len(test_set))]
        print(
            f"\nDEBUG: Limited to {len(train_set)} train, "
            f"{len(val_set)} val, {len(test_set)} test samples"
        )

    # Feature extraction (always regenerate to match paper setup)
    print("\n" + "=" * 60)
    print("Initializing Models")
    print("=" * 60)

    llm = LLMWrapper(config.model.model_name)
    demo_selector = DemoSelector(
        demo_set,
        model_name=config.icl.sentence_transformer_model,
        num_shots=config.icl.num_shots,
    )
    feature_extractor = MICEFeatureExtractor(llm, bertscore_model=config.model.bertscore_model)

    print("\n" + "=" * 60)
    print("Extracting Features")
    print("=" * 60)

    print("\nProcessing training set...")
    train_data = feature_extractor.extract_batch(
        train_set,
        demo_selector,
        max_new_tokens=config.max_new_tokens,
        desc="Train",
        debug_first_n=args.debug_examples,
    )

    print("\nProcessing validation set...")
    val_data = feature_extractor.extract_batch(
        val_set,
        demo_selector,
        max_new_tokens=config.max_new_tokens,
        desc="Val",
        debug_first_n=args.debug_examples,
    )

    print("\nProcessing test set...")
    test_data = feature_extractor.extract_batch(
        test_set,
        demo_selector,
        max_new_tokens=config.max_new_tokens,
        desc="Test",
        debug_first_n=args.debug_examples,
    )

    # Print dataset statistics
    print("\nDataset Statistics:")
    print(
        f"  Train: {len(train_data['labels'])} samples, {train_data['labels'].mean():.2%} accuracy"
    )
    print(f"  Val: {len(val_data['labels'])} samples, {val_data['labels'].mean():.2%} accuracy")
    print(f"  Test: {len(test_data['labels'])} samples, {test_data['labels'].mean():.2%} accuracy")

    # Initialize estimators
    print("\n" + "=" * 60)
    print("Training Confidence Estimators")
    print("=" * 60)

    # Paper Section 4.4: All baseline and MICE estimators
    # Note: For true zero-shot evaluation (leave-one-API-out), use run_zero_shot.py
    estimators = [
        RawConfidenceEstimator(),
        HistogramRegressionEstimator(num_bins=config.hre.num_bins),
        NadarayaWatsonKernelRegressor(),
        MICELogisticRegression(C=config.mice.lr_C, random_state=config.seed),
        MICERandomForest(
            n_estimators=config.mice.rf_n_estimators,
            max_depth=config.mice.rf_max_depth,
            max_features=config.mice.rf_max_features,
            random_state=config.seed,
        ),
    ]

    # Train and evaluate each estimator
    results = {}

    for estimator in estimators:
        print(f"\n--- {estimator.name} ---")

        # Train on training data
        print("  Training...")
        estimator.fit(train_data["features"], train_data["labels"], train_data["raw_confidences"])

        # Predict on test set
        print("  Predicting...")
        test_confidences = estimator.predict_proba(
            test_data["features"], test_data["raw_confidences"]
        )

        # Compute metrics
        print("  Computing metrics...")
        smece = compute_smece(test_confidences, test_data["labels"])
        etcu_metrics = compute_all_etcu_metrics(test_confidences, test_data["labels"])
        class_metrics = compute_classification_metrics(test_confidences, test_data["labels"])

        # Store results
        results[estimator.name] = {"smece": smece, **etcu_metrics, **class_metrics}

        print("  Results:")
        print(f"    smECE: {smece:.4f}")
        print(f"    ETCU (τ=0.1): {etcu_metrics['etcu_low_risk']:.4f}")
        print(f"    ETCU (τ=0.5): {etcu_metrics['etcu_medium_risk']:.4f}")
        print(f"    ETCU (τ=0.9): {etcu_metrics['etcu_high_risk']:.4f}")
        print(f"    ETCU AUC: {etcu_metrics['etcu_auc']:.4f}")
        print(f"    ROC AUC: {class_metrics['roc_auc']:.4f}")

    # Save results
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)

    final_results = {
        "model": config.model.model_name,
        "seed": config.seed,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_shots": config.icl.num_shots,
            "hre_bins": config.hre.num_bins,
            "mice_lr_C": config.mice.lr_C,
            "mice_rf_n_estimators": config.mice.rf_n_estimators,
            "mice_rf_max_depth": config.mice.rf_max_depth,
            "mice_rf_max_features": config.mice.rf_max_features,
        },
        "dataset_stats": {
            "train_size": len(train_data["labels"]),
            "val_size": len(val_data["labels"]),
            "test_size": len(test_data["labels"]),
            "train_accuracy": float(train_data["labels"].mean()),
            "test_accuracy": float(test_data["labels"].mean()),
        },
        "results": results,
    }

    results_path = save_results(final_results, output_dir, config.model.model_name)

    # Print summary table
    print("\n" + "=" * 60)
    print("Results Summary (Test Set)")
    print("=" * 60)
    print(f"\n{'Estimator':<30} {'smECE':>8} {'ETCU(0.5)':>10} {'AUC':>8}")
    print("-" * 60)
    for name, metrics in results.items():
        print(
            f"{name:<30} {metrics['smece']:>8.4f} "
            f"{metrics['etcu_medium_risk']:>10.4f} {metrics['etcu_auc']:>8.4f}"
        )

    print(f"\nResults saved to: {results_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
