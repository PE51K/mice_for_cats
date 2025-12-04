"""Main training script for MICE."""

import json
from pathlib import Path
from typing import Optional

import fire
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import prepare_dataset, split_dataset
from .metrics import evaluate_confidence_estimator
from .models import (
    HistogramRegressionEstimator,
    MICELogisticRegression,
    MICERandomForest,
    NadarayaWatsonKernelRegressor,
)


def train_mice(
    data_path: str,
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    output_dir: str = "outputs",
    max_examples: Optional[int] = None,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    bertscore_model: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_cache: bool = True,
):
    """
    Train MICE models on STE dataset.

    Args:
        data_path: Path to STE dataset JSON file
        model_name: HuggingFace model name
        output_dir: Directory to save outputs
        max_examples: Maximum examples to process (None = all)
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        bertscore_model: BERTScore model name (default: microsoft/deberta-xlarge-mnli)
        device: Device to use
        use_cache: Whether to use cached features if available
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)

    # Get the actual device the model is on
    if device == "cuda" and torch.cuda.is_available():
        # If using device_map, get the device from the first parameter
        first_param = next(model.parameters())
        device = str(first_param.device)

    model.eval()

    # Check for cached features
    cache_file = output_dir / "features_cache.npz"
    examples_log_file = output_dir / "examples_log.json"
    if use_cache and cache_file.exists():
        print("Loading cached features...")
        cache_data = np.load(cache_file)
        features = cache_data["features"]
        labels = cache_data["labels"]
        raw_confidences = cache_data["raw_confidences"]

        # Try to load examples log if it exists
        if examples_log_file.exists():
            with open(examples_log_file, "r", encoding="utf-8") as f:
                examples_log = json.load(f)
            print(
                f"Examples log loaded from {examples_log_file} ({len(examples_log)} examples)"
            )
        else:
            examples_log = []
            print(
                "Note: Examples log not found in cache. Run without --use_cache to generate it."
            )
    else:
        # Prepare dataset
        print("Extracting MICE features...")
        features, labels, raw_confidences, examples_log = prepare_dataset(
            data_path,
            model,
            tokenizer,
            max_examples=max_examples,
            bertscore_model=bertscore_model,
            device=device,
        )

        # Save cache
        if use_cache:
            np.savez(
                cache_file,
                features=features,
                labels=labels,
                raw_confidences=raw_confidences,
            )

        # Save examples log (model outputs and ground truth)
        with open(examples_log_file, "w", encoding="utf-8") as f:
            json.dump(examples_log, f, indent=2, ensure_ascii=False)
        print(
            f"Examples log saved to {examples_log_file} ({len(examples_log)} examples)"
        )

    print(f"Dataset size: {len(labels)}")
    print(f"Feature shape: {features.shape}")
    print(f"Accuracy: {np.mean(labels):.3f}")

    # Split dataset
    train, val, test = split_dataset(
        features, labels, raw_confidences, train_ratio, val_ratio
    )

    train_features, train_labels, train_raw_conf = train
    val_features, val_labels, val_raw_conf = val
    test_features, test_labels, test_raw_conf = test

    print(
        f"Train: {len(train_labels)}, Val: {len(val_labels)}, Test: {len(test_labels)}"
    )

    # Train models
    results = {}

    # 1. Raw Confidence (no training needed)
    print("\nEvaluating Raw Confidence...")
    raw_conf_test = test_raw_conf
    results["raw_confidence"] = evaluate_confidence_estimator(
        raw_conf_test, test_labels
    )

    # 2. Histogram Regression Estimator (HRE)
    print("\nTraining HRE...")
    hre = HistogramRegressionEstimator(n_bins=25)
    hre.fit(train_raw_conf.reshape(-1, 1), train_labels)
    hre_conf_test = hre.predict_confidence(test_raw_conf.reshape(-1, 1))
    results["HRE"] = evaluate_confidence_estimator(hre_conf_test, test_labels)

    # 3. Nadaraya-Watson Kernel Regressor (NWKR)
    print("\nTraining NWKR...")
    nwkr = NadarayaWatsonKernelRegressor()
    nwkr.fit(train_raw_conf.reshape(-1, 1), train_labels)
    nwkr_conf_test = nwkr.predict_confidence(test_raw_conf.reshape(-1, 1))
    results["NWKR"] = evaluate_confidence_estimator(nwkr_conf_test, test_labels)

    # 4. MICE Logistic Regression
    print("\nTraining MICE LR...")
    mice_lr = MICELogisticRegression(l2_regularization=2.0)
    mice_lr.fit(train_features, train_labels)
    mice_lr_conf_test = mice_lr.predict_confidence(test_features)
    results["MICE_LR"] = evaluate_confidence_estimator(mice_lr_conf_test, test_labels)

    # 5. MICE Random Forest
    print("\nTraining MICE RF...")
    mice_rf = MICERandomForest(n_estimators=1000, max_depth=20, max_features=10)
    mice_rf.fit(train_features, train_labels)
    mice_rf_conf_test = mice_rf.predict_confidence(test_features)
    results["MICE_RF"] = evaluate_confidence_estimator(mice_rf_conf_test, test_labels)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    metric_names = ["smECE", "ETCU_low", "ETCU_medium", "ETCU_high", "AUC_ETCU"]

    print(f"\n{'Method':<15} " + " ".join(f"{m:>12}" for m in metric_names))
    print("-" * 80)

    for method, metrics in results.items():
        print(
            f"{method:<15} "
            + " ".join(f"{metrics.get(m, 0):>12.4f}" for m in metric_names)
        )

    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Save models (optional - would need pickle/joblib)
    print("\nTraining complete!")


def main():
    """Entry point for command-line interface."""
    fire.Fire(train_mice)


if __name__ == "__main__":
    main()
