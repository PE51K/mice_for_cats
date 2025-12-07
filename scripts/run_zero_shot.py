#!/usr/bin/env python
"""
Zero-shot MICE evaluation with leave-one-API-out.

Paper Section 5:
"To test MICE's out-of-domain generalization, we simulate encountering new APIs
by holding one out during training. Since there are 50 APIs present in the STE
dataset, we train 50 MICE RF and 50 MICE LR models. Each model is trained on
data from 49 APIs and evaluated solely on the held-out API."

Usage:
    # Run zero-shot evaluation for all 50 APIs
    uv run python scripts/run_zero_shot.py

    # Run for specific APIs only (faster for testing)
    uv run python scripts/run_zero_shot.py --apis WeatherAPI NewsAPI
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env
from dotenv import load_dotenv  # noqa: E402

load_dotenv(project_root / ".env")

# Verify HF token
if not os.environ.get("HF_TOKEN"):
    print("ERROR: HF_TOKEN environment variable not set!")
    sys.exit(1)

# Import from src package
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.config import Config  # noqa: E402
from src.data.dataset import STEDataset  # noqa: E402
from src.data.demo_selector import DemoSelector  # noqa: E402
from src.estimators import MICELogisticRegression, MICERandomForest  # noqa: E402
from src.features.extractor import MICEFeatureExtractor  # noqa: E402
from src.metrics.classification import compute_classification_metrics  # noqa: E402
from src.metrics.etcu import compute_all_etcu_metrics  # noqa: E402
from src.metrics.smece import compute_smece  # noqa: E402
from src.models.llm_wrapper import LLMWrapper  # noqa: E402
from src.utils.io import save_results  # noqa: E402
from src.utils.seed import set_all_seeds  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MICE zero-shot evaluation")
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
        "--output_dir", type=str, default="results_zero_shot", help="Output directory"
    )
    parser.add_argument(
        "--apis",
        nargs="*",
        default=None,
        help="Specific APIs to evaluate (default: all 50 APIs)",
    )
    return parser.parse_args()


def run_zero_shot_for_api(
    api_name: str,
    dataset: STEDataset,
    config: Config,
    llm: LLMWrapper,
    feature_extractor: MICEFeatureExtractor,
    output_dir: Path,
) -> dict:
    """
    Run zero-shot evaluation for a single held-out API.

    Returns dictionary with metrics for this API.
    """
    print(f"\n{'=' * 60}")
    print(f"Zero-Shot Evaluation: Held-Out API = {api_name}")
    print(f"{'=' * 60}")

    # Create splits with this API held out
    demo_set, train_set, _val_set, test_set = dataset.create_leave_one_api_out_splits(
        held_out_api=api_name,
        demo_size=config.data.demo_set_size,
        samples_per_api_train=config.data.samples_per_api_train,
        samples_per_api_val=config.data.samples_per_api_val,
    )

    print(
        f"Splits: demo={len(demo_set)}, train={len(train_set)}, "
        f"val={len(_val_set)}, test={len(test_set)} (all from {api_name})"
    )

    if len(test_set) == 0:
        print(f"WARNING: No test examples found for {api_name}, skipping")
        return None

    # Feature extraction (regenerate every run)
    print("Extracting features...")
    demo_selector = DemoSelector(
        demo_set,
        model_name=config.icl.sentence_transformer_model,
        num_shots=config.icl.num_shots,
    )

    print("  Processing training set (49 APIs)...")
    train_data = feature_extractor.extract_batch(
        train_set,
        demo_selector,
        max_new_tokens=config.max_new_tokens,
        desc=f"Train ({api_name} held out)",
    )

    print(f"  Processing test set ({api_name} only)...")
    test_data = feature_extractor.extract_batch(
        test_set,
        demo_selector,
        max_new_tokens=config.max_new_tokens,
        desc=f"Test ({api_name})",
    )

    print(
        f"\n  Train accuracy: {train_data['labels'].mean():.2%} "
        f"({train_data['labels'].sum()}/{len(train_data['labels'])} correct)"
    )
    print(
        f"  Test accuracy: {test_data['labels'].mean():.2%} "
        f"({test_data['labels'].sum()}/{len(test_data['labels'])} correct)"
    )

    # Train MICE models (only LR and RF for zero-shot)
    estimators = [
        MICELogisticRegression(C=config.mice.lr_C, zero_shot=True, random_state=config.seed),
        MICERandomForest(
            n_estimators=config.mice.rf_n_estimators,
            max_depth=config.mice.rf_max_depth,
            max_features=config.mice.rf_max_features,
            random_state=config.seed,
            zero_shot=True,
        ),
    ]

    results = {}
    for estimator in estimators:
        print(f"\n  Training {estimator.name}...")
        estimator.fit(train_data["features"], train_data["labels"], train_data["raw_confidences"])

        test_confidences = estimator.predict_proba(
            test_data["features"], test_data["raw_confidences"]
        )

        # Compute metrics
        smece = compute_smece(test_confidences, test_data["labels"])
        etcu_metrics = compute_all_etcu_metrics(test_confidences, test_data["labels"])
        class_metrics = compute_classification_metrics(test_confidences, test_data["labels"])

        results[estimator.name] = {"smece": smece, **etcu_metrics, **class_metrics}

        print(f"    smECE: {smece:.4f}, ETCU AUC: {etcu_metrics['etcu_auc']:.4f}")

    return {
        "api": api_name,
        "test_size": len(test_set),
        "test_accuracy": float(test_data["labels"].mean()),
        "results": results,
    }


def main():
    """Main zero-shot evaluation function."""
    args = parse_args()

    print("=" * 60)
    print("MICE Zero-Shot Evaluation (Leave-One-API-Out)")
    print("=" * 60)

    # Set seeds
    set_all_seeds(args.seed)

    # Load config
    config = Config()
    config.model.model_name = args.model
    config.seed = args.seed

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    data_dir = Path(__file__).parent.parent / config.data_dir
    dataset = STEDataset(data_dir, seed=config.seed)

    # Get list of APIs to evaluate
    all_apis = dataset.get_api_list()
    apis_to_evaluate = args.apis if args.apis else all_apis

    print("\nConfiguration:")
    print(f"  Model: {config.model.model_name}")
    print(f"  Total APIs in dataset: {len(all_apis)}")
    print(f"  APIs to evaluate: {len(apis_to_evaluate)}")
    print(f"  Output dir: {output_dir}")

    print("\nInitializing models...")
    llm = LLMWrapper(config.model.model_name)
    feature_extractor = MICEFeatureExtractor(llm, bertscore_model=config.model.bertscore_model)

    # Run zero-shot evaluation for each API
    all_results = []
    for api_name in tqdm(apis_to_evaluate, desc="APIs"):
        result = run_zero_shot_for_api(
            api_name=api_name,
            dataset=dataset,
            config=config,
            llm=llm,
            feature_extractor=feature_extractor,
            output_dir=output_dir,
        )
        if result is not None:
            all_results.append(result)

    # Aggregate results
    print("\n" + "=" * 60)
    print("Aggregated Zero-Shot Results")
    print("=" * 60)

    # Combine predictions from all APIs
    combined_metrics = {"MICE LR (zero-shot)": [], "MICE RF (zero-shot)": []}

    for result in all_results:
        for estimator_name in combined_metrics.keys():
            if estimator_name in result["results"]:
                combined_metrics[estimator_name].append(result["results"][estimator_name])

    # Average metrics across all APIs
    aggregated_results = {}
    for estimator_name, metrics_list in combined_metrics.items():
        if not metrics_list:
            continue

        # Average each metric
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        aggregated_results[estimator_name] = aggregated

    # Print summary
    print(f"\nEvaluated {len(all_results)} APIs")
    print(f"\n{'Estimator':<30} {'smECE':>12} {'ETCU AUC':>12} {'ROC AUC':>12}")
    print("-" * 70)
    for name, metrics in aggregated_results.items():
        print(
            f"{name:<30} "
            f"{metrics['smece']['mean']:>12.4f} "
            f"{metrics['etcu_auc']['mean']:>12.4f} "
            f"{metrics['roc_auc']['mean']:>12.4f}"
        )

    # Save results
    final_results = {
        "model": config.model.model_name,
        "seed": config.seed,
        "timestamp": datetime.now().isoformat(),
        "num_apis_evaluated": len(all_results),
        "config": {
            "num_shots": config.icl.num_shots,
            "mice_lr_C": config.mice.lr_C,
            "mice_rf_n_estimators": config.mice.rf_n_estimators,
            "mice_rf_max_depth": config.mice.rf_max_depth,
            "mice_rf_max_features": config.mice.rf_max_features,
        },
        "per_api_results": all_results,
        "aggregated_results": aggregated_results,
    }

    results_path = save_results(
        final_results, output_dir, config.model.model_name, prefix="zero_shot"
    )

    print(f"\nResults saved to: {results_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
