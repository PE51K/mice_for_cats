"""
Standard classification metrics for confidence estimators.

Additional metrics beyond those in the paper for comprehensive evaluation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    confidences: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """
    Compute standard classification metrics.

    These complement the paper's primary metrics (smECE, ETCU) with
    traditional classification evaluation.

    Args:
        confidences: Predicted confidence scores [n_samples]
        labels: Binary correctness labels [n_samples]
        threshold: Classification threshold for binary metrics (default 0.5)

    Returns:
        Dictionary of classification metrics
    """
    # Binary predictions at threshold
    predictions = (confidences > threshold).astype(int)

    metrics = {}

    # Basic classification metrics
    metrics["accuracy"] = float(accuracy_score(labels, predictions))
    metrics["precision"] = float(precision_score(labels, predictions, zero_division=0))
    metrics["recall"] = float(recall_score(labels, predictions, zero_division=0))
    metrics["f1"] = float(f1_score(labels, predictions, zero_division=0))

    # Brier score (measures calibration + discrimination)
    # Lower is better, 0 is perfect
    metrics["brier_score"] = float(brier_score_loss(labels, confidences))

    # ROC AUC (discrimination ability)
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, confidences))
    except ValueError:
        # Handle case where only one class is present
        metrics["roc_auc"] = 0.5

    # Average Precision (PR AUC approximation)
    try:
        metrics["avg_precision"] = float(average_precision_score(labels, confidences))
    except ValueError:
        metrics["avg_precision"] = labels.mean()

    # Log loss (cross-entropy)
    # Clip confidences to avoid log(0)
    clipped = np.clip(confidences, 1e-15, 1 - 1e-15)
    metrics["log_loss"] = float(log_loss(labels, clipped))

    return metrics


def compute_confusion_matrix_counts(
    confidences: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> dict[str, int]:
    """
    Compute confusion matrix counts.

    Args:
        confidences: Predicted confidence scores [n_samples]
        labels: Binary correctness labels [n_samples]
        threshold: Classification threshold

    Returns:
        Dictionary with TP, TN, FP, FN counts
    """
    predictions = (confidences > threshold).astype(int)

    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    return {
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }
