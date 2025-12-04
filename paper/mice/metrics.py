"""Metrics for evaluating confidence estimators: smooth ECE and ETCU."""

import numpy as np
from typing import Tuple, Optional
from scipy import stats


def smooth_ece(
    confidences: np.ndarray,
    correct: np.ndarray,
    bandwidth: Optional[float] = None,
) -> float:
    """
    Compute smooth Expected Calibration Error (smECE).
    
    Uses Nadaraya-Watson kernel regression with reflected Gaussian kernel.
    
    Args:
        confidences: Predicted confidence scores [n_samples]
        correct: Binary correctness labels [n_samples] (1 = correct, 0 = incorrect)
        bandwidth: Kernel bandwidth (if None, determined automatically)
        
    Returns:
        Smooth ECE value
    """
    n = len(confidences)
    
    # Determine bandwidth if not provided
    if bandwidth is None:
        std = np.std(confidences)
        bandwidth = 1.06 * std * (n ** (-1.0 / 5.0))
        # Ensure minimum bandwidth
        bandwidth = max(bandwidth, 0.01)
    
    # Compute expected calibration error using kernel regression
    ece = 0.0
    
    for i in range(n):
        p = confidences[i]
        
        # Compute kernel weights using reflected Gaussian kernel
        distances = np.abs(confidences - p)
        distances_reflected_0 = np.abs(confidences + p)  # Reflection at 0
        distances_reflected_1 = np.abs(2 - confidences - p)  # Reflection at 1
        
        min_distances = np.minimum(distances, np.minimum(distances_reflected_0, distances_reflected_1))
        
        # Gaussian kernel
        weights = np.exp(-0.5 * (min_distances / bandwidth) ** 2)
        weights_sum = np.sum(weights)
        
        if weights_sum > 0:
            weights = weights / weights_sum
            # Expected accuracy at this confidence level
            expected_accuracy = np.sum(weights * correct)
            # Calibration error
            ece += weights[i] * np.abs(expected_accuracy - p)
    
    return ece


def expected_tool_calling_utility(
    confidences: np.ndarray,
    correct: np.ndarray,
    threshold: float,
    tp: float = 1.0,
    fp: float = -1.0,
    tn: float = 0.0,
    fn: float = 0.0,
) -> float:
    """
    Compute expected tool-calling utility (ETCU) at a given threshold.
    
    Args:
        confidences: Predicted confidence scores [n_samples]
        correct: Binary correctness labels [n_samples] (1 = correct, 0 = incorrect)
        threshold: Confidence threshold for execution decision
        tp: Utility of true positive (default: 1.0)
        fp: Utility of false positive (default: -1.0)
        tn: Utility of true negative (default: 0.0)
        fn: Utility of false negative (default: 0.0)
        
    Returns:
        Expected utility
    """
    # Decision: execute if confidence >= threshold
    execute = confidences >= threshold
    
    # Compute utilities
    utilities = np.zeros(len(confidences))
    
    # True positives: execute and correct
    utilities[(execute) & (correct == 1)] = tp
    
    # False positives: execute and incorrect
    utilities[(execute) & (correct == 0)] = fp
    
    # True negatives: don't execute and incorrect
    utilities[(~execute) & (correct == 0)] = tn
    
    # False negatives: don't execute and correct
    utilities[(~execute) & (correct == 1)] = fn
    
    return np.mean(utilities)


def compute_etcu_curve(
    confidences: np.ndarray,
    correct: np.ndarray,
    tp: float = 1.0,
    fp: float = -1.0,
    tn: float = 0.0,
    fn: float = 0.0,
    num_points: int = 999,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ETCU curve across all thresholds.
    
    Args:
        confidences: Predicted confidence scores [n_samples]
        correct: Binary correctness labels [n_samples]
        tp: Utility of true positive
        fp: Utility of false positive
        tn: Utility of true negative
        fn: Utility of false negative
        num_points: Number of threshold points to evaluate
        
    Returns:
        Tuple of (thresholds, utilities)
    """
    thresholds = np.linspace(0.001, 0.999, num_points)
    utilities = np.zeros(num_points)
    
    for i, threshold in enumerate(thresholds):
        utilities[i] = expected_tool_calling_utility(
            confidences, correct, threshold, tp, fp, tn, fn
        )
    
    return thresholds, utilities


def compute_auc_etcu(
    confidences: np.ndarray,
    correct: np.ndarray,
    tp: float = 1.0,
    fp: float = -1.0,
    tn: float = 0.0,
    fn: float = 0.0,
) -> float:
    """
    Compute area under the ETCU curve (AUC-ETCU).
    
    Args:
        confidences: Predicted confidence scores [n_samples]
        correct: Binary correctness labels [n_samples]
        tp: Utility of true positive
        fp: Utility of false positive
        tn: Utility of true negative
        fn: Utility of false negative
        
    Returns:
        AUC-ETCU value
    """
    thresholds, utilities = compute_etcu_curve(confidences, correct, tp, fp, tn, fn)
    
    # Average utility across all thresholds
    auc = np.mean(utilities)
    
    return auc


def get_risk_settings() -> dict:
    """
    Get predefined risk settings from the paper.
    
    Returns:
        Dictionary with risk level settings
    """
    return {
        'low': {
            'fp': -1.0 / 9.0,
            'tp': 1.0,
            'tn': 0.0,
            'fn': 0.0,
            'threshold': 0.1,
        },
        'medium': {
            'fp': -1.0,
            'tp': 1.0,
            'tn': 0.0,
            'fn': 0.0,
            'threshold': 0.5,
        },
        'high': {
            'fp': -9.0,
            'tp': 1.0,
            'tn': 0.0,
            'fn': 0.0,
            'threshold': 0.9,
        },
    }


def evaluate_confidence_estimator(
    confidences: np.ndarray,
    correct: np.ndarray,
) -> dict:
    """
    Evaluate a confidence estimator using all metrics.
    
    Args:
        confidences: Predicted confidence scores [n_samples]
        correct: Binary correctness labels [n_samples]
        
    Returns:
        Dictionary of metric values
    """
    results = {}
    
    # Smooth ECE
    results['smECE'] = smooth_ece(confidences, correct)
    
    # ETCU at different risk levels
    risk_settings = get_risk_settings()
    for risk_level, settings in risk_settings.items():
        etcu = expected_tool_calling_utility(
            confidences,
            correct,
            settings['threshold'],
            settings['tp'],
            settings['fp'],
            settings['tn'],
            settings['fn'],
        )
        results[f'ETCU_{risk_level}'] = etcu
    
    # AUC-ETCU
    results['AUC_ETCU'] = compute_auc_etcu(confidences, correct)
    
    return results

