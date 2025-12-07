"""
Smooth Expected Calibration Error (smECE).

Paper Section 3.1:
"We use a recently improved variant of ECE, smooth ECE (smECE; Błasiok and
Nakkiran, 2024), which replaces histogram binning with Nadaraya-Watson kernel
regression (Nadaraya, 1964; Watson, 1964). A reflected Gaussian kernel is used;
the kernel width is determined automatically from the data, yielding a
consistent estimator."

Paper: "Lower smECE is better"

Paper Results:
"all of the confidence estimators are well-calibrated-their smECE values are
small and not significantly different-except for the raw confidences, which
have smECEs 3-10x higher than the others"
"""

import numpy as np
from scipy.stats import norm


def compute_smece(confidences: np.ndarray, labels: np.ndarray, num_points: int = 1000) -> float:
    """
    Compute Smooth Expected Calibration Error.

    Uses Nadaraya-Watson kernel regression with reflected Gaussian kernel
    to estimate calibration error without binning artifacts.

    Paper: smECE measures |acc(p) - p| averaged over the confidence distribution,
    where acc(p) is the expected accuracy at confidence level p.

    Args:
        confidences: Predicted confidence scores [n_samples]
        labels: Binary correctness labels [n_samples]
        num_points: Number of points for numerical integration

    Returns:
        smece: Smooth expected calibration error (lower is better)
    """
    n = len(confidences)

    if n == 0:
        return 0.0

    # Automatic bandwidth selection using Silverman's rule
    std = np.std(confidences)
    iqr = np.percentile(confidences, 75) - np.percentile(confidences, 25)

    # Silverman's rule of thumb
    bandwidth = 0.9 * min(std, iqr / 1.34) * (n ** (-1 / 5))
    bandwidth = max(bandwidth, 1e-6)  # Ensure positive

    # Integration points in (0, 1) - avoid exact 0 and 1
    p_grid = np.linspace(0.001, 0.999, num_points)

    calibration_errors = []
    weights = []

    for p in p_grid:
        # Reflected Gaussian kernel weights
        # Standard term
        kernel = norm.pdf((confidences - p) / bandwidth)
        # Reflections at boundaries
        kernel += norm.pdf((confidences + p) / bandwidth)
        kernel += norm.pdf((confidences - 2 + p) / bandwidth)
        kernel += norm.pdf((confidences + 2 - p) / bandwidth)

        weight_sum = kernel.sum()

        if weight_sum > 0:
            # Expected accuracy at confidence p (Nadaraya-Watson estimate)
            expected_acc = (kernel * labels).sum() / weight_sum
            # Calibration error at p: |acc(p) - p|
            calibration_errors.append(abs(expected_acc - p))
            # Weight by density at p
            weights.append(weight_sum / n)
        else:
            calibration_errors.append(0)
            weights.append(0)

    # Normalize weights
    weights = np.array(weights)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(num_points) / num_points

    # Weighted average of calibration errors
    smece = np.average(calibration_errors, weights=weights)

    return float(smece)
