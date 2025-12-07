"""
Nadaraya-Watson Kernel Regression (NWKR).

Paper Section 4.4:
"Kernel Regressor (NWKR): Here, rather than using a histogram with fixed bins
to recalibrate, we use Nadaraya-Watson kernel regression (Nadaraya, 1964;
Watson, 1964), following the exact procedure Błasiok and Nakkiran (2024)
used to compute smECE."

"Analogously to above, since this follows the exact same procedure as in smECE,
we should expect it to perform well under that metric."

Uses a reflected Gaussian kernel with automatic bandwidth selection.
"""


import numpy as np
from scipy.stats import norm

from .base import BaseConfidenceEstimator


class NadarayaWatsonKernelRegressor(BaseConfidenceEstimator):
    """
    Nadaraya-Watson kernel regression for confidence calibration.

    Uses a reflected Gaussian kernel with automatic bandwidth selection
    following Błasiok and Nakkiran (2024).

    Paper: "the kernel width is determined automatically from the data,
    yielding a consistent estimator"
    """

    def __init__(self, bandwidth: float | None = None):
        """
        Initialize NWKR.

        Args:
            bandwidth: Kernel bandwidth (auto-selected if None)
        """
        self.bandwidth = bandwidth
        self.train_confidences = None
        self.train_labels = None

    @property
    def name(self) -> str:
        return "NWKR"

    def _select_bandwidth(self, confidences: np.ndarray) -> float:
        """
        Automatic bandwidth selection using Silverman's rule of thumb.

        Paper: "the kernel width is determined automatically from the data"

        This follows the standard approach for kernel density estimation.
        """
        n = len(confidences)
        std = np.std(confidences)
        iqr = np.percentile(confidences, 75) - np.percentile(confidences, 25)

        # Silverman's rule of thumb
        # h = 0.9 * min(std, IQR/1.34) * n^(-1/5)
        bandwidth = 0.9 * min(std, iqr / 1.34) * (n ** (-1 / 5))

        # Ensure positive bandwidth
        return max(bandwidth, 1e-6)

    def _reflected_gaussian_kernel(self, x: np.ndarray, x0: float, h: float) -> np.ndarray:
        """
        Reflected Gaussian kernel for bounded [0, 1] domain.

        Paper: "A reflected Gaussian kernel is used"

        Adds reflected terms at boundaries to reduce bias near edges.
        """
        # Standard Gaussian contribution
        kernel = norm.pdf((x - x0) / h)

        # Reflected terms at boundaries
        # Reflection at 0
        kernel += norm.pdf((x + x0) / h)
        # Reflection at 1
        kernel += norm.pdf((x - 2 + x0) / h)
        kernel += norm.pdf((x + 2 - x0) / h)

        return kernel

    def fit(
        self, features: np.ndarray, labels: np.ndarray, raw_confidences: np.ndarray | None = None
    ) -> "NadarayaWatsonKernelRegressor":
        """
        Store training data for kernel regression.

        NWKR is a non-parametric method that stores training data
        and computes predictions via weighted averaging.

        Args:
            features: Unused for NWKR
            labels: Binary correctness labels
            raw_confidences: Raw confidence scores
        """
        if raw_confidences is None:
            raise ValueError("NWKR requires raw_confidences")

        self.train_confidences = raw_confidences.copy()
        self.train_labels = labels.copy()

        # Auto-select bandwidth if not provided
        if self.bandwidth is None:
            self.bandwidth = self._select_bandwidth(raw_confidences)

        return self

    def predict_proba(
        self, features: np.ndarray, raw_confidences: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Predict using Nadaraya-Watson kernel regression.

        For each test point, compute kernel-weighted average of training labels.

        Args:
            features: Unused for NWKR
            raw_confidences: Raw confidence scores to calibrate

        Returns:
            Calibrated confidence scores
        """
        if raw_confidences is None:
            raise ValueError("NWKR requires raw_confidences")

        calibrated = np.zeros(len(raw_confidences))

        for i, conf in enumerate(raw_confidences):
            # Compute kernel weights for all training points
            weights = self._reflected_gaussian_kernel(self.train_confidences, conf, self.bandwidth)

            # Normalize and compute weighted average
            weight_sum = weights.sum()
            if weight_sum > 0:
                calibrated[i] = (weights * self.train_labels).sum() / weight_sum
            else:
                # Fallback to global mean
                calibrated[i] = self.train_labels.mean()

        return calibrated
