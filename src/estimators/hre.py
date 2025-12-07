"""
Histogram Regression Estimator (HRE).

Paper Section 4.4:
"Histogram Regression Estimator (HRE; Nobel, 1996): For our second (stronger) 
baseline, we use a standard method to calibrate the previous baseline."

"We use the training set to construct a histogram binned by raw confidence scores. 
We use 25 bins: [0, 0.04), [0.04, 0.08), ..., [0.96, 1.0]. 
To map from a raw confidence score c to a recalibrated estimate p, 
we look up c's bin, and return the percentage of examples in that bin that are correct."

"Note that this is the same histogram construction used to calculate traditional ECE 
(except here constructed on the training set), and so should be expected to perform 
well on ECE metrics."
"""

import numpy as np
from typing import Optional

from .base import BaseConfidenceEstimator


class HistogramRegressionEstimator(BaseConfidenceEstimator):
    """
    Histogram-based calibration of raw confidences.
    
    Paper hyperparameters:
    - num_bins: 25
    - bin_width: 0.04
    - bins: [0, 0.04), [0.04, 0.08), ..., [0.96, 1.0]
    """
    
    def __init__(self, num_bins: int = 25):
        """
        Initialize HRE.
        
        Args:
            num_bins: Number of histogram bins (paper: 25)
        """
        self.num_bins = num_bins
        self.bin_edges = np.linspace(0, 1, num_bins + 1)
        self.bin_accuracies = np.zeros(num_bins)
        self.bin_counts = np.zeros(num_bins)
        self._global_accuracy = 0.5  # Fallback for empty bins
    
    @property
    def name(self) -> str:
        return "HRE"
    
    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        raw_confidences: Optional[np.ndarray] = None
    ) -> "HistogramRegressionEstimator":
        """
        Fit histogram bins based on raw confidence scores.
        
        Computes the accuracy within each confidence bin to create
        a calibration mapping.
        
        Args:
            features: Unused for HRE
            labels: Binary correctness labels
            raw_confidences: Raw confidence scores to bin
        """
        if raw_confidences is None:
            raise ValueError("HRE requires raw_confidences")
        
        # Store global accuracy for empty bin fallback
        self._global_accuracy = labels.mean()
        
        # Reset bins
        self.bin_accuracies = np.zeros(self.num_bins)
        self.bin_counts = np.zeros(self.num_bins)
        
        # Accumulate counts and correct predictions per bin
        for conf, label in zip(raw_confidences, labels):
            # Find bin index (handle edge case of conf=1.0)
            bin_idx = min(int(conf * self.num_bins), self.num_bins - 1)
            self.bin_counts[bin_idx] += 1
            self.bin_accuracies[bin_idx] += label
        
        # Convert sums to means (accuracies)
        for i in range(self.num_bins):
            if self.bin_counts[i] > 0:
                self.bin_accuracies[i] /= self.bin_counts[i]
            else:
                # Default to global accuracy for empty bins
                self.bin_accuracies[i] = self._global_accuracy
        
        return self
    
    def predict_proba(
        self,
        features: np.ndarray,
        raw_confidences: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Map raw confidence to calibrated probability via histogram lookup.
        
        Paper: "To map from a raw confidence score c to a recalibrated 
        estimate p, we look up c's bin, and return the percentage of 
        examples in that bin that are correct."
        
        Args:
            features: Unused for HRE
            raw_confidences: Raw confidence scores to calibrate
        
        Returns:
            Calibrated confidence scores
        """
        if raw_confidences is None:
            raise ValueError("HRE requires raw_confidences")
        
        calibrated = np.zeros(len(raw_confidences))
        
        for i, conf in enumerate(raw_confidences):
            # Find bin and return its accuracy
            bin_idx = min(int(conf * self.num_bins), self.num_bins - 1)
            calibrated[i] = self.bin_accuracies[bin_idx]
        
        return calibrated

