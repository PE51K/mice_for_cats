"""Base class for confidence estimators."""

from abc import ABC, abstractmethod

import numpy as np


class BaseConfidenceEstimator(ABC):
    """
    Base class for all confidence estimators.

    All estimators must implement:
    - fit(): Train on data (may be no-op for some estimators)
    - predict_proba(): Return calibrated confidence scores
    - name: Property returning estimator name
    """

    @abstractmethod
    def fit(
        self, features: np.ndarray, labels: np.ndarray, raw_confidences: np.ndarray | None = None
    ) -> "BaseConfidenceEstimator":
        """
        Fit the estimator on training data.

        Args:
            features: MICE features (BERTScores) [n_samples, n_features]
            labels: Binary correctness labels [n_samples]
            raw_confidences: Raw confidence scores [n_samples]

        Returns:
            self: Fitted estimator
        """
        pass

    @abstractmethod
    def predict_proba(
        self, features: np.ndarray, raw_confidences: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Predict confidence probabilities.

        Args:
            features: MICE features [n_samples, n_features]
            raw_confidences: Raw confidence scores [n_samples]

        Returns:
            confidences: Calibrated confidence scores [n_samples]
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return estimator name for logging and results."""
        pass
