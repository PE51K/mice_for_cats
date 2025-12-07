"""
Raw Confidence baseline estimator.

Paper Section 4.4:
"Raw Confidence: Our first baseline is the raw confidence score from §2,
which can be used directly as a confidence estimate p."

"Recall that we defined this as ∏_{i∈S} p(w_i|w_{<i}), where S is the subset
of token indices that are relevant to the tool call."

Note: "calculating raw confidence does not require any learning, so neither
the training nor validation set is used."
"""


import numpy as np

from .base import BaseConfidenceEstimator


class RawConfidenceEstimator(BaseConfidenceEstimator):
    """
    Raw confidence baseline - uses token probabilities directly.

    This is the simplest baseline: just return the raw confidence
    computed from the product of token probabilities.

    Paper results show this is poorly calibrated (smECE 3-10x higher than others)
    but still useful as a baseline and as an input feature for other estimators.
    """

    @property
    def name(self) -> str:
        return "Raw Confidence"

    def fit(
        self, features: np.ndarray, labels: np.ndarray, raw_confidences: np.ndarray | None = None
    ) -> "RawConfidenceEstimator":
        """
        No-op fit - raw confidence doesn't require training.

        Paper: "calculating raw confidence does not require any learning"
        """
        return self

    def predict_proba(
        self, features: np.ndarray, raw_confidences: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Return raw confidences directly as predictions.

        Args:
            features: Unused for this estimator
            raw_confidences: Raw confidence scores to return

        Returns:
            Raw confidence scores unchanged
        """
        if raw_confidences is None:
            raise ValueError("RawConfidenceEstimator requires raw_confidences")

        return raw_confidences
