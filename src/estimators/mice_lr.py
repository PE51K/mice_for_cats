"""
MICE Logistic Regression.

Paper Section 4.4:
"MICE Logistic Regressor (MICE LR): We train a logistic regression model
with an L2 regularization strength of 2 to predict whether the tool call
is correct or not."

Paper Footnote 11:
"For LR, this is exactly Platt scaling with L2 regularization."

Paper Section 7:
"MICE LR could be viewed as an extension to Platt scaling because MICE
conditions on model internals in addition to the original confidence."
"""


import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import BaseConfidenceEstimator


class MICELogisticRegression(BaseConfidenceEstimator):
    """
    MICE with Logistic Regression classifier.

    Paper hyperparameters:
    - L2 regularization strength: 2.0
    - sklearn C parameter: 0.5 (C = 1/lambda)

    Features:
    - BERTScore from each layer (ℓ-1 features)
    - Raw confidence (1 feature)
    """

    def __init__(
        self,
        C: float = 0.5,  # 1 / L2_strength = 1/2 = 0.5
        zero_shot: bool = False,
        random_state: int = 42,
    ):
        """
        Initialize MICE LR.

        Args:
            C: Inverse regularization strength (paper: 0.5 for L2=2)
            zero_shot: If True, indicates zero-shot evaluation on new APIs
            random_state: Random seed for reproducibility
        """
        self.C = C
        self.zero_shot = zero_shot
        self.random_state = random_state

        self.model = LogisticRegression(
            C=C, solver="lbfgs", max_iter=1000, random_state=random_state
        )

    @property
    def name(self) -> str:
        if self.zero_shot:
            return "MICE LR (zero-shot)"
        return "MICE LR"

    def fit(
        self, features: np.ndarray, labels: np.ndarray, raw_confidences: np.ndarray | None = None
    ) -> "MICELogisticRegression":
        """
        Fit logistic regression on MICE features.

        Paper: "Features used by MICE regressors were described in §2"
        - BERTScore features from each layer
        - Raw confidence as additional feature

        Args:
            features: BERTScore features [n_samples, n_layers-1]
            labels: Binary correctness labels [n_samples]
            raw_confidences: Raw confidence scores [n_samples]
        """
        if raw_confidences is None:
            raise ValueError("MICE LR requires raw_confidences as a feature")

        # Combine features: [BERTScores, raw_confidence]
        X = np.column_stack([features, raw_confidences])

        # Check if we have both classes (required for LogisticRegression)
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            # Fallback: predict constant probability equal to label mean
            self._fallback_prob = float(labels.mean())
            self._use_fallback = True
            print(
                f"  Warning: Only one class in training data, using fallback (p={self._fallback_prob})"
            )
            return self

        self._use_fallback = False
        self.model.fit(X, labels)

        return self

    def predict_proba(
        self, features: np.ndarray, raw_confidences: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Predict probability of correctness.

        Returns calibrated confidence scores.

        Args:
            features: BERTScore features [n_samples, n_layers-1]
            raw_confidences: Raw confidence scores [n_samples]

        Returns:
            Probability of correct tool call [n_samples]
        """
        if raw_confidences is None:
            raise ValueError("MICE LR requires raw_confidences as a feature")

        # Handle fallback case (single class in training)
        if getattr(self, "_use_fallback", False):
            return np.full(len(raw_confidences), self._fallback_prob)

        X = np.column_stack([features, raw_confidences])

        # Return probability of class 1 (correct)
        return self.model.predict_proba(X)[:, 1]

    def get_coefficients(self) -> np.ndarray:
        """
        Get learned feature coefficients.

        Paper Figure 6: "Coefficients for the trained MICE LR model"
        Shows confidence is ~2x as important as other features.

        Returns:
            Coefficients for each feature (BERTScores + confidence)
        """
        return self.model.coef_[0]
