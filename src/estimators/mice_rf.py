"""
MICE Random Forest.

Paper Section 4.4:
"MICE Random Forest (MICE RF): We train a random forest classifier using
1000 trees each with a maximum depth of 20 and a maximum of 10 features
to use at each split, using the Scikit-Learn package (Pedregosa et al., 2011).
Other hyperparameters are set to defaults. This model is also trained to
predict whether the tool call is correct."

Paper Results (Section 5):
"Across all three LLMs, MICE RF performs best at nearly every risk level."
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import BaseConfidenceEstimator


class MICERandomForest(BaseConfidenceEstimator):
    """
    MICE with Random Forest classifier.

    Paper hyperparameters:
    - n_estimators: 1000 trees
    - max_depth: 20
    - max_features: 10 features per split
    - Other parameters: sklearn defaults

    Paper results show MICE RF:
    - Performs best at nearly every risk level
    - Significantly outperforms baselines at medium/high risk
    - Is sample efficient (beats NWKR with only 300 examples)
    """

    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: int = 20,
        max_features: int = 10,
        random_state: int = 42,
        zero_shot: bool = False,
    ):
        """
        Initialize MICE RF with paper hyperparameters.

        Args:
            n_estimators: Number of trees (paper: 1000)
            max_depth: Maximum tree depth (paper: 20)
            max_features: Features per split (paper: 10)
            random_state: Random seed for reproducibility
            zero_shot: If True, indicates zero-shot evaluation on new APIs
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.zero_shot = zero_shot

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores
        )

    @property
    def name(self) -> str:
        """Estimator name."""
        if self.zero_shot:
            return "MICE RF (zero-shot)"
        return "MICE RF"

    def fit(
        self, features: np.ndarray, labels: np.ndarray, raw_confidences: np.ndarray | None = None
    ) -> "MICERandomForest":
        """
        Fit random forest on MICE features.

        Paper: "trained to predict whether the tool call is correct"

        Args:
            features: BERTScore features [n_samples, n_layers-1]
            labels: Binary correctness labels [n_samples]
            raw_confidences: Raw confidence scores [n_samples]
        """
        if raw_confidences is None:
            raise ValueError("MICE RF requires raw_confidences as a feature")

        # Combine features: [BERTScores, raw_confidence]
        X = np.column_stack([features, raw_confidences])  # noqa: N806

        # Check if we have both classes
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            # Fallback: predict constant probability equal to label mean
            self._fallback_prob = float(labels.mean())
            self._use_fallback = True
            print(
                "  Warning: Only one class in training data, "
                f"using fallback (p={self._fallback_prob})"
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

        Args:
            features: BERTScore features [n_samples, n_layers-1]
            raw_confidences: Raw confidence scores [n_samples]

        Returns:
            Probability of correct tool call [n_samples]
        """
        if raw_confidences is None:
            raise ValueError("MICE RF requires raw_confidences as a feature")

        # Handle fallback case (single class in training)
        if getattr(self, "_use_fallback", False):
            return np.full(len(raw_confidences), self._fallback_prob)

        X = np.column_stack([features, raw_confidences])  # noqa: N806

        # Return probability of class 1 (correct)
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> np.ndarray:
        """
        Get Gini feature importance scores.

        Paper Figure 5: "Feature importance for BERTScore features
        and confidence on the trained MICE RF model"

        Shows confidence is ~3x as important as other features.

        Returns:
            Gini importance for each feature (BERTScores + confidence)
        """
        return self.model.feature_importances_
