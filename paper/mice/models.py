"""MICE classifier models: Logistic Regression and Random Forest."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import List, Tuple, Optional
import warnings


class MICELogisticRegression:
    """MICE Logistic Regression classifier."""
    
    def __init__(self, l2_regularization: float = 2.0):
        """
        Initialize MICE Logistic Regression.
        
        Args:
            l2_regularization: L2 regularization strength (C = 1/regularization)
        """
        self.model = LogisticRegression(
            C=1.0 / l2_regularization,
            max_iter=1000,
            random_state=42,
        )
        self.l2_regularization = l2_regularization
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the logistic regression model.
        
        Args:
            X: Feature matrix [n_samples, n_features]
            y: Binary labels [n_samples] (1 = correct, 0 = incorrect)
        """
        self.model.fit(X, y)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict confidence probabilities.
        
        Args:
            X: Feature matrix [n_samples, n_features]
            
        Returns:
            Confidence probabilities [n_samples, 2] (second column is probability of correctness)
        """
        return self.model.predict_proba(X)
    
    def predict_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Predict confidence scores (probability of correctness).
        
        Args:
            X: Feature matrix [n_samples, n_features]
            
        Returns:
            Confidence scores [n_samples]
        """
        proba = self.predict_proba(X)
        return proba[:, 1]  # Probability of class 1 (correct)
    
    def get_feature_weights(self) -> np.ndarray:
        """Get feature weights/coefficients."""
        return self.model.coef_[0]


class MICERandomForest:
    """MICE Random Forest classifier."""
    
    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: int = 20,
        max_features: int = 10,
        random_state: int = 42,
    ):
        """
        Initialize MICE Random Forest.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            max_features: Maximum features to consider at each split
            random_state: Random seed
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the random forest model.
        
        Args:
            X: Feature matrix [n_samples, n_features]
            y: Binary labels [n_samples] (1 = correct, 0 = incorrect)
        """
        self.model.fit(X, y)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict confidence probabilities.
        
        Args:
            X: Feature matrix [n_samples, n_features]
            
        Returns:
            Confidence probabilities [n_samples, 2] (second column is probability of correctness)
        """
        return self.model.predict_proba(X)
    
    def predict_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Predict confidence scores (probability of correctness).
        
        Args:
            X: Feature matrix [n_samples, n_features]
            
        Returns:
            Confidence scores [n_samples]
        """
        proba = self.predict_proba(X)
        return proba[:, 1]  # Probability of class 1 (correct)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance (Gini coefficients)."""
        return self.model.feature_importances_


class HistogramRegressionEstimator:
    """Histogram Regression Estimator (HRE) baseline."""
    
    def __init__(self, n_bins: int = 25):
        """
        Initialize HRE.
        
        Args:
            n_bins: Number of bins for histogram
        """
        self.n_bins = n_bins
        self.bin_edges = None
        self.bin_accuracies = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train HRE by constructing histogram.
        
        Args:
            X: Raw confidence scores [n_samples, 1] or [n_samples]
            y: Binary labels [n_samples] (1 = correct, 0 = incorrect)
        """
        if X.ndim > 1:
            X = X[:, 0]  # Take first (and only) feature
        
        # Create bins: [0, 0.04), [0.04, 0.08), ..., [0.96, 1.0]
        self.bin_edges = np.linspace(0, 1.0, self.n_bins + 1)
        self.bin_accuracies = np.zeros(self.n_bins)
        
        # Compute accuracy for each bin
        for i in range(self.n_bins):
            mask = (X >= self.bin_edges[i]) & (X < self.bin_edges[i + 1])
            if i == self.n_bins - 1:  # Include upper bound for last bin
                mask = (X >= self.bin_edges[i]) & (X <= self.bin_edges[i + 1])
            
            if np.sum(mask) > 0:
                self.bin_accuracies[i] = np.mean(y[mask])
            else:
                # If no examples in bin, use nearest neighbor
                if i > 0:
                    self.bin_accuracies[i] = self.bin_accuracies[i - 1]
                else:
                    self.bin_accuracies[i] = np.mean(y)
    
    def predict_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Predict confidence by looking up bin accuracy.
        
        Args:
            X: Raw confidence scores [n_samples, 1] or [n_samples]
            
        Returns:
            Recalibrated confidence scores [n_samples]
        """
        if X.ndim > 1:
            X = X[:, 0]
        
        # Find bin for each score
        bin_indices = np.digitize(X, self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        return self.bin_accuracies[bin_indices]


class NadarayaWatsonKernelRegressor:
    """Nadaraya-Watson Kernel Regression (NWKR) baseline."""
    
    def __init__(self, bandwidth: Optional[float] = None):
        """
        Initialize NWKR.
        
        Args:
            bandwidth: Kernel bandwidth (if None, will be determined automatically)
        """
        self.bandwidth = bandwidth
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Store training data for kernel regression.
        
        Args:
            X: Raw confidence scores [n_samples, 1] or [n_samples]
            y: Binary labels [n_samples] (1 = correct, 0 = incorrect)
        """
        if X.ndim > 1:
            X = X[:, 0]
        
        self.X_train = X
        self.y_train = y
        
        # Determine bandwidth if not provided
        if self.bandwidth is None:
            # Use Silverman's rule of thumb
            std = np.std(X)
            n = len(X)
            self.bandwidth = 1.06 * std * (n ** (-1.0 / 5.0))
    
    def predict_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Predict confidence using Nadaraya-Watson kernel regression.
        
        Uses a reflected Gaussian kernel as in the paper.
        
        Args:
            X: Raw confidence scores [n_samples, 1] or [n_samples]
            
        Returns:
            Recalibrated confidence scores [n_samples]
        """
        if X.ndim > 1:
            X = X[:, 0]
        
        predictions = np.zeros(len(X))
        
        for i, x in enumerate(X):
            # Compute kernel weights using reflected Gaussian kernel
            # Reflect at boundaries 0 and 1
            distances = np.abs(self.X_train - x)
            # Also consider reflections
            distances_reflected_0 = np.abs(self.X_train + x)  # Reflection at 0
            distances_reflected_1 = np.abs(2 - self.X_train - x)  # Reflection at 1
            
            # Take minimum distance (closest point or reflection)
            min_distances = np.minimum(distances, np.minimum(distances_reflected_0, distances_reflected_1))
            
            # Gaussian kernel
            weights = np.exp(-0.5 * (min_distances / self.bandwidth) ** 2)
            
            # Normalize weights
            weights_sum = np.sum(weights)
            if weights_sum > 0:
                weights = weights / weights_sum
            else:
                weights = np.ones_like(weights) / len(weights)
            
            # Weighted average
            predictions[i] = np.sum(weights * self.y_train)
        
        return predictions

