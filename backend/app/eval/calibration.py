import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, f1_score
from typing import Tuple, Dict

class PlattScaler:
    """
    Fits a logistic regression model on raw model scores vs. true labels
    to produce well-calibrated probabilities.
    """
    def __init__(self):
        self.scaler = LogisticRegression(solver='lbfgs', C=1.0) # Adding L2 regularization for stability
        self.is_fitted = False
        
    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray):
        """
        Fits the platt scaler.
        Args:
            raw_scores: (N,) array of raw model outputs (e.g., logits)
            y_true: (N,) array of binary labels {0, 1}
        """
        X = raw_scores.reshape(-1, 1)
        self.scaler.fit(X, y_true)
        self.is_fitted = True
        
    def predict_proba(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Returns calibrated probabilities.
        """
        if not self.is_fitted:
            raise RuntimeError("PlattScaler must be fitted before predict_proba.")
            
        X = raw_scores.reshape(-1, 1)
        # scaler.predict_proba returns (N, 2), we want the probability of class 1
        return self.scaler.predict_proba(X)[:, 1]

class ThresholdTuner:
    """
    Sweeps thresholds on calibrated probabilities to find the optimal decision boundary
    that maximizes the F1 score on the validation set.
    """
    def __init__(self):
        self.optimal_threshold = 0.5
        self.is_fitted = False
        
    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Finds the optimal threshold.
        Args:
            y_prob: (N,) array of calibrated probabilities in [0, 1]
            y_true: (N,) array of binary labels {0, 1}
        Returns:
            Dict containing best_threshold and best_f1
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        
        # Calculate F1 score for each threshold
        # f1 = 2 * (precision * recall) / (precision + recall)
        # Avoid division by zero
        f1_scores = np.divide(
            2 * (precisions * recalls),
            (precisions + recalls),
            out=np.zeros_like(precisions),
            where=(precisions + recalls) != 0
        )
        
        # PR curve returns len(thresholds)+1 precisions/recalls, so we drop the last element
        f1_scores = f1_scores[:-1]
        
        best_idx = np.argmax(f1_scores)
        self.optimal_threshold = float(thresholds[best_idx])
        self.is_fitted = True
        
        return {
            "best_threshold": self.optimal_threshold,
            "best_f1": float(f1_scores[best_idx])
        }
        
    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Applies the optimal threshold to probabilities.
        """
        if not self.is_fitted:
            raise RuntimeError("ThresholdTuner must be fitted before predict.")
            
        return (y_prob >= self.optimal_threshold).astype(int)
