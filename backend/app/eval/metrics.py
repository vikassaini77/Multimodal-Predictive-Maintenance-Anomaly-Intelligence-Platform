import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Dict

def calculate_node_metrics(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Calculates precision, recall, and F1 score for binary node predictions.
    """
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    if len(labels_np) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    precision = precision_score(labels_np, preds_np, zero_division=0)
    recall = recall_score(labels_np, preds_np, zero_division=0)
    f1 = f1_score(labels_np, preds_np, zero_division=0)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }

def evaluate_hetero_metrics(preds_dict: Dict[str, torch.Tensor], labels_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    """
    Evaluates metrics per node type.
    """
    results = {}
    for node_type, preds in preds_dict.items():
        if node_type in labels_dict:
            results[node_type] = calculate_node_metrics(preds, labels_dict[node_type])
    return results

import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Callable, Tuple

def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable,
    n_bootstraps: int = 1000,
    alpha: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Computes the confidence interval for a given metric using bootstrapping.
    Args:
        y_true: True labels
        y_pred: Predictions (probabilities or binary, depending on metric_func)
        metric_func: Function to compute the metric (e.g., roc_auc_score or f1_score)
        n_bootstraps: Number of bootstrap samples
        alpha: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed for reproducibility
    Returns:
        lower_bound, upper_bound
    """
    rng = np.random.RandomState(random_state)
    bootstrapped_scores = []
    
    indices = np.arange(len(y_true))
    
    for _ in range(n_bootstraps):
        # Sample with replacement
        sample_idx = rng.choice(indices, size=len(indices), replace=True)
        
        # Ensure there's at least one positive and one negative sample in the bootstrap
        # (especially important for AUROC)
        if len(np.unique(y_true[sample_idx])) < 2:
            continue
            
        score = metric_func(y_true[sample_idx], y_pred[sample_idx])
        bootstrapped_scores.append(score)
        
    if not bootstrapped_scores:
        return 0.0, 0.0
        
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    # Calculate percentiles
    lower_percentile = (1.0 - alpha) / 2.0 * 100
    upper_percentile = (alpha + (1.0 - alpha) / 2.0) * 100
    
    lower_bound = np.percentile(sorted_scores, lower_percentile)
    upper_bound = np.percentile(sorted_scores, upper_percentile)
    
    return float(lower_bound), float(upper_bound)
