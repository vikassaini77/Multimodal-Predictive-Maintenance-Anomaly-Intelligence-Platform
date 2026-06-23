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
