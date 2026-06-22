import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch_geometric.data import HeteroData
from typing import Dict

class GNNFaultTrainer:
    """
    Training loop for HeteroEquipmentGNN and FaultPredictionHead.
    Uses weighted BCE to handle heavily imbalanced fault signals.
    """
    def __init__(self, gnn_model: nn.Module, head_model: nn.Module, optimizer: Optimizer):
        self.gnn = gnn_model
        self.head = head_model
        self.optimizer = optimizer

    def train_step(self, data: HeteroData, target_node_type: str = "machine") -> Dict[str, float]:
        self.gnn.train()
        self.head.train()
        self.optimizer.zero_grad()

        # Forward pass
        node_embeddings = self.gnn(data.x_dict, data.edge_index_dict)
        
        # Predict faults for the target_node_type (e.g., 'machine')
        target_emb = node_embeddings[target_node_type]
        logits = self.head(target_emb).squeeze(-1)
        
        labels = data[target_node_type].y
        
        # Calculate pos_weight for imbalanced BCE
        num_pos = (labels == 1.0).sum()
        num_neg = (labels == 0.0).sum()
        
        if num_pos > 0:
            pos_weight = num_neg / num_pos
        else:
            pos_weight = torch.tensor(1.0, device=labels.device)
            
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        loss = criterion(logits, labels)
        loss.backward()
        self.optimizer.step()
        
        # Calculate metrics
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct = (preds == labels).sum().item()
        accuracy = correct / labels.size(0)
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "pos_weight": pos_weight.item()
        }

    def evaluate(self, data: HeteroData, target_node_type: str = "machine") -> Dict[str, float]:
        self.gnn.eval()
        self.head.eval()
        with torch.no_grad():
            node_embeddings = self.gnn(data.x_dict, data.edge_index_dict)
            target_emb = node_embeddings[target_node_type]
            logits = self.head(target_emb).squeeze(-1)
            
            labels = data[target_node_type].y
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct = (preds == labels).sum().item()
            accuracy = correct / labels.size(0)
            
        return {
            "accuracy": accuracy,
            "predictions": preds,
            "logits": logits
        }
