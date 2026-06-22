import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, HeteroConv
from typing import Dict, Tuple

class HeteroEquipmentGNN(nn.Module):
    """
    Heterogeneous GraphSAGE network for modeling fault propagation across equipment nodes.
    Applies dedicated SAGEConv layers for each edge type using mean aggregation.
    """
    def __init__(self, hidden_channels: int, out_channels: int, edge_types: list[Tuple[str, str, str]]):
        super().__init__()
        
        # Build layer 1 dictionary mapping each edge type to a SAGEConv layer
        conv1_dict = {
            edge_type: SAGEConv((-1, -1), hidden_channels, aggr='mean')
            for edge_type in edge_types
        }
        self.conv1 = HeteroConv(conv1_dict, aggr='sum')
        
        # Build layer 2 dictionary
        conv2_dict = {
            edge_type: SAGEConv((-1, -1), out_channels, aggr='mean')
            for edge_type in edge_types
        }
        self.conv2 = HeteroConv(conv2_dict, aggr='sum')
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_dict: Dictionary of node features per node type.
            edge_index_dict: Dictionary of edge indices per edge type.
        """
        out_dict1 = self.conv1(x_dict, edge_index_dict)
        for key in x_dict.keys():
            if key not in out_dict1:
                out_dict1[key] = x_dict[key]
                
        x_dict = {key: self.relu(x) for key, x in out_dict1.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        
        out_dict2 = self.conv2(x_dict, edge_index_dict)
        for key in x_dict.keys():
            if key not in out_dict2:
                out_dict2[key] = x_dict[key]
                
        return out_dict2

class FaultPredictionHead(nn.Module):
    """
    Binary classification head for node-level fault prediction.
    Outputs the probability of fault within the next N cycles.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes node embeddings and outputs raw logits.
        Use BCEWithLogitsLoss during training.
        """
        return self.fc(x)
