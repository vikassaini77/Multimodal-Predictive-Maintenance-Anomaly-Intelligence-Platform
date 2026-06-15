import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class EquipmentTopologyGNN(nn.Module):
    """GraphSAGE network for modeling fault propagation across equipment nodes."""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        # Two-layer GraphSAGE to aggregate neighborhood fault signals
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features representing equipment states
            edge_index: Directed graph connectivity
        """
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x
