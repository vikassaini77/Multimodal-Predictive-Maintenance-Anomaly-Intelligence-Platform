import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class EquipmentTopologyGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
