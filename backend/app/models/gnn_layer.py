import torch
import torch.nn as nn

class EquipmentTopologyGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
