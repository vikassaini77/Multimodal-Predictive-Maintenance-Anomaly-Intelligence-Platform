import torch
import torch.nn as nn
import math

class SensorTower(nn.Module):
    """
    Sensor Tower for sequence data (e.g., CMAPSS).
    """
    def __init__(self, in_channels: int = 14):
        super().__init__()
        # 1. 1D CNN Feature Extractor
        # 3 conv blocks with BatchNorm, ReLU, MaxPool - outputs 512-dim
        self.cnn_extractor = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Block 2
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Block 3
            nn.Conv1d(in_channels=128, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # 2. Transformer Encoder
        # 4 heads, 2 layers on CNN output, with Positional Encoding
        self.d_model = 512
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        # x shape: (batch_size, in_channels, seq_len)
        features = self.cnn_extractor(x)
        
        # features shape: (batch_size, channels, new_seq_len)
        features = features.permute(0, 2, 1) # (batch_size, new_seq_len, channels)
        
        # Add simple positional encoding
        # Create a positional encoding tensor dynamically based on sequence length
        seq_len = features.size(1)
        pe = torch.zeros(seq_len, self.d_model, device=features.device)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add PE to features
        features = features + pe.unsqueeze(0)
        
        # Pass through transformer encoder
        features = self.transformer_encoder(features)
        
        return features
