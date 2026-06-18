import torch
import torch.nn as nn

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
        # The CNN outputs 512 channels, so the transformer's d_model will be 512.
        self.d_model = 512
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        # x shape: (batch_size, in_channels, seq_len)
        features = self.cnn_extractor(x)
        
        # features shape: (batch_size, channels, new_seq_len)
        # Transformer expects (batch_size, seq_len, channels) with batch_first=True
        features = features.permute(0, 2, 1)
        
        # We need a simple positional encoding before transformer
        # Since seq_len might vary or be fixed, we can just use a learned parameter or let transformer handle without if not strict,
        # but the prompt specifically says "Positional encoding + multi-head attention"
        # Let's add a simple sinusoidal or learned PE. I'll add learned PE in __init__.
        # Wait, I can't easily add it to __init__ if I don't know new_seq_len, but usually we just add sinusoidal PE.
        # Let's write a quick PositionalEncoding module or just add it inline.

