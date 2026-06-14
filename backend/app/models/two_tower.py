import torch
import torch.nn as nn
import torchvision.models as models

class SensorTower(nn.Module):
    """1D-CNN + Transformer for NASA CMAPSS and MIMII time-series data."""
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, input_dim, seq_len)
        x = self.conv(x).squeeze(-1)
        return self.fc(x)

class VisualTower(nn.Module):
    """EfficientNet-B4 for MVTec AD visual defect detection."""
    def __init__(self, embed_dim: int):
        super().__init__()
        # Use a pre-trained EfficientNet-B4
        effnet = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        # Remove the classification head
        self.features = effnet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        # EfficientNet-B4 outputs 1792 features
        self.fc = nn.Linear(1792, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, 3, H, W)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

class TwoTowerAnomalyModel(nn.Module):
    """
    Multimodal fusion of sensor and visual towers via a joint embedding space.
    Used for contrastive loss and anomaly scoring.
    """
    def __init__(self, sensor_input_dim: int, embed_dim: int = 256):
        super().__init__()
        self.sensor_tower = SensorTower(input_dim=sensor_input_dim, embed_dim=embed_dim)
        self.visual_tower = VisualTower(embed_dim=embed_dim)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, sensor_data: torch.Tensor, visual_data: torch.Tensor) -> dict:
        """
        Forward pass for both modalities.
        
        Args:
            sensor_data: Tensor of shape (B, sensor_input_dim, seq_len)
            visual_data: Tensor of shape (B, 3, H, W)
            
        Returns:
            dict with normalized embeddings.
        """
        sensor_emb = self.sensor_tower(sensor_data)
        visual_emb = self.visual_tower(visual_data)
        
        # L2 Normalize embeddings for contrastive learning (CLIP-style)
        sensor_emb = nn.functional.normalize(sensor_emb, p=2, dim=1)
        visual_emb = nn.functional.normalize(visual_emb, p=2, dim=1)
        
        return {
            "sensor_embeddings": sensor_emb,
            "visual_embeddings": visual_emb,
            "temperature": torch.clamp(self.temperature.exp(), max=100.0)
        }
