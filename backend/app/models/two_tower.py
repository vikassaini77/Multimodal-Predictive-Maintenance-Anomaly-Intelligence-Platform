import torch
import torch.nn as nn
import torchvision.models as models

class SensorTower(nn.Module):
    """1D-CNN + Transformer for NASA CMAPSS and MIMII time-series data."""
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        # 1. Added BatchNorm to stabilize training
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # 2. Added Transformer layer for long-range temporal dependencies
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. Added non-linear projection head (SimCLR/CLIP style)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, input_dim, seq_len)
        x = self.conv(x)  # shape: (batch_size, 128, seq_len)
        
        # Transformer expects (batch, seq, feature) when batch_first=True
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        # Convert back to (batch, feature, seq) for pooling
        x = x.permute(0, 2, 1)
        
        x = self.pool(x).squeeze(-1)
        return self.projection_head(x)

class VisualTower(nn.Module):
    """EfficientNet-B4 for MVTec AD visual defect detection."""
    def __init__(self, embed_dim: int, freeze_backbone: bool = False):
        super().__init__()
        # Use a pre-trained EfficientNet-B4
        effnet = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        # Remove the classification head
        self.features = effnet.features
        
        # 4. Optional freezing of the massive backbone to prevent overfitting early on
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
                
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # 3. Added non-linear projection head for visual tower
        # EfficientNet-B4 outputs 1792 features
        self.projection_head = nn.Sequential(
            nn.Linear(1792, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, 3, H, W)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.projection_head(x)

class TwoTowerAnomalyModel(nn.Module):
    """
    Multimodal fusion of sensor and visual towers via a joint embedding space.
    Used for contrastive loss and anomaly scoring.
    """
    def __init__(self, sensor_input_dim: int, embed_dim: int = 256, freeze_visual: bool = False):
        super().__init__()
        self.sensor_tower = SensorTower(input_dim=sensor_input_dim, embed_dim=embed_dim)
        self.visual_tower = VisualTower(embed_dim=embed_dim, freeze_backbone=freeze_visual)
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
        
    def predict_anomaly(self, sensor_data: torch.Tensor, visual_data: torch.Tensor) -> torch.Tensor:
        """
        5. Fuses modalities at inference time to predict an anomaly score.
        Calculates the cosine distance. A high distance indicates the sensor
        and visual data disagree, implying a potential anomaly.
        """
        with torch.no_grad():
            sensor_emb = self.sensor_tower(sensor_data)
            visual_emb = self.visual_tower(visual_data)
            
            sensor_emb = nn.functional.normalize(sensor_emb, p=2, dim=1)
            visual_emb = nn.functional.normalize(visual_emb, p=2, dim=1)
            
            # Cosine similarity is dot product of L2 normalized vectors
            # Similarity is 1 if identical, -1 if opposite
            similarity = (sensor_emb * visual_emb).sum(dim=1)
            
            # Convert similarity to an anomaly score (0 to 1)
            # High similarity = low anomaly (0)
            # Low similarity = high anomaly (1)
            anomaly_score = 1.0 - ((similarity + 1.0) / 2.0)
            
            return anomaly_score
