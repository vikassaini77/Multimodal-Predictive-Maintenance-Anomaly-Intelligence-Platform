import torch
import torch.nn as nn
import torchvision.models as models
import timm

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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)
        
        # 3. Added non-linear projection head (SimCLR/CLIP style)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x: torch.Tensor, return_seq: bool = False) -> torch.Tensor:
        # x shape: (batch_size, input_dim, seq_len)
        x = self.conv(x)  # shape: (batch_size, 128, seq_len)
        
        # Transformer expects (batch, seq, feature) when batch_first=True
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        
        if return_seq:
            # We need to project the sequence to embed_dim
            # x is (B, seq_len, 128)
            # projection_head is linear, so we can apply it directly to the last dimension
            return self.projection_head(x)
            
        # Convert back to (batch, feature, seq) for pooling
        x = x.permute(0, 2, 1)
        
        x = self.pool(x).squeeze(-1)
        return self.projection_head(x)

class VisualTower(nn.Module):
    """EfficientNet-B4 for MVTec AD visual defect detection."""
    def __init__(self, embed_dim: int, freeze_backbone: bool = False):
        super().__init__()
        # Use a pre-trained EfficientNet-B4 via timm
        # global_pool='' ensures we get the unpooled 2D spatial feature map
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0, global_pool='')
        
        # 4. Freeze all but the last 2 blocks to prevent overfitting
        if freeze_backbone:
            # First freeze all parameters
            for param in self.backbone.parameters():
                param.requires_grad = False
                
            # Unfreeze the last 2 blocks
            for block in self.backbone.blocks[-2:]:
                for param in block.parameters():
                    param.requires_grad = True
                    
            # Unfreeze the final projection conv and batch norm
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True
                
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # 3. Added non-linear projection head for visual tower
        # EfficientNet-B4 outputs 1792 features
        self.projection_head = nn.Sequential(
            nn.Linear(1792, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x: torch.Tensor, return_seq: bool = False) -> torch.Tensor:
        # x shape: (batch_size, 3, H, W)
        x = self.backbone(x) # (B, 1792, H', W')
        
        if return_seq:
            B, C, H, W = x.shape
            x = x.view(B, C, H * W).transpose(1, 2) # (B, seq_len, 1792)
            return self.projection_head(x) # (B, seq_len, embed_dim)
            
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

class VisualTowerClassifier(nn.Module):
    """
    Wrapper around VisualTower to add a binary classification head.
    Used for fine-tuning the visual tower on MVTec AD with Focal Loss.
    """
    def __init__(self, visual_tower: VisualTower):
        super().__init__()
        self.visual_tower = visual_tower
        # The visual tower outputs embed_dim (default 256)
        embed_dim = visual_tower.projection_head[-1].out_features
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.visual_tower(x)
        # return logits
        return self.classifier(emb).squeeze(-1)

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss for contrastive learning.
    Pulls matched (positive) sensor and visual embeddings together, 
    while pushing all other (negative) pairs apart.
    """
    def __init__(self):
        super().__init__()
        # We don't initialize temperature here because TwoTowerAnomalyModel holds the learned temperature parameter.
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, sensor_emb: torch.Tensor, visual_emb: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sensor_emb: L2 normalized tensor (B, D)
            visual_emb: L2 normalized tensor (B, D)
            temperature: Scalar tensor
        """
        batch_size = sensor_emb.size(0)
        
        # Calculate similarity between all sensor and visual embeddings in the batch
        # similarity shape: (B, B)
        # element (i, j) is the cosine similarity between sensor_i and visual_j
        logits = torch.matmul(sensor_emb, visual_emb.T)
        
        # Scale by temperature
        logits = logits * temperature
        
        # The targets are the diagonal elements (i == j are positive pairs)
        targets = torch.arange(batch_size, device=sensor_emb.device)
        
        # Symmetric loss: we want to predict the correct visual for each sensor,
        # and the correct sensor for each visual.
        loss_s2v = self.criterion(logits, targets)
        loss_v2s = self.criterion(logits.T, targets)
        
        return (loss_s2v + loss_v2s) / 2.0
