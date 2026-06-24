import torch
import torch.nn as nn
from typing import Optional, Tuple
from backend.app.models.two_tower import TwoTowerAnomalyModel

class CrossModalAttentionBlock(nn.Module):
    """
    Standard multi-head attention block for cross-modal fusion.
    Query comes from one modality, Key and Value come from the other.
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # batch_first=True expects (batch, seq, feature)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Tensor of shape (B, L_q, D)
            key_value: Tensor of shape (B, L_k, D)
            key_padding_mask: Optional mask for the keys
        """
        # Multi-head attention
        attn_out, attn_weights = self.mha(
            query, key_value, key_value, key_padding_mask=key_padding_mask
        )
        
        # Residual + Norm
        x = self.norm1(query + attn_out)
        
        # Feed Forward
        ffn_out = self.ffn(x)
        
        # Residual + Norm
        x = self.norm2(x + ffn_out)
        
        return x, attn_weights

class SymmetricFusion(nn.Module):
    """
    Bidirectional cross-attention that captures complementary signals from both modalities.
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # Sensor queries Visual
        self.s2v_attn = CrossModalAttentionBlock(embed_dim, num_heads, dropout)
        # Visual queries Sensor
        self.v2s_attn = CrossModalAttentionBlock(embed_dim, num_heads, dropout)
        
        # Projection after concatenation
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, sensor_seq: torch.Tensor, visual_seq: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            sensor_seq: (B, L_s, D)
            visual_seq: (B, L_v, D)
        """
        # Sensor querying Visual
        s2v_out, s2v_weights = self.s2v_attn(query=sensor_seq, key_value=visual_seq)
        
        # Visual querying Sensor
        v2s_out, v2s_weights = self.v2s_attn(query=visual_seq, key_value=sensor_seq)
        
        # Pool sequences into single representations per modality
        s2v_pooled = s2v_out.mean(dim=1)  # (B, D)
        v2s_pooled = v2s_out.mean(dim=1)  # (B, D)
        
        # Concatenate and project back to 256-dim
        concat_feats = torch.cat([s2v_pooled, v2s_pooled], dim=1) # (B, 2*D)
        fused_feats = self.proj(concat_feats) # (B, D)
        
        attn_weights = {
            's2v': s2v_weights,
            'v2s': v2s_weights
        }
        
        return fused_feats, attn_weights

class MultimodalFusionModel(nn.Module):
    """
    Wraps the TwoTower backbone with Symmetric Fusion and Learnable Modality Gating.
    Handles graceful degradation if a modality is missing.
    """
    def __init__(self, sensor_input_dim: int, embed_dim: int = 256):
        super().__init__()
        self.two_tower = TwoTowerAnomalyModel(sensor_input_dim=sensor_input_dim, embed_dim=embed_dim, freeze_visual=False)
        self.fusion = SymmetricFusion(embed_dim=embed_dim)
        
        # Gating network: learns to weight modalities based on confidence
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)  # Weights sum to 1
        )
        
        self.fallback_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, sensor_data: Optional[torch.Tensor], visual_data: Optional[torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass allowing missing modalities (None).
        """
        # If both are present, use full fusion
        if sensor_data is not None and visual_data is not None:
            # Extract sequence features (return_seq=True)
            sensor_seq = self.two_tower.sensor_tower(sensor_data, return_seq=True) # (B, L_s, D)
            visual_seq = self.two_tower.visual_tower(visual_data, return_seq=True) # (B, L_v, D)
            
            # Cross-attention fusion
            fused_feats, attn_weights = self.fusion(sensor_seq, visual_seq)
            
            # Get pooled embeddings for gating (we can pool the sequences)
            sensor_emb = sensor_seq.mean(dim=1)
            visual_emb = visual_seq.mean(dim=1)
            
            # Calculate gating weights
            concat_unfused = torch.cat([sensor_emb, visual_emb], dim=1) # (B, 2*D)
            gate_weights = self.gate(concat_unfused) # (B, 2)
            
            sensor_weight = gate_weights[:, 0:1]
            visual_weight = gate_weights[:, 1:2]
            
            # Gated residual connection
            gated_unfused = sensor_emb * sensor_weight + visual_emb * visual_weight
            final_output = fused_feats + gated_unfused
            
            return final_output, {"attn_weights": attn_weights, "gate_weights": gate_weights}
            
        elif sensor_data is not None:
            # Missing visual
            sensor_emb = self.two_tower.sensor_tower(sensor_data) # Pooled
            device = sensor_data.device
            # Default to full weight for sensor
            gate_weights = torch.tensor([[1.0, 0.0]], device=device).expand(sensor_emb.size(0), -1)
            return self.fallback_proj(sensor_emb), {"gate_weights": gate_weights}
            
        elif visual_data is not None:
            # Missing sensor
            visual_emb = self.two_tower.visual_tower(visual_data) # Pooled
            device = visual_data.device
            # Default to full weight for visual
            gate_weights = torch.tensor([[0.0, 1.0]], device=device).expand(visual_emb.size(0), -1)
            return self.fallback_proj(visual_emb), {"gate_weights": gate_weights}
            
        else:
            raise ValueError("Both sensor_data and visual_data cannot be None")
