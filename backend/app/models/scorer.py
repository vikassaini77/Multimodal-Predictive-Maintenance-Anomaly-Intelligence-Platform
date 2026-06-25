import torch
import torch.nn as nn
from typing import Optional, Tuple

class AnomalyScorerHead(nn.Module):
    """
    3-layer MLP that takes a fused embedding (or any feature vector)
    and maps it to a single logit representing anomaly risk.
    """
    def __init__(self, input_dim: int = 256, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, x: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input embeddings of shape (batch, input_dim)
            return_probs: If True, applies sigmoid to return probabilities.
                          If False, returns raw logits.
        """
        logits = self.mlp(x).squeeze(-1) # (batch,)
        
        if return_probs:
            return torch.sigmoid(logits)
        return logits

class MahalanobisDistance(nn.Module):
    """
    Computes Mahalanobis distance from a learned normal-class centroid.
    Used as an auxiliary anomaly signal, especially useful for OOD detection.
    """
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        # Register buffers so they are saved with state_dict but not trained via backprop
        self.register_buffer('centroid', torch.zeros(feature_dim))
        self.register_buffer('precision_matrix', torch.eye(feature_dim)) # Inverse covariance
        self.is_fitted = False
        
    def fit(self, normal_embeddings: torch.Tensor, epsilon: float = 1e-6):
        """
        Fits the centroid and precision matrix using a batch of normal-class embeddings.
        Args:
            normal_embeddings: Tensor of shape (N, feature_dim) where N > feature_dim
            epsilon: small constant for numerical stability during matrix inversion
        """
        if normal_embeddings.size(0) <= self.feature_dim:
            raise ValueError("Number of samples must be greater than feature_dim to estimate covariance.")
            
        self.centroid = normal_embeddings.mean(dim=0)
        
        # Mean center
        centered = normal_embeddings - self.centroid
        
        # Calculate covariance matrix
        cov = torch.matmul(centered.T, centered) / (normal_embeddings.size(0) - 1)
        
        # Add small ridge penalty for numerical stability
        cov = cov + torch.eye(self.feature_dim, device=cov.device) * epsilon
        
        # Precision matrix is inverse of covariance
        self.precision_matrix = torch.linalg.inv(cov)
        self.is_fitted = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates squared Mahalanobis distance for each embedding in x.
        Args:
            x: Input embeddings of shape (B, feature_dim)
        Returns:
            distances: Tensor of shape (B,)
        """
        if not self.is_fitted:
            raise RuntimeError("MahalanobisDistance must be fitted with normal samples before inference.")
            
        # (B, D)
        centered = x - self.centroid
        
        # Mahalanobis Distance: (x - \mu)^T \Sigma^{-1} (x - \mu)
        # We can compute this efficiently using batched operations
        # (B, 1, D) @ (D, D) @ (B, D, 1) -> (B, 1, 1) -> (B,)
        
        left_term = torch.matmul(centered, self.precision_matrix) # (B, D)
        # element-wise multiplication and sum across features is equivalent to dot product
        distances = (left_term * centered).sum(dim=-1) 
        
        return distances
