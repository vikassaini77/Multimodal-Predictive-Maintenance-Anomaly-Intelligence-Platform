import torch
import pytest
from backend.app.models.two_tower import TwoTowerAnomalyModel

def test_two_tower_forward_pass():
    """
    Test the forward pass of the Two-Tower multimodal model with dummy data
    to ensure tensor shapes and normalizations are correct.
    """
    batch_size = 2
    sensor_input_dim = 14
    seq_len = 50
    embed_dim = 128
    
    # Create dummy tensors
    # Sensor: (batch_size, input_dim, seq_len)
    dummy_sensor = torch.randn(batch_size, sensor_input_dim, seq_len)
    # Visual: (batch_size, channels, height, width) for EfficientNet
    dummy_visual = torch.randn(batch_size, 3, 224, 224)
    
    model = TwoTowerAnomalyModel(sensor_input_dim=sensor_input_dim, embed_dim=embed_dim)
    
    # We use torch.no_grad() as we are just testing the forward pass
    with torch.no_grad():
        output = model(dummy_sensor, dummy_visual)
        
    assert "sensor_embeddings" in output
    assert "visual_embeddings" in output
    
    sensor_emb = output["sensor_embeddings"]
    visual_emb = output["visual_embeddings"]
    
    # Check output dimensions
    assert sensor_emb.shape == (batch_size, embed_dim)
    assert visual_emb.shape == (batch_size, embed_dim)
    
    # Check L2 normalization (norm should be ~1.0)
    assert torch.allclose(torch.norm(sensor_emb, p=2, dim=1), torch.ones(batch_size))
    assert torch.allclose(torch.norm(visual_emb, p=2, dim=1), torch.ones(batch_size))
