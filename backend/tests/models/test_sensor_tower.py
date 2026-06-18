import torch
import pytest
from backend.app.models.sensor_tower import SensorTower

def test_sensor_tower_forward_shape():
    """Test the forward pass and output shape of SensorTower."""
    batch_size = 16
    in_channels = 14
    seq_len = 30
    
    model = SensorTower(in_channels=in_channels)
    x = torch.randn(batch_size, in_channels, seq_len)
    
    # Forward pass
    output = model(x)
    
    # Assert output shape is (batch, 256)
    assert output.shape == (batch_size, 256)

def test_sensor_tower_l2_norm():
    """Test that the output embeddings are L2 normalized."""
    batch_size = 4
    in_channels = 14
    seq_len = 30
    
    model = SensorTower(in_channels=in_channels)
    x = torch.randn(batch_size, in_channels, seq_len)
    
    output = model(x)
    
    # Check L2 norm along dimension 1 is 1.0
    norms = torch.norm(output, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_sensor_tower_grad_flow():
    """Test that gradients flow without NaNs."""
    batch_size = 4
    in_channels = 14
    seq_len = 30
    
    model = SensorTower(in_channels=in_channels)
    x = torch.randn(batch_size, in_channels, seq_len)
    
    # Forward pass
    output = model(x)
    
    # Dummy loss
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_grad = True
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            
    assert has_grad, "No gradients were computed"
