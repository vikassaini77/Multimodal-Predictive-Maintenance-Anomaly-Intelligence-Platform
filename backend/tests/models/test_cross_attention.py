import torch
import pytest
from backend.app.models.cross_attention import MultimodalFusionModel, CrossModalAttentionBlock, SymmetricFusion

@pytest.fixture
def input_dims():
    return {"sensor_dim": 14, "embed_dim": 256}

@pytest.fixture
def fusion_model(input_dims):
    return MultimodalFusionModel(
        sensor_input_dim=input_dims["sensor_dim"],
        embed_dim=input_dims["embed_dim"]
    )

def test_cross_attention_block(input_dims):
    block = CrossModalAttentionBlock(embed_dim=input_dims["embed_dim"], num_heads=4)
    # Mock sequence: batch=2, seq_len=10, dim=256
    query = torch.randn(2, 10, 256)
    # Mock sequence: batch=2, seq_len=15, dim=256
    key_value = torch.randn(2, 15, 256)
    
    out, weights = block(query, key_value)
    
    # Output should match query shape
    assert out.shape == (2, 10, 256)
    # Weights should be (batch*num_heads, query_seq_len, kv_seq_len) depending on batch_first,
    # actually PyTorch MultiheadAttention with batch_first=True returns average over heads unless need_weights=True 
    # and average_attn_weights=False. Default is averaged over heads -> (batch, query_seq_len, kv_seq_len)
    assert weights.shape == (2, 10, 15)

def test_symmetric_fusion(input_dims):
    fusion = SymmetricFusion(embed_dim=input_dims["embed_dim"])
    sensor_seq = torch.randn(2, 5, 256)
    visual_seq = torch.randn(2, 16, 256)
    
    out, weights = fusion(sensor_seq, visual_seq)
    
    # Should project back to embed_dim after pooling and concatenating
    assert out.shape == (2, 256)
    assert 's2v' in weights
    assert 'v2s' in weights

def test_multimodal_fusion_forward_both(fusion_model, input_dims):
    sensor_data = torch.randn(2, input_dims["sensor_dim"], 50) # (B, dim, seq_len)
    visual_data = torch.randn(2, 3, 224, 224) # (B, C, H, W)
    
    out, aux = fusion_model(sensor_data, visual_data)
    
    assert out.shape == (2, 256)
    
    gate_weights = aux['gate_weights']
    assert gate_weights.shape == (2, 2)
    # Softmax sums to 1
    assert torch.allclose(gate_weights.sum(dim=1), torch.ones(2))

def test_multimodal_fusion_missing_visual(fusion_model, input_dims):
    sensor_data = torch.randn(2, input_dims["sensor_dim"], 50)
    
    out, aux = fusion_model(sensor_data=sensor_data, visual_data=None)
    
    assert out.shape == (2, 256)
    gate_weights = aux['gate_weights']
    assert gate_weights.shape == (2, 2)
    # Sensor weight should be 1.0, visual 0.0
    assert torch.allclose(gate_weights, torch.tensor([[1.0, 0.0], [1.0, 0.0]]))

def test_multimodal_fusion_missing_sensor(fusion_model, input_dims):
    visual_data = torch.randn(2, 3, 224, 224)
    
    out, aux = fusion_model(sensor_data=None, visual_data=visual_data)
    
    assert out.shape == (2, 256)
    gate_weights = aux['gate_weights']
    assert gate_weights.shape == (2, 2)
    # Sensor weight should be 0.0, visual 1.0
    assert torch.allclose(gate_weights, torch.tensor([[0.0, 1.0], [0.0, 1.0]]))

def test_multimodal_fusion_both_none(fusion_model):
    with pytest.raises(ValueError):
        fusion_model(sensor_data=None, visual_data=None)
