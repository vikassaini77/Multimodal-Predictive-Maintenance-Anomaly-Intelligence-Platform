import torch
import pytest
import numpy as np
from backend.app.models.two_tower import VisualTower, VisualTowerClassifier
from backend.app.models.gradcam import VisualGradCAM

def test_gradcam_heatmap_generation():
    """
    Test that VisualGradCAM generates a heatmap of the correct shape and value range
    for a dummy image tensor.
    """
    # Create dummy VisualTower and Classifier
    visual_tower = VisualTower(embed_dim=128)
    model = VisualTowerClassifier(visual_tower)
    
    # In timm's efficientnet_b4, the last conv layer before pooling is conv_head
    target_layer = model.visual_tower.backbone.conv_head
    
    cam = VisualGradCAM(model, target_layer)
    
    # Dummy input image tensor (1 batch, 3 channels, 224 height, 224 width)
    input_tensor = torch.randn(1, 3, 224, 224)
    
    # Generate heatmap for target class 1 (anomaly)
    heatmap = cam.generate_heatmap(input_tensor, target_class=1)
    
    # Check shape
    assert heatmap.shape == (224, 224), f"Expected shape (224, 224), got {heatmap.shape}"
    
    # Check value range [0, 1]
    assert np.min(heatmap) >= 0.0, f"Min value {np.min(heatmap)} is < 0"
    assert np.max(heatmap) <= 1.0, f"Max value {np.max(heatmap)} is > 1"

def test_gradcam_overlay():
    """
    Test that the heatmap overlay function produces the correct output shape and type.
    """
    cam = VisualGradCAM(torch.nn.Identity(), torch.nn.Identity()) # Dummy model
    
    dummy_image = np.random.rand(224, 224, 3) # [0, 1] range
    dummy_heatmap = np.random.rand(224, 224)
    
    overlay = cam.overlay_heatmap(dummy_image, dummy_heatmap, alpha=0.5)
    
    assert overlay.shape == (224, 224, 3), f"Expected shape (224, 224, 3), got {overlay.shape}"
    assert overlay.dtype == np.uint8, f"Expected dtype uint8, got {overlay.dtype}"
