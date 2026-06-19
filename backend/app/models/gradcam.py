import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Optional

class VisualGradCAM:
    """
    GradCAM wrapper for the VisualTower to generate heatmaps for anomaly detection.
    Provides visual explainability by hooking into the last convolutional layer.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generates a GradCAM heatmap for a single image tensor.
        
        Args:
            input_tensor: Tensor of shape (1, 3, H, W)
            target_class: Class index to compute gradients for (e.g. 1 for anomaly)
                          If None, defaults to the predicted class (0 or 1).
                          
        Returns:
            np.ndarray: Heatmap of shape (H, W) normalized between 0 and 1.
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass (assumes model is VisualTowerClassifier or similar returning logits)
        logits = self.model(input_tensor)
        
        if target_class is None:
            # If binary (shape [] or [1]), check if logit > 0
            if logits.numel() == 1:
                target_class = int(torch.sigmoid(logits).item() > 0.5)
            else:
                target_class = logits.argmax(dim=1).item()
                
        # Backward pass
        if logits.numel() == 1:
            # Binary classification (single logit)
            # To get gradient for class 1, we just backward on the logit itself
            # If target_class is 0, we'd backward on -logit
            score = logits if target_class == 1 else -logits
        else:
            # Multi-class
            score = logits[0, target_class]
            
        score.backward()
        
        # Compute GradCAM
        # Global average pooling of the gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of the activations
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # ReLU to keep only positive influences
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.detach().cpu().numpy()
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        # Resize to input tensor size
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        return cam

    def overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Overlays a heatmap on an RGB image.
        
        Args:
            image: Original image as numpy array (H, W, 3) in [0, 255] or [0, 1]
            heatmap: GradCAM heatmap as numpy array (H, W) in [0, 1]
            alpha: Transparency of the heatmap
            
        Returns:
            np.ndarray: Overlayed image.
        """
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
            
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay
