import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any

class FusionTrainer:
    """
    Fine-tunes the MultimodalFusionModel with frozen tower backbones.
    Only the cross-attention, gating network, and final projection layers are trainable.
    """
    def __init__(self, model: nn.Module, device: torch.device, learning_rate: float = 1e-4):
        self.model = model.to(device)
        self.device = device
        
        # Freeze backbones to prevent catastrophic forgetting
        for param in self.model.two_tower.sensor_tower.parameters():
            param.requires_grad = False
        for param in self.model.two_tower.visual_tower.parameters():
            param.requires_grad = False
            
        # Ensure fusion and gating layers are trainable
        for param in self.model.fusion.parameters():
            param.requires_grad = True
        for param in self.model.gate.parameters():
            param.requires_grad = True
        for param in self.model.fallback_proj.parameters():
            param.requires_grad = True
            
        # We also need a dummy loss for fine-tuning if we don't have downstream labels yet.
        # Alternatively, if this is fine-tuned on contrastive loss, we could use NTXentLoss.
        # But fusion combines them into a single representation. Typically, fusion is fine-tuned
        # on the downstream task (like anomaly classification).
        # We will assume a binary classification task for the fusion fine-tuning.
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)
        
        # Optimizer includes fusion layers and classifier
        self.optimizer = optim.Adam([
            {'params': self.model.fusion.parameters()},
            {'params': self.model.gate.parameters()},
            {'params': self.model.fallback_proj.parameters()},
            {'params': self.classifier.parameters()}
        ], lr=learning_rate)
        
        # Binary Cross Entropy with Logits
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        self.classifier.train()
        total_loss = 0.0
        
        for batch in dataloader:
            # Assume batch is a dict with 'sensor_data', 'visual_data', 'label'
            sensor_data = batch.get('sensor_data')
            visual_data = batch.get('visual_data')
            labels = batch['label'].to(self.device).float()
            
            if sensor_data is not None:
                sensor_data = sensor_data.to(self.device)
            if visual_data is not None:
                visual_data = visual_data.to(self.device)
                
            self.optimizer.zero_grad()
            
            # Forward pass
            # Returns fused representation and dict of attention/gate weights
            fused_repr, aux_outputs = self.model(sensor_data, visual_data)
            
            # Predict anomaly
            logits = self.classifier(fused_repr).squeeze(-1)
            
            # Calculate loss
            loss = self.criterion(logits, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            
        return total_loss / len(dataloader.dataset)
        
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        self.classifier.eval()
        total_loss = 0.0
        correct = 0
        
        with torch.no_grad():
            for batch in dataloader:
                sensor_data = batch.get('sensor_data')
                visual_data = batch.get('visual_data')
                labels = batch['label'].to(self.device).float()
                
                if sensor_data is not None:
                    sensor_data = sensor_data.to(self.device)
                if visual_data is not None:
                    visual_data = visual_data.to(self.device)
                    
                fused_repr, _ = self.model(sensor_data, visual_data)
                logits = self.classifier(fused_repr).squeeze(-1)
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                
        avg_loss = total_loss / len(dataloader.dataset)
        accuracy = correct / len(dataloader.dataset)
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }
