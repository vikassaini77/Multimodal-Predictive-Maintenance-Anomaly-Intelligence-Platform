import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import os
import mlflow
import math
from sklearn.metrics import roc_auc_score

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        epochs: int = 10,
        lr: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = train_loader.batch_size if hasattr(train_loader, 'batch_size') else 32
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.criterion = nn.MSELoss() # Assuming regression for RMSE target
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.scaler = GradScaler()
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                output = self.model(data)
                loss = self.criterion(output, target)
                
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
        
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
        
    def train(self):
        mlflow.log_params({
            "epochs": self.epochs,
            "learning_rate": self.lr,
            "batch_size": self.batch_size
        })
        
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            # Also calculate RMSE
            val_rmse = math.sqrt(val_loss)
            
            self.scheduler.step()
            
            print(f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val RMSE: {val_rmse:.4f}")
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "lr": self.scheduler.get_last_lr()[0]
            }, step=epoch)
            
            # Checkpointing
            checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': val_loss,
            }, checkpoint_path)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in anomaly detection (e.g. MVTec AD).
    Focuses training on hard-to-classify examples.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class AnomalyTrainer:
    """
    Trainer for binary classification anomaly detection models (e.g. VisualTowerClassifier).
    Implements Focal Loss, AUROC tracking, and Early Stopping.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        epochs: int = 50,
        lr: float = 1e-4,
        patience: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints_anomaly"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = train_loader.batch_size if hasattr(train_loader, 'batch_size') else 32
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.criterion = FocalLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.scaler = GradScaler()
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        for data, mask, target in self.train_loader:
            # MVTecDataset returns (image, mask, label)
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                output = self.model(data)
                loss = self.criterion(output, target)
                
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
        
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_preds = []
        
        with torch.no_grad():
            for data, mask, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                total_loss += loss.item()
                
                probs = torch.sigmoid(output)
                all_targets.extend(target.cpu().numpy())
                all_preds.extend(probs.cpu().numpy())
                
        # Calculate AUROC if we have both classes, else default to 0.5
        if len(set(all_targets)) > 1:
            auroc = roc_auc_score(all_targets, all_preds)
        else:
            auroc = 0.5
            
        return total_loss / len(self.val_loader), auroc
        
    def train(self):
        mlflow.log_params({
            "epochs": self.epochs,
            "learning_rate": self.lr,
            "batch_size": self.batch_size,
            "patience": self.patience
        })
        
        best_auroc = 0.0
        patience_counter = 0
        
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_auroc = self.validate()
            
            self.scheduler.step()
            
            print(f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val AUROC: {val_auroc:.4f}")
            
            mlflow.log_metrics({
                "anomaly_train_loss": train_loss,
                "anomaly_val_loss": val_loss,
                "anomaly_val_auroc": val_auroc,
                "lr": self.scheduler.get_last_lr()[0]
            }, step=epoch)
            
            # Early stopping and checkpointing based on AUROC
            if val_auroc > best_auroc:
                best_auroc = val_auroc
                patience_counter = 0
                
                checkpoint_path = os.path.join(self.checkpoint_dir, "best_anomaly_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auroc': val_auroc,
                }, checkpoint_path)
                print(f"[*] New best model saved with AUROC: {val_auroc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch} epochs. Best AUROC: {best_auroc:.4f}")
                    break

from backend.app.models.two_tower import TwoTowerAnomalyModel, NTXentLoss

class TwoTowerContrastiveTrainer:
    """
    Trainer for TwoTowerAnomalyModel using NT-Xent contrastive loss.
    Trains the sensor and visual towers jointly.
    """
    def __init__(
        self,
        model: TwoTowerAnomalyModel,
        train_loader,
        val_loader,
        epochs: int = 50,
        sensor_lr: float = 3e-4,
        visual_lr: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints_two_tower"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.sensor_lr = sensor_lr
        self.visual_lr = visual_lr
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.criterion = NTXentLoss()
        
        # Separate learning rates for the two towers
        # Visual tower uses a smaller LR as it's fine-tuning a pre-trained EfficientNet
        # Sensor tower uses a larger LR as it's trained from scratch
        optimizer_groups = [
            {'params': self.model.sensor_tower.parameters(), 'lr': self.sensor_lr},
            {'params': self.model.visual_tower.parameters(), 'lr': self.visual_lr},
            {'params': [self.model.temperature], 'lr': self.sensor_lr}
        ]
        
        self.optimizer = AdamW(optimizer_groups, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.scaler = GradScaler()
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        for sensor_data, visual_data, _ in self.train_loader:
            sensor_data = sensor_data.to(self.device)
            visual_data = visual_data.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                out = self.model(sensor_data, visual_data)
                loss = self.criterion(out['sensor_embeddings'], out['visual_embeddings'], out['temperature'])
                
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
        
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for sensor_data, visual_data, _ in self.val_loader:
                sensor_data = sensor_data.to(self.device)
                visual_data = visual_data.to(self.device)
                
                with autocast():
                    out = self.model(sensor_data, visual_data)
                    loss = self.criterion(out['sensor_embeddings'], out['visual_embeddings'], out['temperature'])
                    
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
        
    def train(self):
        mlflow.log_params({
            "epochs": self.epochs,
            "sensor_lr": self.sensor_lr,
            "visual_lr": self.visual_lr,
        })
        
        best_loss = float('inf')
        
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            self.scheduler.step()
            
            print(f"Epoch {epoch}/{self.epochs} - Train Contrastive Loss: {train_loss:.4f} - Val Contrastive Loss: {val_loss:.4f} - Temp: {torch.clamp(self.model.temperature.exp(), max=100.0).item():.4f}")
            
            mlflow.log_metrics({
                "contrastive_train_loss": train_loss,
                "contrastive_val_loss": val_loss,
                "sensor_lr": self.optimizer.param_groups[0]['lr'],
                "visual_lr": self.optimizer.param_groups[1]['lr'],
                "temperature": torch.clamp(self.model.temperature.exp(), max=100.0).item()
            }, step=epoch)
            
            if val_loss < best_loss:
                best_loss = val_loss
                
                checkpoint_path = os.path.join(self.checkpoint_dir, "best_twotower_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"[*] New best model saved with Loss: {val_loss:.4f}")
