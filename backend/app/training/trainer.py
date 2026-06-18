import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import os

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
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step()
            
            print(f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            # Checkpointing
            checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': val_loss,
            }, checkpoint_path)
