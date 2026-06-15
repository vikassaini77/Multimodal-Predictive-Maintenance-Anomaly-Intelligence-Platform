import torch
from torch.utils.data import Dataset
import os

class MIMIIDataset(Dataset):
    def __init__(self, file_paths, labels, audio_loader, spec_extractor):
        self.file_paths = file_paths
        self.labels = labels # normal=0, anomaly=1
        self.audio_loader = audio_loader
        self.spec_extractor = spec_extractor

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load and process
        audio = self.audio_loader.load(path)
        spec = self.spec_extractor.extract(audio)
        
        # Add channel dimension for 1D-CNN or 2D-CNN
        spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return spec_tensor, label_tensor
