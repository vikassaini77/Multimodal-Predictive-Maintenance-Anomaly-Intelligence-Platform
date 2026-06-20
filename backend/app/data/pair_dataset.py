import torch
from torch.utils.data import Dataset
import random
from typing import Tuple

class PairDataset(Dataset):
    """
    Multimodal Dataset that aligns sensor (e.g. MIMII) and visual (e.g. MVTec AD) data.
    Groups indices by their binary label (normal=0, anomaly=1).
    During fetch, randomly pairs a sensor sequence and an image of the same label.
    """
    def __init__(self, sensor_dataset: Dataset, visual_dataset: Dataset):
        self.sensor_dataset = sensor_dataset
        self.visual_dataset = visual_dataset
        
        # Group indices by label
        self.sensor_normal_idx = []
        self.sensor_anomaly_idx = []
        
        # We assume dataset__getitem__ returns (data, label) or (data, mask, label)
        # We will peek at labels if they are accessible, else we will have to build an index
        self._build_index()
        
    def _build_index(self):
        print("Building PairDataset index for sensor dataset...")
        for i in range(len(self.sensor_dataset)):
            # Assuming sensor dataset has a `.labels` attribute as in MIMIIDataset
            if hasattr(self.sensor_dataset, 'labels'):
                label = self.sensor_dataset.labels[i]
            else:
                # Fallback: get item (slow but robust)
                item = self.sensor_dataset[i]
                label = item[-1].item() if isinstance(item[-1], torch.Tensor) else item[-1]
                
            if label == 0:
                self.sensor_normal_idx.append(i)
            else:
                self.sensor_anomaly_idx.append(i)
                
        print("Building PairDataset index for visual dataset...")
        self.visual_normal_idx = []
        self.visual_anomaly_idx = []
        
        for i in range(len(self.visual_dataset)):
            if hasattr(self.visual_dataset, 'labels'):
                label = self.visual_dataset.labels[i]
            else:
                item = self.visual_dataset[i]
                label = item[-1].item() if isinstance(item[-1], torch.Tensor) else item[-1]
                
            if label == 0:
                self.visual_normal_idx.append(i)
            else:
                self.visual_anomaly_idx.append(i)
                
        self.length = max(len(self.sensor_dataset), len(self.visual_dataset))
        
    def __len__(self) -> int:
        return self.length
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # Decide if we yield a normal or anomalous pair
        # We will roughly keep the ratio of normal vs anomalous
        # But for contrastive learning, typically we want a balanced batch, or just random
        # Let's randomly pick based on available data, or just use the index's label from sensor data
        
        # To ensure all data is eventually seen, we can wrap the index
        sensor_idx = idx % len(self.sensor_dataset)
        
        # Get the label from the sensor data
        if hasattr(self.sensor_dataset, 'labels'):
            label = self.sensor_dataset.labels[sensor_idx]
        else:
            item = self.sensor_dataset[sensor_idx]
            label = item[-1].item() if isinstance(item[-1], torch.Tensor) else item[-1]
            
        # Fetch the sensor data
        sensor_item = self.sensor_dataset[sensor_idx]
        sensor_data = sensor_item[0] # assuming first element is the data tensor
        
        # Pick a random visual sample with the SAME label
        if label == 0:
            if not self.visual_normal_idx:
                raise ValueError("No normal data in visual dataset.")
            vis_idx = random.choice(self.visual_normal_idx)
        else:
            if not self.visual_anomaly_idx:
                raise ValueError("No anomalous data in visual dataset.")
            vis_idx = random.choice(self.visual_anomaly_idx)
            
        visual_item = self.visual_dataset[vis_idx]
        visual_data = visual_item[0] # assuming first element is the image tensor
        
        # Return paired data and the shared label
        return sensor_data, visual_data, label
