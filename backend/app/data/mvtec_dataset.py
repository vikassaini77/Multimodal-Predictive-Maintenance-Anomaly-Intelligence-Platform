import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Tuple, Callable, Optional

class MVTecDataset(Dataset):
    """
    MVTec AD Dataset with mask-aware loading.
    Returns (image, mask, label) where label is 0 (normal) or 1 (anomaly).
    Mask is 0 for normal pixels, and >0 for anomalous pixels.
    """
    def __init__(self, root_dir: str, category: str, split: str = "train", transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir) / category
        self.split = split
        self.transform = transform
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset category directory not found: {self.root_dir}")
            
        self.images = []
        self.masks = []
        self.labels = []
        
        self._load_dataset()
        
    def _load_dataset(self):
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            return
            
        # Normal images
        good_dir = split_dir / "good"
        if good_dir.exists():
            for img_path in sorted(good_dir.glob("*.png")):
                self.images.append(str(img_path))
                self.masks.append(None) # Normal has no mask file
                self.labels.append(0)   # 0 for normal
                
        # Defect images (only present in test split usually, but let's be general)
        if self.split == "test":
            for defect_dir in split_dir.iterdir():
                if defect_dir.is_dir() and defect_dir.name != "good":
                    mask_dir = self.root_dir / "ground_truth" / defect_dir.name
                    for img_path in sorted(defect_dir.glob("*.png")):
                        self.images.append(str(img_path))
                        # Mask filenames usually match image filenames ending with _mask.png
                        mask_name = f"{img_path.stem}_mask.png"
                        mask_path = mask_dir / mask_name
                        self.masks.append(str(mask_path) if mask_path.exists() else None)
                        self.labels.append(1)   # 1 for anomaly

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]
        mask_path = self.masks[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Load mask or create a blank one if normal
        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.new("L", image.size, 0)
            
        if self.transform:
            # Note: A real implementation would need a joint transform for both image and mask
            # to preserve spatial alignment during random crops/flips. 
            # Assuming a custom transform or torchvision v2 transforms are used.
            image = self.transform(image)
            # Basic tensor conversion for mask if transform doesn't handle it
            import torchvision.transforms.functional as F
            mask = F.to_tensor(mask)
        else:
            import torchvision.transforms.functional as F
            image = F.to_tensor(image)
            mask = F.to_tensor(mask)
            
        return image, mask, label
