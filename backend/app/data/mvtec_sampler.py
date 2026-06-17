import torch
from torch.utils.data import WeightedRandomSampler, Dataset
import numpy as np

def get_balanced_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """
    Implement class-balanced WeightedRandomSampler.
    Handles severe normal/anomaly class imbalance in MVTec.
    """
    labels = []
    # Try to access labels directly if possible to avoid loading all images
    if hasattr(dataset, 'labels'):
        labels = dataset.labels
    else:
        for idx in range(len(dataset)):
            _, _, label = dataset[idx]
            labels.append(label)
            
    labels = np.array(labels)
    class_counts = np.bincount(labels)
    
    # Calculate weights for each class
    # To handle potential zero counts, add a small epsilon or max with 1
    class_weights = 1.0 / np.maximum(class_counts, 1)
    
    # Assign weight to each sample
    sample_weights = [class_weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler
