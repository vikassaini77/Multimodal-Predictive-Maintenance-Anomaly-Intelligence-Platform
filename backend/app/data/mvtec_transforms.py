import torch
from torchvision import transforms

def get_train_transforms():
    """
    ImageNet-style normalization transform pipeline for training.
    Includes Resize(256), RandomCrop(224), and ColorJitter.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def get_test_transforms():
    """
    Standard test transforms.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def get_tta_transforms():
    """
    Test-time augmentation (TTA) transform set.
    Uses FiveCrop + horizontal flip for robust inference.
    Returns a list of transforms that can be applied to create an ensemble.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.FiveCrop(224),
        transforms.Lambda(lambda crops: torch.stack([
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(crop) for crop in crops
        ]))
    ])
