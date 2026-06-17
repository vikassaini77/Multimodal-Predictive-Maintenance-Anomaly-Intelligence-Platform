import pytest
import torch
import tempfile
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from PIL import Image
from app.data.mvtec_dataset import MVTecDataset
from app.data.mvtec_transforms import get_train_transforms, get_test_transforms, get_tta_transforms

@pytest.fixture
def mock_mvtec_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "bottle"
        
        # Train split
        train_good = base_path / "train" / "good"
        train_good.mkdir(parents=True)
        img = Image.new('RGB', (300, 300), color = 'red')
        img.save(train_good / "000.png")
        
        # Test split
        test_good = base_path / "test" / "good"
        test_good.mkdir(parents=True)
        img.save(test_good / "000.png")
        
        test_defect = base_path / "test" / "broken"
        test_defect.mkdir(parents=True)
        img.save(test_defect / "001.png")
        
        # Ground truth
        gt_dir = base_path / "ground_truth" / "broken"
        gt_dir.mkdir(parents=True)
        mask = Image.new('L', (300, 300), color = 255)
        mask.save(gt_dir / "001_mask.png")
        
        yield tmpdir

def test_mvtec_dataset_shapes_and_labels(mock_mvtec_dir):
    # Test train split (normals only)
    train_dataset = MVTecDataset(root_dir=mock_mvtec_dir, category="bottle", split="train")
    assert len(train_dataset) == 1
    
    img, mask, label = train_dataset[0]
    assert img.shape == (3, 300, 300)
    assert mask.shape == (1, 300, 300)
    assert torch.all(mask == 0) # Normal should have empty mask
    assert label == 0
    
    # Test test split (normals and anomalies)
    test_dataset = MVTecDataset(root_dir=mock_mvtec_dir, category="bottle", split="test")
    assert len(test_dataset) == 2
    
    # Verify label consistency
    labels = test_dataset.labels
    assert labels.count(0) == 1
    assert labels.count(1) == 1

def test_transforms():
    img = Image.new('RGB', (300, 300), color = 'red')
    
    # Train transform output
    train_tf = get_train_transforms()
    out = train_tf(img)
    assert out.shape == (3, 224, 224)
    
    # Test transform output
    test_tf = get_test_transforms()
    out = test_tf(img)
    assert out.shape == (3, 224, 224)
    
    # TTA output
    tta_tf = get_tta_transforms()
    out = tta_tf(img)
    assert out.shape == (5, 3, 224, 224) # 5 crops
