import pytest
from utils.data_loading import get_dataloaders
from utils.pairedimage_dataset import PairedImageDataset
import torch

def test_dataloaders():
    # Mock configuration
    config = {
        "directories": {
            "data": {
                "train": "./data/train",
                "val": "./data/val",
                "test": "./data/test"
            }
        },
        "training": {
            "batch_size": 4,
            "img_height": 960,
            "img_width": 640
        }
    }
    
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # Check if DataLoader objects are created
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)
    
    # Check batch size
    for loader in [train_loader, val_loader, test_loader]:
        for images, labels in loader:
            assert len(images) == config['training']['batch_size']
            break  # Only check the first batch

@pytest.mark.parametrize("path", ["./data/train", "./data/val", "./data/test"])
def test_pairedimage_dataset(path):
    dataset = PairedImageDataset(directory=path)
    assert len(dataset) > 0  # Assuming there's at least one image pair in each directory
    before_image, after_image = dataset[0]
    assert before_image.size() == torch.Size([3, 640, 960])  # Assuming RGB images
    assert after_image.size() == torch.Size([3, 640, 960])  # And they're resized correctly
