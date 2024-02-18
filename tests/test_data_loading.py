# tests/test_data_loading.py
import pytest
from utils.data_loading import get_dataloaders
from utils.pairedimage_dataset import PairedImageDataset
import torch
import os

config = os.path.join('./config/config.json')

def test_dataloaders():
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # Verify DataLoader creation and batch size
    for loader in [train_loader, val_loader, test_loader]:
        for images, labels in loader:
            assert isinstance(loader, torch.utils.data.DataLoader)
            assert len(images) == config['training']['batch_size']
            break

@pytest.mark.parametrize("key", ["train", "val", "test"])
def test_pairedimage_dataset(key):
    path = config['directories']['data'][key]
    dataset = PairedImageDataset(directory=path)
    assert len(dataset) > 0
    before_image, after_image = dataset[0]
    # Assuming RGB images of specified dimensions from config
    expected_size = torch.Size([3, config['training']['img_height'], config['training']['img_width']])
    assert before_image.size() == expected_size
    assert after_image.size() == expected_size
