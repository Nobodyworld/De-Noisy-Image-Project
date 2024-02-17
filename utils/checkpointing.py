# /utils/checkpointing.py
import torch
import os

def save_checkpoint(state_dict, filename='other_model.pth'):
    """
    Saves the model's state dictionary (weights) to a file.
    
    Args:
        state_dict: Model's state dictionary.
        filename (str, optional): File path where the state dict will be saved.
    """
    # Ensure directory exists (extract directory from filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state_dict, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, filename, device):
    """
    Loads the model's state dictionary (weights) from a file into the model.
    
    Args:
        model: Model instance where the state dict will be loaded.
        filename (str): File path from where the state dict will be loaded.
        device: The device to load the state dict on.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")
    
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Checkpoint loaded from {filename}")
