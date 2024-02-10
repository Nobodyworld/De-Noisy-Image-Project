# /utils/chekpointing.py
import torch

def save_checkpoint(state_dict, filename='best_psnr_denocoder_pytorch.pth'):
    torch.save(state_dict, filename)

def load_checkpoint(model, filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

