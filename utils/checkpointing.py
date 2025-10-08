# /utils/checkpointing.py
import os
from typing import Any, Callable

import torch


def _resolve_loader(device):
    """Return a callable that can load a state dict from disk.

    The kata ships with a lightweight torch subset that exposes ``load`` and
    ``save_state_dict`` helpers producing JSON files.  When running against a
    full PyTorch installation we fall back to ``torch.load`` and ``torch.save``.
    """

    if hasattr(torch, "load"):
        signature = getattr(torch.load, "__code__", None)

        def _load(path: str):
            kwargs = {}
            if signature and "map_location" in signature.co_varnames:
                kwargs["map_location"] = device
            return torch.load(path, **kwargs)

        return _load

    raise RuntimeError("No torch-compatible load helper is available")


def _resolve_saver() -> Callable[[Any, str], None]:
    if hasattr(torch, "save_state_dict"):
        return torch.save_state_dict
    if hasattr(torch, "save"):
        return torch.save
    raise RuntimeError("No torch-compatible save helper is available")


def save_checkpoint(state_dict, filename="other_model.json"):
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    saver = _resolve_saver()
    saver(state_dict, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, filename, device):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    loader = _resolve_loader(device)
    checkpoint = loader(filename)
    model.load_state_dict(checkpoint)
    print(f"Checkpoint loaded from {filename}")
