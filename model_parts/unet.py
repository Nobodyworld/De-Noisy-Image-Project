"""A tiny, deterministic U-Net inspired model built on the torch stub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class _Block(torch.nn.Module):
    """Small helper block applying an affine transformation."""

    weight: torch.Parameter
    bias: torch.Parameter

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.weight + self.bias


class UNet(torch.nn.Module):
    """Simplified U-Net style network.

    The class keeps the public surface of a conventional PyTorch module but the
    implementation is intentionally tiny: the *encoder*, *mid* and *decoder*
    stages are represented by affine blocks that operate element wise.  This is
    sufficient for the accompanying unit tests which focus on parameter loading
    and tensor plumbing rather than deep learning performance.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Block(torch.Parameter([1.0]), torch.Parameter([0.0]))
        self.mid = _Block(torch.Parameter([1.0]), torch.Parameter([0.0]))
        self.decoder = _Block(torch.Parameter([1.0]), torch.Parameter([0.0]))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(tensor)
        mid = self.mid(encoded)
        return self.decoder(mid)

    # PyTorch style convenience wrappers -------------------------------------------------
    def state_dict(self) -> Dict[str, torch.Tensor]:  # pragma: no cover - delegated
        return {
            "encoder.weight": self.encoder.weight.clone(),
            "encoder.bias": self.encoder.bias.clone(),
            "mid.weight": self.mid.weight.clone(),
            "mid.bias": self.mid.bias.clone(),
            "decoder.weight": self.decoder.weight.clone(),
            "decoder.bias": self.decoder.bias.clone(),
        }

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> "UNet":
        for key, value in state.items():
            module_name, attr = key.split(".")
            module = getattr(self, module_name)
            setattr(module, attr, torch.Parameter(value.clone().tolist()))
        return self

    def parameters(self):  # pragma: no cover - delegate helper
        for module in (self.encoder, self.mid, self.decoder):
            yield module.weight
            yield module.bias
