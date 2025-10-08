# /utils/transforms.py
"""Lightweight image transform utilities that avoid heavy dependencies."""

from __future__ import annotations

import random
from typing import Callable, Iterable, Tuple

from PIL import Image, ImageEnhance
import torch


class Compose:
    """Apply a sequence of callables in order."""

    def __init__(self, transforms: Iterable[Callable[[Image.Image], Image.Image | torch.Tensor]]):
        self.transforms = list(transforms)

    def __call__(self, image: Image.Image) -> Image.Image | torch.Tensor:
        result: Image.Image | torch.Tensor = image
        for transform in self.transforms:
            result = transform(result)
        return result


class Resize:
    """Resize a PIL image to the specified (width, height)."""

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, image: Image.SimpleImage) -> Image.SimpleImage:
        return image.resize(self.size)


def _pil_to_tensor(image: Image.SimpleImage) -> torch.Tensor:
    """Convert a PIL image to a normalised float tensor without NumPy."""

    processed = image.convert("RGB")
    width, height = processed.size
    data = list(processed.tobytes())

    array = []
    index = 0
    for channel in range(3):
        channel_values = []
        for _ in range(height):
            row = []
            for _ in range(width):
                row.append(data[index] / 255.0)
                index += 1
            channel_values.append(row)
        array.append(channel_values)
    return torch.tensor(array, dtype=torch.float32)


class ToTensor:
    """Convert a PIL image to a PyTorch tensor."""

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return _pil_to_tensor(image)


def tensor_to_pil(tensor: torch.Tensor) -> Image.SimpleImage:
    """Convert a CHW tensor in the range [0, 1] back to a PIL image."""

    if tensor.dim() != 3:
        raise ValueError("Expected a 3D tensor in CHW format")

    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    channels, height, width = tensor.shape
    byte_tensor = (tensor * 255).to(dtype=torch.uint8)

    if channels == 1:
        mode = "L"
        flat = byte_tensor.view(-1)
    elif channels == 3:
        mode = "RGB"
        flat = byte_tensor.permute(1, 2, 0).contiguous().view(-1)
    else:
        raise ValueError(f"Unsupported number of channels: {channels}")

    data = bytes(flat.tolist())
    return Image.frombytes(mode, (width, height), data)


class RandomColorJitterWithRandomFactors:
    """Apply colour jitter with individual random factors per call."""

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
        p: float = 0.0,
    ) -> None:
        self.brightness = max(0.0, float(brightness))
        self.contrast = max(0.0, float(contrast))
        self.saturation = max(0.0, float(saturation))
        self.hue = max(0.0, float(hue))
        self.p = max(0.0, min(1.0, float(p)))

    def _sample_factor(self, value: float) -> float:
        if value <= 0.0:
            return 1.0
        low = max(0.0, 1.0 - value)
        high = 1.0 + value
        return random.uniform(low, high)

    def __call__(self, image: Image.SimpleImage) -> Image.SimpleImage:
        if random.random() > self.p:
            return image

        result = image
        if self.brightness > 0.0:
            factor = self._sample_factor(self.brightness)
            result = ImageEnhance.Brightness(result).enhance(factor)
        if self.contrast > 0.0:
            factor = self._sample_factor(self.contrast)
            result = ImageEnhance.Contrast(result).enhance(factor)
        if self.saturation > 0.0:
            factor = self._sample_factor(self.saturation)
            result = ImageEnhance.Color(result).enhance(factor)
        # Hue adjustments are intentionally ignored – the simplified colour model
        # used by SimpleImage does not encode hue separately.  The parameter is
        # accepted to maintain API compatibility with the configuration format.
        return result


def get_transforms(config: dict) -> Compose:
    """Create a transform pipeline based on the configuration dictionary."""

    width = config['training']['img_width']
    height = config['training']['img_height']
    transforms: list[Callable[[Image.Image], Image.Image | torch.Tensor]] = [Resize((width, height))]

    jitter_cfg = config.get('augmentation', {}).get('color_jitter', {})
    if jitter_cfg.get('enabled'):
        transforms.append(
            RandomColorJitterWithRandomFactors(
                brightness=jitter_cfg.get('brightness', 0.0),
                contrast=jitter_cfg.get('contrast', 0.0),
                saturation=jitter_cfg.get('saturation', 0.0),
                hue=jitter_cfg.get('hue', 0.0),
                p=jitter_cfg.get('p', 0.0),
            )
        )

    transforms.append(ToTensor())
    return Compose(transforms)


__all__ = [
    "Compose",
    "Resize",
    "ToTensor",
    "tensor_to_pil",
    "RandomColorJitterWithRandomFactors",
    "get_transforms",
]
