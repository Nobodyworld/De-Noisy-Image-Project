"""Simple image enhancement helpers used by the tests."""

from __future__ import annotations

from typing import Callable

from .Image import SimpleImage, _clamp


class _BaseEnhancer:
    def __init__(self, image: SimpleImage) -> None:
        self.image = image

    def _apply(self, func: Callable[[int], int]) -> SimpleImage:
        pixels = []
        for row in self.image._pixels:
            new_row = []
            for value in row:
                if isinstance(value, tuple):
                    new_row.append(tuple(func(channel) for channel in value))
                else:
                    new_row.append(func(int(value)))
            pixels.append(new_row)
        return SimpleImage(self.image.mode, self.image.size, pixels)


class Brightness(_BaseEnhancer):
    def enhance(self, factor: float) -> SimpleImage:
        return self._apply(lambda px: _clamp(px * factor))


class Contrast(_BaseEnhancer):
    def enhance(self, factor: float) -> SimpleImage:
        midpoint = 128
        return self._apply(lambda px: _clamp(midpoint + (px - midpoint) * factor))


class Color(_BaseEnhancer):
    def enhance(self, factor: float) -> SimpleImage:
        return Brightness(self.image).enhance(factor)


__all__ = ["Brightness", "Contrast", "Color"]

