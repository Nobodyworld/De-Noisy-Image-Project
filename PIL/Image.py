"""Simplified image manipulation helpers used by the kata.

The module implements a tiny subset of Pillow so that the rest of the code can
operate on deterministic JSON backed image files.  Images are represented by
the :class:`SimpleImage` class which stores pixels as a nested list structure
``[[[r, g, b], ...], ...]``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


Pixel = Tuple[int, int, int]


def _clamp(value: float) -> int:
    return max(0, min(255, int(round(value))))


@dataclass
class SimpleImage:
    mode: str
    _size: Tuple[int, int]
    _pixels: List[List[Pixel]]

    def __post_init__(self) -> None:
        self._pixels = [
            [tuple(map(int, pixel)) for pixel in row]
            for row in self._pixels
        ]

    # ------------------------------------------------------------------
    @property
    def size(self) -> Tuple[int, int]:
        return self._size

    def copy(self) -> "SimpleImage":
        pixels = [[tuple(pixel) for pixel in row] for row in self._pixels]
        return SimpleImage(self.mode, self._size, pixels)

    # Context manager support -------------------------------------------------
    def __enter__(self) -> "SimpleImage":  # pragma: no cover - convenience
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - convenience
        return None

    # ------------------------------------------------------------------
    def convert(self, mode: str) -> "SimpleImage":
        if mode == self.mode:
            return self.copy()

        width, height = self.size
        if mode == "RGB":
            if self.mode == "L":
                new_pixels: List[List[Pixel]] = []
                for row in self._pixels:
                    new_row: List[Pixel] = []
                    for value in row:
                        if isinstance(value, tuple):
                            grey = sum(value) / 3.0
                        else:
                            grey = float(value)
                        grey_int = _clamp(grey)
                        new_row.append((grey_int, grey_int, grey_int))
                    new_pixels.append(new_row)
                return SimpleImage("RGB", (width, height), new_pixels)
            raise ValueError(f"Unsupported conversion from {self.mode} to {mode}")

        if mode == "L":
            new_pixels_l: List[List[int]] = []
            for row in self._pixels:
                new_row_l: List[int] = []
                for pixel in row:
                    if isinstance(pixel, tuple):
                        grey = sum(pixel) / 3.0
                    else:
                        grey = float(pixel)
                    new_row_l.append(_clamp(grey))
                new_pixels_l.append(new_row_l)
            return SimpleImage("L", (width, height), new_pixels_l)  # type: ignore[arg-type]

        raise ValueError(f"Unsupported mode conversion to {mode}")

    # ------------------------------------------------------------------
    def resize(self, size: Tuple[int, int], resample: int | None = None) -> "SimpleImage":
        width, height = self.size
        new_width, new_height = size
        if new_width <= 0 or new_height <= 0:
            raise ValueError("Image dimensions must be positive")

        new_pixels: List[List[Pixel]] = []
        for y in range(new_height):
            row: List[Pixel] = []
            src_y = min(int(round(y * (height / new_height))), height - 1)
            for x in range(new_width):
                src_x = min(int(round(x * (width / new_width))), width - 1)
                row.append(self._pixels[src_y][src_x])
            new_pixels.append(row)
        return SimpleImage(self.mode, (new_width, new_height), new_pixels)

    # ------------------------------------------------------------------
    def tobytes(self) -> bytes:
        width, height = self.size
        if self.mode == "RGB":
            flat: List[int] = []
            for row in self._pixels:
                for r, g, b in row:
                    flat.extend([r, g, b])
        elif self.mode == "L":
            flat = [int(value) for row in self._pixels for value in row]
        else:  # pragma: no cover - defensive programming
            raise ValueError(f"Unsupported mode: {self.mode}")
        return bytes(flat)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        data = {
            "mode": self.mode,
            "size": list(self.size),
            "pixels": self._pixels,
        }
        Path(path).write_text(json.dumps(data), encoding="utf8")


def open(path: str | Path) -> SimpleImage:
    payload = json.loads(Path(path).read_text(encoding="utf8"))
    mode = payload["mode"]
    width, height = payload["size"]
    pixels = payload["pixels"]
    return SimpleImage(mode, (int(width), int(height)), pixels)


def frombytes(mode: str, size: Sequence[int], data: bytes) -> SimpleImage:
    width, height = int(size[0]), int(size[1])
    if mode == "RGB":
        expected = width * height * 3
        if len(data) != expected:
            raise ValueError("Byte data does not match image dimensions")
        iterator = iter(data)
        pixels: List[List[Pixel]] = []
        for _ in range(height):
            row: List[Pixel] = []
            for _ in range(width):
                row.append((next(iterator), next(iterator), next(iterator)))
            pixels.append(row)
        return SimpleImage("RGB", (width, height), pixels)
    if mode == "L":
        expected = width * height
        if len(data) != expected:
            raise ValueError("Byte data does not match image dimensions")
        iterator = iter(data)
        pixels_l = [[next(iterator) for _ in range(width)] for _ in range(height)]
        return SimpleImage("L", (width, height), pixels_l)  # type: ignore[arg-type]
    raise ValueError(f"Unsupported mode: {mode}")


# Pillow exposes ``Image.Resampling.BILINEAR``.  A similar enum is not required
# here, but we provide a placeholder object so client code can keep referring to
# it.  The resize implementation above uses a very small nearest neighbour
# scheme regardless of the requested resampling algorithm which is sufficient
# for the purposes of the kata.
class _Resampling:
    BILINEAR = 0


Resampling = _Resampling


__all__ = ["SimpleImage", "open", "frombytes", "Resampling"]

