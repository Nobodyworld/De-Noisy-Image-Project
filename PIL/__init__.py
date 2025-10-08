"""Very small, pure Python stand in for :mod:`PIL`.

Only the behaviour used throughout the kata is implemented which keeps the
module intentionally small and easy to audit.  The :mod:`PIL.Image` module
understands simple image files that are stored as JSON.  The JSON structure is

``{"mode": "RGB", "size": [width, height], "pixels": [[[r, g, b], ...], ...]}``

where the pixel values are expressed as integers in the inclusive range
``[0, 255]``.  Files use a ``.json`` extension so that platforms which disallow
binary files can still inspect the sample assets while exercising the expected
code paths.
"""

from . import Image  # noqa: F401  (re-export for ``from PIL import Image``)

__all__ = ["Image"]

