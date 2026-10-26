"""Cheap input checks that run before the model.

MRI slices are grayscale. In the training data no image has more than 4.4% of
pixels with a channel spread above 30 (a few carry colored annotations), so an
image with more than 10% such pixels is treated as not an MRI.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

COLOR_SPREAD = 30  # max - min over RGB channels for a pixel to count as colored
MAX_COLORED_FRACTION = 0.10
MIN_SIDE = 64  # pixels
MAX_SIDE = 8000


def colored_fraction(image: np.ndarray) -> float:
    """Share of pixels whose RGB channels differ by more than COLOR_SPREAD."""
    if image.ndim < 3 or image.shape[-1] == 1:
        return 0.0
    spread = image[..., :3].max(axis=-1).astype(np.float32) - image[..., :3].min(axis=-1)
    return float((spread > COLOR_SPREAD).mean())


def check_array(image: np.ndarray) -> str | None:
    """Reason to reject a preprocessed (H, W, 3) image, or None if it passes."""
    if colored_fraction(image) > MAX_COLORED_FRACTION:
        return "not_grayscale"
    return None


def check_bytes(data: bytes) -> str | None:
    """Reason to reject an uploaded file, or None: it must decode as a sensibly sized image."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
            arr = np.asarray(img.convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return "not_an_image"
    if min(width, height) < MIN_SIDE or max(width, height) > MAX_SIDE:
        return "bad_size"
    return check_array(arr)
