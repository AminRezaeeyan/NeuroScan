"""Pixel-statistics shortcut probe."""

import numpy as np

from neuroscan.shortcut import PIXEL_FEATURES, pixel_stats


def test_pixel_stats():
    image = np.zeros((224, 224, 3), np.float32)
    image[56:168, 56:168] = 200.0
    stats = pixel_stats(image)
    assert list(stats) == PIXEL_FEATURES
    assert stats["center_mean"] == 200.0 and stats["border_mean"] == 0.0
    assert stats["background_frac"] == 0.75
    assert stats["sharpness"] > 0
