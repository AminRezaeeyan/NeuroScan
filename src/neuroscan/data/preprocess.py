"""Image preprocessing shared by training, evaluation and serving.

Every model takes raw RGB pixels in [0, 255] at ``image_size x image_size`` and
does its own scaling inside, so this module only decodes and resizes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

DEFAULT_IMAGE_SIZE = 224


def decode_and_resize(image_bytes: tf.Tensor, image_size: int = DEFAULT_IMAGE_SIZE) -> tf.Tensor:
    """Encoded image bytes (JPEG/PNG/BMP/GIF) -> float32 tensor (size, size, 3) in [0, 255].

    Grayscale is replicated to 3 channels and alpha is dropped by ``channels=3``.
    """
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, (image_size, image_size), method="bilinear", antialias=True)
    return tf.clip_by_value(image, 0.0, 255.0)


def preprocess_bytes(image_bytes: bytes, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Serving entry point: bytes -> numpy array (size, size, 3), same ops as training."""
    return decode_and_resize(tf.constant(image_bytes), image_size).numpy()


def preprocess_file(path: str | Path, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    return preprocess_bytes(Path(path).read_bytes(), image_size)
