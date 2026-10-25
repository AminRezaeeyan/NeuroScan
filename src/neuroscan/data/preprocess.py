"""Image preprocessing shared by training, evaluation and serving.

Every model takes raw RGB pixels in [0, 255] at ``image_size x image_size`` and
does its own scaling inside. Two preprocessing modes exist:

- ``standard``: decode and resize; nothing else.
- ``shortcut_reduced``: crop to the head and normalize intensity per image before
  resizing. This removes framing, head size, brightness and contrast differences,
  which differ between the dataset's sources and partly reveal the class.

Both are pure TensorFlow, so the ``tf.data`` training pipeline and the API run the
exact same operations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

DEFAULT_IMAGE_SIZE = 224
PREPROCESSING_MODES = ("standard", "shortcut_reduced")

HEAD_THRESHOLD = 0.10  # share of the (smoothed) maximum intensity that counts as head
CROP_MARGIN = 0.04  # added on each side, as a share of the head's extent
LOW_PERCENTILE, HIGH_PERCENTILE = 1.0, 99.0


def check_mode(mode: str) -> str:
    if mode not in PREPROCESSING_MODES:
        raise ValueError(f"Unknown preprocessing {mode!r}; choose from {PREPROCESSING_MODES}")
    return mode


def crop_to_head(image: tf.Tensor) -> tf.Tensor:
    """Bounding box of pixels brighter than HEAD_THRESHOLD x max, plus a margin.

    Smoothing first ignores isolated bright pixels (text, noise). An image with no
    such pixels is returned unchanged.
    """
    gray = tf.reduce_mean(image, axis=-1)
    smooth = tf.nn.avg_pool2d(gray[None, :, :, None], 5, 1, "SAME")[0, :, :, 0]
    mask = smooth > HEAD_THRESHOLD * tf.reduce_max(smooth)
    rows = tf.cast(tf.where(tf.reduce_any(mask, axis=1))[:, 0], tf.int32)
    cols = tf.cast(tf.where(tf.reduce_any(mask, axis=0))[:, 0], tf.int32)
    height, width = tf.shape(gray)[0], tf.shape(gray)[1]

    def crop() -> tf.Tensor:
        top, bottom = tf.reduce_min(rows), tf.reduce_max(rows)
        left, right = tf.reduce_min(cols), tf.reduce_max(cols)
        dy = tf.cast(tf.cast(bottom - top + 1, tf.float32) * CROP_MARGIN, tf.int32)
        dx = tf.cast(tf.cast(right - left + 1, tf.float32) * CROP_MARGIN, tf.int32)
        top, left = tf.maximum(top - dy, 0), tf.maximum(left - dx, 0)
        bottom, right = tf.minimum(bottom + dy, height - 1), tf.minimum(right + dx, width - 1)
        return tf.image.crop_to_bounding_box(image, top, left, bottom - top + 1, right - left + 1)

    return tf.cond(tf.size(rows) > 0, crop, lambda: image)


def normalize_intensity(image: tf.Tensor) -> tf.Tensor:
    """Map the 1st-99th intensity percentiles of the image to [0, 255] (all channels alike)."""
    gray = tf.sort(tf.reshape(tf.reduce_mean(image, axis=-1), [-1]))
    last = tf.cast(tf.size(gray) - 1, tf.float32)
    low = gray[tf.cast(last * LOW_PERCENTILE / 100, tf.int32)]
    high = gray[tf.cast(last * HIGH_PERCENTILE / 100, tf.int32)]
    scale = tf.maximum(high - low, 1.0)
    return tf.clip_by_value((image - low) / scale, 0.0, 1.0) * 255.0


def decode_and_resize(
    image_bytes: tf.Tensor, image_size: int = DEFAULT_IMAGE_SIZE, mode: str = "standard"
) -> tf.Tensor:
    """Encoded image bytes (JPEG/PNG/BMP/GIF) -> float32 tensor (size, size, 3) in [0, 255].

    Grayscale is replicated to 3 channels and alpha is dropped by ``channels=3``.
    """
    check_mode(mode)
    image = tf.cast(
        tf.io.decode_image(image_bytes, channels=3, expand_animations=False), tf.float32
    )
    if mode == "shortcut_reduced":
        image = normalize_intensity(crop_to_head(image))
    image = tf.image.resize(image, (image_size, image_size), method="bilinear", antialias=True)
    return tf.clip_by_value(image, 0.0, 255.0)


def preprocess_bytes(
    image_bytes: bytes, image_size: int = DEFAULT_IMAGE_SIZE, mode: str = "standard"
) -> np.ndarray:
    """Serving entry point: bytes -> numpy array (size, size, 3), same ops as training."""
    return decode_and_resize(tf.constant(image_bytes), image_size, mode).numpy()


def preprocess_file(
    path: str | Path, image_size: int = DEFAULT_IMAGE_SIZE, mode: str = "standard"
) -> np.ndarray:
    return preprocess_bytes(Path(path).read_bytes(), image_size, mode)
