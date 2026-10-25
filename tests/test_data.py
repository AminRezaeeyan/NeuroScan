"""Preprocessing and dataset tests on tiny synthetic images."""

import io

import numpy as np
import pytest
from PIL import Image

from neuroscan.config import CLASS_NAMES, Config
from neuroscan.data.datasets import make_dataset, scan_raw
from neuroscan.data.preprocess import preprocess_bytes


def _encode(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.mark.parametrize(
    ("mode", "fmt", "size"),
    [
        ("RGB", "JPEG", (300, 200)),
        ("L", "PNG", (512, 512)),
        ("RGBA", "PNG", (90, 400)),
        ("RGB", "BMP", (224, 224)),
    ],
)
def test_preprocess_shape_dtype_range(mode, fmt, size):
    img = Image.new(mode, size, color=120 if mode == "L" else (120,) * len(mode))
    out = preprocess_bytes(_encode(img, fmt))
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 255.0


def test_grayscale_is_replicated_to_three_equal_channels():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (64, 64), dtype=np.uint8), mode="L")
    out = preprocess_bytes(_encode(img, "PNG"))
    np.testing.assert_array_equal(out[..., 0], out[..., 1])
    np.testing.assert_array_equal(out[..., 0], out[..., 2])


def test_non_image_bytes_raise():
    with pytest.raises(Exception):  # noqa: B017 - TF raises InvalidArgumentError
        preprocess_bytes(b"definitely not an image")


def test_scan_raw(raw_dir):
    df = scan_raw(raw_dir)
    assert len(df) == 16
    assert set(df["label"]) == set(CLASS_NAMES)
    assert (df["label"].map(CLASS_NAMES.index) == df["label_idx"]).all()


def test_make_dataset_batches(raw_dir):
    cfg = Config()
    cfg.data.batch_size = 4
    df = scan_raw(raw_dir)
    for training in (False, True):
        x, y = next(iter(make_dataset(df, raw_dir, cfg, training=training)))
        assert x.shape == (4, 224, 224, 3)
        assert y.dtype.name == "int32"


def _encode_png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_shortcut_reduced_crops_to_the_head_and_normalizes():
    from neuroscan.data.preprocess import preprocess_bytes

    canvas = np.zeros((300, 300), np.uint8)
    canvas[100:200, 120:180] = 60  # a dim "head" in a large black frame
    canvas[140:160, 140:160] = 90  # a brighter "lesion"
    out = preprocess_bytes(_encode_png(canvas), 64, mode="shortcut_reduced")
    gray = out.mean(axis=-1)
    assert gray[6:-6, 6:-6].min() > 100  # the head fills the frame (small margin aside)
    assert (gray > 100).mean() > 0.7
    assert out.max() == pytest.approx(255, abs=1)  # intensity stretched to the full range
    assert out.shape == (64, 64, 3)


def test_shortcut_reduced_keeps_blank_images_safe():
    from neuroscan.data.preprocess import preprocess_bytes

    out = preprocess_bytes(_encode_png(np.zeros((80, 80), np.uint8)), 32, mode="shortcut_reduced")
    assert np.isfinite(out).all() and out.max() == 0


def test_unknown_preprocessing_mode():
    from neuroscan.data.preprocess import preprocess_bytes

    with pytest.raises(ValueError, match="Unknown preprocessing"):
        preprocess_bytes(_encode_png(np.zeros((80, 80), np.uint8)), 32, mode="fancy")
