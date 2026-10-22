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


@pytest.fixture
def raw_dir(tmp_path):
    for split in ("Training", "Testing"):
        for i, name in enumerate(CLASS_NAMES):
            folder = tmp_path / split / name
            folder.mkdir(parents=True)
            for j in range(2):
                Image.new("L", (50, 60), color=i * 40 + j).save(folder / f"{j}.jpg")
    return tmp_path


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
