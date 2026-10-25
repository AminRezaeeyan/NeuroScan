"""Train/serve parity: the training pipeline and the API see identical tensors."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from neuroscan.config import Config
from neuroscan.data.datasets import make_dataset
from neuroscan.data.preprocess import preprocess_bytes

CASES = [
    ("gray.png", "L", (300, 260), "PNG"),
    ("rgb.jpg", "RGB", (512, 512), "JPEG"),
    ("rgba.png", "RGBA", (150, 400), "PNG"),
    ("odd.bmp", "RGB", (97, 223), "BMP"),
]


@pytest.fixture
def files(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for name, mode, size, fmt in CASES:
        channels = {"L": 1, "RGB": 3, "RGBA": 4}[mode]
        arr = rng.integers(0, 256, (size[1], size[0], channels), dtype=np.uint8)
        Image.fromarray(arr.squeeze(), mode=mode).save(tmp_path / name, format=fmt)
        rows.append({"path": name, "label": "glioma", "label_idx": 0})
    return tmp_path, pd.DataFrame(rows)


@pytest.mark.parametrize("mode", ["standard", "shortcut_reduced"])
def test_training_pipeline_and_api_produce_identical_tensors(files, mode):
    """The API preprocesses uploads with preprocess_bytes; training uses make_dataset."""
    root, df = files
    cfg = Config()
    cfg.data.batch_size = len(df)
    cfg.data.preprocessing = mode
    batch, _ = next(iter(make_dataset(df, root, cfg, training=False)))
    for i, name in enumerate(df["path"]):
        served = preprocess_bytes((root / name).read_bytes(), cfg.data.image_size, mode)
        np.testing.assert_array_equal(batch[i].numpy(), served, err_msg=name)
