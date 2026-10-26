"""Shared fixtures: a tiny synthetic dataset laid out like the Kaggle one, plus its split."""

import pandas as pd
import pytest
from PIL import Image

from neuroscan.config import CLASS_NAMES


@pytest.fixture
def raw_dir(tmp_path):
    root = tmp_path / "raw"
    for split in ("Training", "Testing"):
        for i, name in enumerate(CLASS_NAMES):
            folder = root / split / name
            folder.mkdir(parents=True)
            for j in range(2):
                Image.new("L", (50, 60), color=i * 40 + j).save(folder / f"{j}.jpg")
    return root


@pytest.fixture
def split_csv(raw_dir, tmp_path):
    """Training images -> train, Testing images -> val/test alternately."""
    from neuroscan.data.datasets import scan_raw

    df = scan_raw(raw_dir)
    df["group"] = range(len(df))
    df["split"] = "train"
    test_rows = df.index[df["kaggle_split"] == "Testing"]
    df.loc[test_rows, "split"] = ["val", "test"] * (len(test_rows) // 2)
    path = tmp_path / "split.csv"
    df[["path", "label", "group", "split"]].to_csv(path, index=False)
    return path


@pytest.fixture
def tiny_config_file(tmp_path):
    path = tmp_path / "tiny.yaml"
    path.write_text(
        "run_name: tiny\n"
        "model: {name: cnn_gap, dense_units: 8}\n"
        "data: {image_size: 64, batch_size: 4}\n"
        "train: {epochs: 2}\n"
    )
    return path


def split_frame(path):
    return pd.read_csv(path)
