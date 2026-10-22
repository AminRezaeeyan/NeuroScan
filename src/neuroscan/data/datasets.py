"""Index the raw dataset and build tf.data pipelines from a split file."""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from neuroscan.config import CLASS_NAMES, AugmentConfig, Config
from neuroscan.data.download import SPLITS, list_images
from neuroscan.data.preprocess import decode_and_resize


def scan_raw(raw_dir: str | Path) -> pd.DataFrame:
    """One row per image: path (relative to raw_dir), label, label_idx, kaggle_split."""
    root = Path(raw_dir)
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "label": name,
            "label_idx": idx,
            "kaggle_split": split,
        }
        for split in SPLITS
        for idx, name in enumerate(CLASS_NAMES)
        for path in list_images(root / split / name)
    ]
    if not rows:
        raise FileNotFoundError(f"No images under {root}; run `make data` first")
    return pd.DataFrame(rows)


def load_split(split_csv: str | Path) -> pd.DataFrame:
    """The versioned split file: path, label, group, split (train/val/test)."""
    df = pd.read_csv(split_csv)
    df["label_idx"] = df["label"].map(CLASS_NAMES.index)
    return df


def build_augmentation(cfg: AugmentConfig, seed: int) -> keras.Sequential:
    layers: list[keras.layers.Layer] = []
    if cfg.flip:
        layers.append(keras.layers.RandomFlip("horizontal", seed=seed))
    if cfg.rotation:
        layers.append(keras.layers.RandomRotation(cfg.rotation, fill_mode="nearest", seed=seed))
    if cfg.zoom:
        layers.append(keras.layers.RandomZoom(cfg.zoom, fill_mode="nearest", seed=seed))
    if cfg.brightness:
        # value_range matches our [0, 255] inputs.
        layers.append(
            keras.layers.RandomBrightness(cfg.brightness, value_range=(0, 255), seed=seed)
        )
    return keras.Sequential(layers, name="augmentation")


def make_dataset(
    df: pd.DataFrame,
    raw_dir: str | Path,
    cfg: Config,
    training: bool = False,
) -> tf.data.Dataset:
    """(image float32 [0, 255], label int32) batches. Shuffled and augmented only when training."""
    paths = [str(Path(raw_dir) / p) for p in df["path"]]
    labels = df["label_idx"].to_numpy(dtype=np.int32)
    size = cfg.data.image_size

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=cfg.data.seed, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, y: (decode_and_resize(tf.io.read_file(p), size), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(cfg.data.batch_size)
    if training:
        augment = build_augmentation(cfg.augment, cfg.data.seed)
        ds = ds.map(
            lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE
        )
    return ds.prefetch(tf.data.AUTOTUNE)
