"""Duplicate detection and split construction on synthetic 'scans'."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from neuroscan.config import CLASS_NAMES
from neuroscan.data.datasets import scan_raw
from neuroscan.data.dedup import compute_signatures, find_groups
from neuroscan.data.splits import assign_splits, build_split, check_split


def blobs(seed: int, size: int = 128) -> np.ndarray:
    """A smooth random image: a few Gaussian blobs, distinct for each seed."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.zeros((size, size))
    for _ in range(6):
        cy, cx = rng.uniform(0, size, 2)
        img += rng.uniform(0.3, 1) * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / rng.uniform(80, 600)
        )
    return (255 * img / img.max()).astype(np.uint8)


@pytest.fixture
def dup_raw(tmp_path):
    """Kaggle-like layout with planted duplicates across Training and Testing."""
    root = tmp_path / "raw"
    seed = 0
    for split in ("Training", "Testing"):
        for name in CLASS_NAMES:
            folder = root / split / name
            folder.mkdir(parents=True)
            for j in range(10):
                seed += 1
                Image.fromarray(blobs(seed)).save(folder / f"{j}.png")
    base = blobs(1)
    test = root / "Testing" / "glioma"
    Image.fromarray(base).save(test / "exact.png")
    Image.fromarray((255 * (base / 255) ** 0.5).astype(np.uint8)).save(test / "gamma.png")
    Image.fromarray(base).save(test / "resaved.jpg", quality=70)
    Image.fromarray(np.flipud(base)).save(root / "Training" / "glioma" / "Tr-aug-gl_1.png")
    return root


def test_planted_duplicates_are_grouped(dup_raw):
    df, vectors = compute_signatures(scan_raw(dup_raw), dup_raw)
    groups, pairs = find_groups(df, vectors)
    original = groups.loc[groups["path"] == "Training/glioma/0.png", "group"].item()
    planted = groups[groups["path"].str.contains("exact|gamma|resaved")]
    assert (planted["group"] == original).all()
    assert pairs["reason"].str.contains("exact").any()

    others = groups[~groups["path"].str.contains("exact|gamma|resaved|aug|Training/glioma/0.png")]
    assert (others["group_size"] == 1).all(), "distinct images must not be grouped"


def test_signature_cache_round_trip(dup_raw, tmp_path):
    df = scan_raw(dup_raw)
    cache = tmp_path / "sig.npz"
    first, v1 = compute_signatures(df, dup_raw, cache)
    second, v2 = compute_signatures(df, dup_raw, cache)
    np.testing.assert_array_equal(v1, v2)
    assert (first["md5"] == second["md5"]).all()
    assert (first["phash"] == second["phash"]).all()


def test_split_keeps_groups_together_and_drops_augmented(dup_raw):
    split = build_split(dup_raw)
    assert not split["path"].str.contains("-aug-").any()
    assert split.groupby("group")["split"].nunique().max() == 1
    assert set(split["split"]) == {"train", "val", "test"}


def test_check_split_rejects_crossing_group():
    df = pd.DataFrame(
        {
            "label": list(CLASS_NAMES) * 3,
            "group": [0] + list(range(1, 12)),
            "split": ["train"] * 4 + ["val"] * 4 + ["test"] * 4,
        }
    )
    df.loc[5, "group"] = 0  # same group as row 0, but in val
    with pytest.raises(ValueError, match="cross splits"):
        check_split(df)


def test_assign_splits_proportions():
    n = 2000
    df = pd.DataFrame(
        {"label": np.array(CLASS_NAMES)[np.arange(n) % 4], "group": np.arange(n) // 2}
    )
    share = assign_splits(df)["split"].value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.70, abs=0.02)
    assert share["val"] == pytest.approx(0.15, abs=0.02)
    assert share["test"] == pytest.approx(0.15, abs=0.02)


def test_committed_split_is_clean():
    split = pd.read_csv("splits/v1.csv")
    check_split(split)
    assert not split["path"].str.contains("-aug-").any()
    assert set(split["label"]) == set(CLASS_NAMES)
