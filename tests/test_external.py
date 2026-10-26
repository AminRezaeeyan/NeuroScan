"""External test set: labels from folder names, original images only, overlap detection."""

import numpy as np
import pytest
from PIL import Image

from neuroscan.config import CLASS_NAMES
from neuroscan.data.external import find_overlap, label_from_path, original_images, scan


@pytest.mark.parametrize(
    ("path", "label"),
    [
        ("Raw Data/Raw/512Glioma/g_1.jpg", "glioma"),
        ("Raw Data/Raw/512Meningioma/m_1.jpg", "meningioma"),
        ("Raw Data/Raw/512Pituitary/p_1.jpg", "pituitary"),
        ("Raw Data/Raw/512Normal/n_1.jpg", "notumor"),
        ("readme/cover.jpg", None),
    ],
)
def test_label_from_path(path, label):
    assert label_from_path(path) == label


def test_original_images_skip_augmented(tmp_path):
    for rel in ["Raw/512Glioma/a.jpg", "Augmented Data/Augmented/512Glioma/b.jpg"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (32, 32)).save(tmp_path / rel)
    found = original_images(tmp_path)
    assert list(found["source_path"]) == ["Raw/512Glioma/a.jpg"]


def blobs(seed, size=128):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.zeros((size, size))
    for _ in range(6):
        cy, cx = rng.uniform(0, size, 2)
        img += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / rng.uniform(80, 600))
    return (255 * img / img.max()).astype(np.uint8)


def test_find_overlap(tmp_path):
    raw, ext = tmp_path / "raw", tmp_path / "ext"
    for split in ("Training", "Testing"):
        for i, name in enumerate(CLASS_NAMES):
            (raw / split / name).mkdir(parents=True)
            Image.fromarray(blobs(10 * i + (split == "Testing"))).save(raw / split / name / "a.png")
    for name in CLASS_NAMES:
        (ext / name).mkdir(parents=True)
    Image.fromarray(blobs(0)).save(ext / "glioma" / "copy.png")  # same as Training/glioma/a.png
    Image.fromarray(blobs(999)).save(ext / "glioma" / "new.png")
    df = scan(ext)
    flags = find_overlap(df, ext, raw, raw_cache=None)
    assert dict(zip(df["path"], flags, strict=True)) == {
        "glioma/copy.png": True,
        "glioma/new.png": False,
    }


def test_summary_ignores_absent_classes():
    import pandas as pd

    from neuroscan.evaluate_external import summarize

    preds = pd.DataFrame(
        {
            "label_idx": [0, 0, 1, 2],  # no pituitary images
            "prediction": ["glioma", "glioma", "meningioma", "notumor"],
            "status": ["ok", "ok", "uncertain", "rejected_ood"],
        }
    )
    m = summarize(preds)
    assert m["forced.macro_f1_present"] == 1.0
    assert m["forced.macro_f1"] < 1.0  # the 4-class average counts pituitary as 0
    assert m["answered.share"] == 0.5 and m["answered.accuracy"] == 1.0
    assert m["status.rejected_ood"] == 0.25
