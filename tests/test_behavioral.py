"""Behavioral tests of the registered champion (run with: make test-behavioral).

They need the local MLflow registry and data, so they are skipped elsewhere (CI).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.behavioral

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "src/neuroscan/serving/static/samples"
needs = [ROOT / "mlflow.db", ROOT / "data/raw", ROOT / "splits/v1.csv"]
if not all(p.exists() for p in needs):
    pytest.skip("needs the local MLflow registry and dataset", allow_module_level=True)


@pytest.fixture(scope="module")
def champion():
    import os

    import mlflow

    os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
    from neuroscan.tracking import get_client

    get_client()
    return mlflow.pyfunc.load_model("models:/neuroscan-classifier@champion")


@pytest.fixture(scope="module")
def test_images():
    from neuroscan.data.datasets import load_split
    from neuroscan.data.preprocess import preprocess_file

    split = load_split(ROOT / "splits/v1.csv")
    picked = split[split["split"] == "test"].groupby("label").sample(12, random_state=0)
    images = np.stack([preprocess_file(ROOT / "data/raw" / p) for p in picked["path"]])
    return picked.reset_index(drop=True), images


def test_demo_samples_keep_their_answers(champion):
    from neuroscan.data.preprocess import preprocess_file

    expected = {
        "glioma.jpg": ("ok", "glioma"),
        "meningioma.jpg": ("ok", "meningioma"),
        "pituitary.jpg": ("ok", "pituitary"),
        "no_tumor.jpg": ("ok", "notumor"),
        "hard_case.jpg": ("uncertain", None),
    }
    names = list(expected)
    out = champion.predict(np.stack([preprocess_file(SAMPLES / n) for n in names]))
    for name, (_, row) in zip(names, out.iterrows(), strict=True):
        status, label = expected[name]
        assert row["status"] == status, name
        if label:
            assert row["prediction"] == label, name


@pytest.mark.parametrize("factor", [0.9, 1.1])
def test_stable_under_small_brightness_changes(champion, test_images, factor):
    _, images = test_images
    base = champion.predict(images)
    shifted = champion.predict(np.clip(images * factor, 0, 255).astype(np.float32))
    answered = (base["status"] == "ok").to_numpy()
    agree = (base["prediction"] == shifted["prediction"]).to_numpy()[answered]
    assert agree.mean() >= 0.95


def test_real_test_mris_are_mostly_accepted_and_right(champion, test_images):
    picked, images = test_images
    out = champion.predict(images)
    assert (out["status"] == "rejected_ood").mean() <= 0.08  # ~2% expected; 48 images
    ok = (out["status"] == "ok").to_numpy()
    assert (out["prediction"].to_numpy()[ok] == picked["label"].to_numpy()[ok]).mean() >= 0.95


@pytest.mark.parametrize("source", ["natural_photos", "chest_xray", "spine_mri", "ct"])
def test_out_of_distribution_images_are_refused(champion, source):
    from neuroscan.data.preprocess import preprocess_file

    folder = ROOT / "data/ood" / source
    if not folder.is_dir():
        pytest.skip(f"no {folder} (make ood-data)")
    paths = sorted(folder.iterdir())[:20]
    out = champion.predict(np.stack([preprocess_file(p) for p in paths]))
    refused = out["status"].isin(["rejected_ood", "rejected_input"]).mean()
    assert refused >= (0.85 if source == "ct" else 0.95), pd.Series(out["status"]).value_counts()
