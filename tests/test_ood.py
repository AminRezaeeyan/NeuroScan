"""OOD detector, rejection thresholds, input checks and the packaged model's statuses."""

import io

import numpy as np
import pytest
from PIL import Image

from neuroscan.ood import MahalanobisDetector, detection_metrics, uncertain_threshold
from neuroscan.serving.checks import check_array, check_bytes, colored_fraction


def clusters(n=200, d=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 5, (4, d))
    labels = np.repeat(np.arange(4), n)
    return centers[labels] + rng.normal(0, 1, (4 * n, d)), labels, centers


def test_mahalanobis_separates_far_points(tmp_path):
    feats, labels, centers = clusters()
    det = MahalanobisDetector.fit(feats, labels)
    inside = centers + np.random.default_rng(1).normal(0, 1, centers.shape)
    outside = centers.mean(0) + np.random.default_rng(2).normal(0, 1, (50, 16)) + 30
    assert det.score(inside).max() < det.score(outside).min()
    assert detection_metrics(det.score(inside), det.score(outside))["auroc"] == 1.0

    det.threshold = 12.5
    det.save(tmp_path / "d.npz")
    loaded = MahalanobisDetector.load(tmp_path / "d.npz")
    np.testing.assert_allclose(loaded.score(outside), det.score(outside))
    assert loaded.threshold == 12.5


def test_uncertain_threshold_reaches_target():
    conf = np.r_[np.full(90, 0.99), np.full(10, 0.55)]
    correct = np.r_[np.ones(90, bool), np.zeros(10, bool)]
    t = uncertain_threshold(conf, correct, 0.99)
    assert 0.55 < t <= 0.99
    assert correct[conf >= t].mean() >= 0.99


def encode(arr, fmt="PNG"):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


def test_input_checks():
    gray = np.full((128, 128, 3), 90, np.uint8)
    photo = np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    assert colored_fraction(gray) == 0.0 and check_array(gray) is None
    assert check_array(photo) == "not_grayscale"
    assert check_bytes(encode(gray)) is None
    assert check_bytes(encode(photo, "JPEG")) == "not_grayscale"
    assert check_bytes(b"%PDF-1.4 not an image") == "not_an_image"
    assert check_bytes(encode(np.zeros((20, 20, 3), np.uint8))) == "bad_size"


def test_packaged_model_statuses(raw_dir, split_csv, tiny_config_file, tmp_path):
    import keras
    import mlflow

    from neuroscan.config import load_config
    from neuroscan.ood import feature_model
    from neuroscan.serving.packaged import register_packaged_model
    from neuroscan.tracking import REGISTERED_MODEL, get_client
    from neuroscan.tracking.import_run import import_bundle
    from neuroscan.train import train

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_dir = train(load_config(tiny_config_file), raw_dir, split_csv, tmp_path / "runs")
    result = import_bundle(run_dir, tracking_uri=uri, artifact_root=tmp_path / "art")
    client = get_client(uri, tmp_path / "art")

    model = keras.models.load_model(run_dir / "model.keras")
    gray = np.stack([np.full((64, 64, 3), v, np.float32) for v in (40, 80, 120, 160)])
    feats = np.asarray(feature_model(model)(gray)[0])
    det = MahalanobisDetector.fit(np.repeat(feats, 5, 0) + 0.01, np.repeat(np.arange(4), 5))
    det.threshold = float(det.score(feats).max() * 2 + 1)
    det.save(tmp_path / "ood.npz")

    register_packaged_model(
        client,
        result.run_id,
        temperature=1.0,
        ood_path=tmp_path / "ood.npz",
        uncertain_threshold=0.0,
    )
    packaged = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}@champion")
    photo = np.random.default_rng(0).uniform(0, 255, (1, 64, 64, 3)).astype(np.float32)
    out = packaged.predict(np.concatenate([gray, photo]))
    assert list(out["status"][:4]) == ["ok"] * 4
    assert out["status"].iloc[4] == "rejected_input"
    assert np.isfinite(out["ood_score"]).all()

    det.threshold = -1.0  # everything is out of distribution
    det.save(tmp_path / "ood.npz")
    register_packaged_model(client, result.run_id, temperature=1.0, ood_path=tmp_path / "ood.npz")
    packaged = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}@champion")
    assert set(packaged.predict(gray)["status"]) == {"rejected_ood"}


@pytest.mark.parametrize("value", [0.0, 0.7])
def test_uncertain_status(value):
    from neuroscan.serving.packaged import NeuroScanModel

    class Fixed:
        def predict(self, x, verbose=0):
            return np.tile([0.6, 0.2, 0.1, 0.1], (len(x), 1))

    m = NeuroScanModel()
    m.model, m.detector, m.features = Fixed(), None, None
    m.config = {
        "temperature": 1.0,
        "class_names": ["a", "b", "c", "d"],
        "uncertain_threshold": value,
    }
    out = m.predict(None, np.full((2, 8, 8, 3), 100.0, np.float32))
    assert set(out["status"]) == ({"ok"} if value == 0.0 else {"uncertain"})
