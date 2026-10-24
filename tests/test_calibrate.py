"""Temperature scaling, ECE and the packaged (calibrated) MLflow model."""

import numpy as np
import pandas as pd
import pytest

from neuroscan.calibrate import (
    PROB_COLUMNS,
    apply_temperature,
    calibrate,
    expected_calibration_error,
    fit_temperature,
    softmax,
)


def overconfident(n=4000, factor=2.5, seed=0):
    """Labels drawn from softmax(z); the 'model' reports softmax(factor * z)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, (n, 4))
    true = softmax(z)
    y = np.array([rng.choice(4, p=p) for p in true])
    return softmax(factor * z), y


def test_temperature_recovers_overconfidence():
    probs, y = overconfident(factor=2.5)
    assert fit_temperature(probs, y) == pytest.approx(2.5, rel=0.1)


def test_temperature_keeps_predictions_and_lowers_ece():
    probs, y = overconfident()
    t = fit_temperature(probs, y)
    fixed = apply_temperature(probs, t)
    np.testing.assert_array_equal(fixed.argmax(axis=1), probs.argmax(axis=1))
    np.testing.assert_allclose(fixed.sum(axis=1), 1.0)
    assert expected_calibration_error(fixed, y) < expected_calibration_error(probs, y) / 2


def test_ece_by_hand():
    # Two images at confidence 0.9: one right, one wrong -> |0.5 - 0.9| = 0.4.
    probs = np.array([[0.9, 0.1, 0, 0], [0.9, 0.1, 0, 0]])
    assert expected_calibration_error(probs, np.array([0, 1])) == pytest.approx(0.4)


def test_calibrate_fits_on_validation_only():
    val_p, val_y = overconfident(seed=1)
    test_p, test_y = overconfident(seed=2, factor=1.0)  # already calibrated
    frame = lambda p, y: pd.DataFrame(p, columns=PROB_COLUMNS).assign(label_idx=y)  # noqa: E731
    report = calibrate({"val": frame(val_p, val_y), "test": frame(test_p, test_y)})
    assert report.temperature == pytest.approx(2.5, rel=0.1)
    assert report.metrics["test.ece_after"] > report.metrics["test.ece_before"]  # T from val


def test_packaged_model_round_trip(raw_dir, split_csv, tiny_config_file, tmp_path):
    import mlflow

    from neuroscan.config import CLASS_NAMES, load_config
    from neuroscan.serving.packaged import register_packaged_model
    from neuroscan.tracking import REGISTERED_MODEL, get_client
    from neuroscan.tracking.import_run import import_bundle
    from neuroscan.train import train

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_dir = train(load_config(tiny_config_file), raw_dir, split_csv, tmp_path / "runs")
    result = import_bundle(run_dir, tracking_uri=uri, artifact_root=tmp_path / "art")
    client = get_client(uri, tmp_path / "art")
    end_time = client.get_run(result.run_id).info.end_time

    version = register_packaged_model(client, result.run_id, temperature=2.0)
    assert version == "2"
    assert str(client.get_model_version_by_alias(REGISTERED_MODEL, "champion").version) == "2"
    assert client.get_model_version(REGISTERED_MODEL, "2").tags["source_version"] == "1"
    assert client.get_run(result.run_id).info.end_time == end_time

    packaged = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}@champion")
    x = np.random.default_rng(0).uniform(0, 255, (3, 64, 64, 3)).astype(np.float32)
    out = packaged.predict(x)
    assert list(out.columns) == [*CLASS_NAMES, "prediction", "confidence", "ood_score", "status"]
    np.testing.assert_allclose(out[list(CLASS_NAMES)].sum(axis=1), 1.0, rtol=1e-5)

    raw = mlflow.keras.load_model(f"runs:/{result.run_id}/model").predict(x, verbose=0)
    np.testing.assert_array_equal(out[list(CLASS_NAMES)].to_numpy().argmax(1), raw.argmax(1))
    assert (out["confidence"].to_numpy() <= raw.max(axis=1) + 1e-6).all()  # T > 1 softens
