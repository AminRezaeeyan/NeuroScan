"""Evaluation metrics, evaluation sets and an end-to-end run on a tiny model."""

import numpy as np
import pandas as pd
import pytest

from neuroscan.config import CLASS_NAMES, NO_TUMOR_INDEX, load_config
from neuroscan.data.datasets import load_split
from neuroscan.evaluate import (
    bootstrap_ci,
    evaluate,
    point_metrics,
    select_eval_set,
    tumor_rates,
)
from neuroscan.models import build_model

G, M, N, P = range(4)  # glioma, meningioma, notumor, pituitary


def test_tumor_rates_by_hand():
    y_true = np.array([G, M, P, N, N])
    y_pred = np.array([M, N, P, N, G])  # one tumor missed, one false alarm
    sensitivity, specificity = tumor_rates(y_true, y_pred)
    assert sensitivity == pytest.approx(2 / 3)  # confusing tumor types still counts as "tumor"
    assert specificity == pytest.approx(1 / 2)
    assert NO_TUMOR_INDEX == N


def test_point_metrics_are_macro_not_weighted():
    y_true = np.array([G] * 8 + [M, N, P, P])
    y_pred = np.array([G] * 8 + [G, N, P, P])
    m = point_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(11 / 12)
    assert m["meningioma.recall"] == 0.0
    assert m["macro_recall"] == pytest.approx(0.75)  # weighted recall would equal accuracy
    assert m["macro_f1"] < m["accuracy"]


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, 400)
    y_pred = np.where(rng.random(400) < 0.9, y_true, (y_true + 1) % 4)
    ci = bootstrap_ci(y_true, y_pred, n=300)
    acc = (y_true == y_pred).mean()
    assert ci["accuracy_ci_low"] < acc < ci["accuracy_ci_high"]
    assert ci["accuracy_ci_high"] - ci["accuracy_ci_low"] < 0.1


def test_eval_sets():
    split = pd.DataFrame(
        {
            "path": [f"x{i}" for i in range(6)],
            "label": ["glioma"] * 6,
            "group": [0, 0, 1, 2, 3, 3],
            "kaggle_split": ["Training", "Testing", "Testing", "Training", "Testing", "Testing"],
            "split": ["train", "train", "test", "val", "test", "test"],
        }
    )
    assert list(select_eval_set(split, "test")["path"]) == ["x2", "x4", "x5"]
    assert list(select_eval_set(split, "kaggle-testing")["path"]) == ["x1", "x2", "x4", "x5"]
    # x1 shares group 0 with a Training image, so it's not clean.
    assert list(select_eval_set(split, "kaggle-testing-clean")["path"]) == ["x2", "x4", "x5"]
    with pytest.raises(ValueError):
        select_eval_set(split, "nope")


def test_evaluate_end_to_end(raw_dir, split_csv, tiny_config_file, tmp_path):
    model = build_model(load_config(tiny_config_file))
    split = load_split(split_csv)
    split["kaggle_split"] = np.where(split["split"] == "train", "Training", "Testing")
    report = evaluate(
        model, split, raw_dir, "test", batch_size=4, metadata_cache=tmp_path / "meta.csv"
    )
    m = report.metrics
    assert m["n_images"] == 4
    assert 0 <= m["accuracy"] <= 1
    assert {
        "macro_f1_ci_low",
        "tumor_sensitivity",
        "probe_accuracy",
        "latency_cpu_p50_ms",
    } <= m.keys()
    assert set(report.figures) == {"confusion_matrix.png", "misclassified.png"}
    assert list(report.predictions.columns[-4:]) == [f"p_{c}" for c in CLASS_NAMES]
    np.testing.assert_allclose(report.predictions.filter(like="p_").sum(axis=1), 1, rtol=1e-5)
