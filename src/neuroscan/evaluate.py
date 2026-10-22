"""Evaluate a tracked model and log the results back into its MLflow run.

    python -m neuroscan.evaluate --run-id <id>                 # test split
    python -m neuroscan.evaluate --run-id <id> --eval-set val

Evaluation sets come from the split file:

* ``test`` / ``val``: the duplicate-aware split;
* ``kaggle-testing``: the Kaggle Testing folder (the original benchmark protocol);
* ``kaggle-testing-clean``: Kaggle Testing images with no duplicate in Kaggle Training.

Metrics are logged as ``<eval-set>.<name>``; figures and predictions go under
``evaluation/<eval-set>/``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from neuroscan.config import CLASS_NAMES, NO_TUMOR_INDEX, Config
from neuroscan.data.datasets import load_split, make_dataset
from neuroscan.data.metadata import PROBE_FEATURES, compute_metadata, fit_probe
from neuroscan.reports.plots import confusion_matrix_figure, misclassified_grid

EVAL_SETS = ("test", "val", "kaggle-testing", "kaggle-testing-clean")
BOOTSTRAP_SAMPLES = 1000


@dataclass
class EvalReport:
    eval_set: str
    metrics: dict[str, float]
    predictions: pd.DataFrame
    figures: dict[str, Any] = field(default_factory=dict)


def select_eval_set(split: pd.DataFrame, eval_set: str) -> pd.DataFrame:
    if eval_set in ("test", "val"):
        return split[split["split"] == eval_set]
    testing = split[split["kaggle_split"] == "Testing"]
    if eval_set == "kaggle-testing":
        return testing
    if eval_set == "kaggle-testing-clean":
        training_groups = set(split.loc[split["kaggle_split"] == "Training", "group"])
        return testing[~testing["group"].isin(training_groups)]
    raise ValueError(f"Unknown eval set {eval_set!r}; choose from {EVAL_SETS}")


def tumor_rates(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """(sensitivity, specificity) for tumor (any of the 3 tumor classes) vs. no tumor."""
    true_tumor = y_true != NO_TUMOR_INDEX
    pred_tumor = y_pred != NO_TUMOR_INDEX
    sensitivity = float(pred_tumor[true_tumor].mean()) if true_tumor.any() else float("nan")
    specificity = float((~pred_tumor)[~true_tumor].mean()) if (~true_tumor).any() else float("nan")
    return sensitivity, specificity


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    labels = list(range(len(CLASS_NAMES)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    sensitivity, specificity = tumor_rates(y_true, y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "tumor_sensitivity": sensitivity,
        "tumor_specificity": specificity,
    }
    for i, name in enumerate(CLASS_NAMES):
        metrics[f"{name}.precision"] = float(precision[i])
        metrics[f"{name}.recall"] = float(recall[i])
        metrics[f"{name}.f1"] = float(f1[i])
    return metrics


def bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n: int = BOOTSTRAP_SAMPLES, seed: int = 0
) -> dict[str, float]:
    """95% percentile intervals for accuracy, macro F1 and tumor sensitivity."""
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {"accuracy": [], "macro_f1": [], "tumor_sensitivity": []}
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        t, p = y_true[idx], y_pred[idx]
        samples["accuracy"].append(float((t == p).mean()))
        samples["macro_f1"].append(
            float(f1_score(t, p, average="macro", labels=range(len(CLASS_NAMES)), zero_division=0))
        )
        samples["tumor_sensitivity"].append(tumor_rates(t, p)[0])
    out = {}
    for key, values in samples.items():
        low, high = np.nanpercentile(values, [2.5, 97.5])
        out[f"{key}_ci_low"], out[f"{key}_ci_high"] = float(low), float(high)
    return out


def shortcut_check(
    split: pd.DataFrame,
    preds: pd.DataFrame,
    raw_dir: Path,
    cache: Path | None,
) -> dict[str, float]:
    """Model accuracy where a metadata-only probe is right vs. wrong.

    The probe is trained on the train split's metadata. A model that relies on
    anatomy should do about as well on images the probe gets wrong; a large gap
    means it leans on the same source cues as the probe.
    """
    train_meta = compute_metadata(split[split["split"] == "train"], raw_dir, cache)
    eval_meta = compute_metadata(preds[["path", "label"]], raw_dir, cache)
    probe = fit_probe(train_meta)
    probe_right = probe.predict(eval_meta[PROBE_FEATURES]) == eval_meta["label"].to_numpy()

    y_true, y_pred = preds["label_idx"].to_numpy(), preds["pred_idx"].to_numpy()
    out = {"probe_accuracy": float(probe_right.mean())}
    for name, mask in (("probe_right", probe_right), ("probe_wrong", ~probe_right)):
        out[f"shortcut.{name}.n"] = float(mask.sum())
        if mask.any():
            out[f"shortcut.{name}.accuracy"] = float((y_true[mask] == y_pred[mask]).mean())
            out[f"shortcut.{name}.tumor_sensitivity"] = tumor_rates(y_true[mask], y_pred[mask])[0]
    if "shortcut.probe_wrong.accuracy" in out and "shortcut.probe_right.accuracy" in out:
        out["shortcut.accuracy_gap"] = (
            out["shortcut.probe_right.accuracy"] - out["shortcut.probe_wrong.accuracy"]
        )
    return out


def cpu_latency_ms(model: keras.Model, runs: int = 50, warmup: int = 5) -> dict[str, float]:
    """Single-image inference latency on CPU (p50 and p95)."""
    import tensorflow as tf

    size = model.input_shape[1]
    x = tf.constant(np.full((1, size, size, 3), 128.0, np.float32))
    times = []
    with tf.device("/CPU:0"):
        for i in range(warmup + runs):
            start = time.perf_counter()
            model(x, training=False)
            if i >= warmup:
                times.append((time.perf_counter() - start) * 1000)
    return {
        "latency_cpu_p50_ms": float(np.median(times)),
        "latency_cpu_p95_ms": float(np.percentile(times, 95)),
    }


def evaluate(
    model: keras.Model,
    split: pd.DataFrame,
    raw_dir: Path,
    eval_set: str = "test",
    batch_size: int = 32,
    metadata_cache: Path | None = Path("data/cache/metadata.csv"),
    with_shortcut: bool = True,
    with_latency: bool = True,
) -> EvalReport:
    df = select_eval_set(split, eval_set).reset_index(drop=True)
    cfg = Config()
    cfg.data.image_size = model.input_shape[1]
    cfg.data.batch_size = batch_size
    probs = model.predict(make_dataset(df, raw_dir, cfg, training=False), verbose=0)
    preds = df[["path", "label", "label_idx", "kaggle_split", "split"]].assign(
        pred_idx=probs.argmax(axis=1),
        confidence=probs.max(axis=1),
    )
    preds["pred"] = preds["pred_idx"].map(lambda i: CLASS_NAMES[i])
    for i, name in enumerate(CLASS_NAMES):
        preds[f"p_{name}"] = probs[:, i]

    y_true, y_pred = preds["label_idx"].to_numpy(), preds["pred_idx"].to_numpy()
    metrics = {"n_images": float(len(df)), **point_metrics(y_true, y_pred)}
    metrics.update(bootstrap_ci(y_true, y_pred))
    if with_shortcut:
        metrics.update(shortcut_check(split, preds, raw_dir, metadata_cache))
    if with_latency:
        metrics.update(cpu_latency_ms(model))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    figures = {
        "confusion_matrix.png": confusion_matrix_figure(cm, CLASS_NAMES, title=eval_set),
        "misclassified.png": misclassified_grid(preds[y_true != y_pred], raw_dir),
    }
    return EvalReport(eval_set, metrics, preds, figures)


def log_report(run_id: str, report: EvalReport) -> None:
    import tempfile

    from mlflow import MlflowClient

    client = MlflowClient()
    prefix = f"evaluation/{report.eval_set}"
    for key, value in report.metrics.items():
        if np.isfinite(value):
            client.log_metric(run_id, f"{report.eval_set}.{key}", value)
    for name, fig in report.figures.items():
        client.log_figure(run_id, fig, f"{prefix}/{name}")
    with tempfile.TemporaryDirectory() as tmp:
        report.predictions.to_csv(Path(tmp) / "predictions.csv", index=False)
        (Path(tmp) / "metrics.json").write_text(json.dumps(report.metrics, indent=2))
        client.log_artifacts(run_id, tmp, prefix)


def print_report(report: EvalReport) -> None:
    m = report.metrics
    print(f"\n{report.eval_set}: {int(m['n_images'])} images")
    for key in ("accuracy", "macro_f1", "tumor_sensitivity"):
        print(
            f"  {key:20s} {m[key]:.4f}  (95% CI {m[f'{key}_ci_low']:.4f}-{m[f'{key}_ci_high']:.4f})"
        )
    print(f"  {'tumor_specificity':20s} {m['tumor_specificity']:.4f}")
    for name in CLASS_NAMES:
        p, r, f = (m[f"{name}.{k}"] for k in ("precision", "recall", "f1"))
        print(f"  {name:12s} P={p:.3f} R={r:.3f} F1={f:.3f}")
    if "shortcut.accuracy_gap" in m:
        print(
            f"  shortcut check: probe accuracy {m['probe_accuracy']:.3f}; model accuracy "
            f"{m['shortcut.probe_right.accuracy']:.3f} where probe is right vs "
            f"{m['shortcut.probe_wrong.accuracy']:.3f} where it is wrong "
            f"(gap {m['shortcut.accuracy_gap']:+.3f})"
        )
    if "latency_cpu_p50_ms" in m:
        p50, p95 = m["latency_cpu_p50_ms"], m["latency_cpu_p95_ms"]
        print(f"  CPU latency p50 {p50:.1f} ms, p95 {p95:.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--eval-set", choices=EVAL_SETS, nargs="+", default=["test"])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split-csv", type=Path, default=Path("splits/v1.csv"))
    args = parser.parse_args()

    import mlflow

    import neuroscan.models  # noqa: F401 - registers custom layers
    from neuroscan.tracking import get_client

    get_client()
    model = mlflow.keras.load_model(f"runs:/{args.run_id}/model")
    split = load_split(args.split_csv)
    for eval_set in args.eval_set:
        report = evaluate(model, split, args.raw_dir, eval_set)
        log_report(args.run_id, report)
        print_report(report)


if __name__ == "__main__":
    main()
