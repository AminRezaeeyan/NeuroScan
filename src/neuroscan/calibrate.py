"""Measure and fix probability calibration with temperature scaling.

    python -m neuroscan.calibrate --run-id <id>             # measure and log
    python -m neuroscan.calibrate --champion --register     # also package and promote

Uses the per-image probabilities the evaluation already logged
(``evaluation/<set>/predictions.csv``). One temperature T is fitted on the
validation split by minimizing negative log-likelihood; dividing the logits by T
never changes the predicted class. With ``--register``, the Keras model and T are
packaged as one MLflow model (``neuroscan.serving.packaged``), registered as a new
version and given the ``champion`` alias.
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from neuroscan.config import CLASS_NAMES
from neuroscan.reports.plots import reliability_diagram

N_BINS = 15
PROB_COLUMNS = [f"p_{name}" for name in CLASS_NAMES]


def probs_to_logits(probs: np.ndarray) -> np.ndarray:
    """Log-probabilities: equal to the logits up to a per-row constant, which softmax ignores."""
    return np.log(np.clip(probs, 1e-12, 1.0))


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(probs_to_logits(probs) / temperature)


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0))))


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    """The T in [0.05, 20] that minimizes NLL of softmax(logits / T)."""
    logits = probs_to_logits(probs)
    result = minimize_scalar(
        lambda t: nll(softmax(logits / t), y), bounds=(0.05, 20.0), method="bounded"
    )
    return float(result.x)


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    """Top-label ECE: |accuracy - confidence| per confidence bin, weighted by bin size."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


@dataclass
class CalibrationReport:
    temperature: float
    metrics: dict[str, float]
    probs: dict[str, np.ndarray]  # eval set -> uncalibrated probabilities
    labels: dict[str, np.ndarray]


def calibrate(predictions: dict[str, pd.DataFrame]) -> CalibrationReport:
    """``predictions`` maps eval set ("val", "test", ...) to evaluation predictions.

    The temperature is fitted on "val" only.
    """
    probs = {k: df[PROB_COLUMNS].to_numpy(dtype=np.float64) for k, df in predictions.items()}
    labels = {k: df["label_idx"].to_numpy() for k, df in predictions.items()}
    temperature = fit_temperature(probs["val"], labels["val"])
    metrics = {"temperature": temperature}
    for name in predictions:
        p, y = probs[name], labels[name]
        q = apply_temperature(p, temperature)
        metrics[f"{name}.ece_before"] = expected_calibration_error(p, y)
        metrics[f"{name}.ece_after"] = expected_calibration_error(q, y)
        metrics[f"{name}.nll_before"] = nll(p, y)
        metrics[f"{name}.nll_after"] = nll(q, y)
        metrics[f"{name}.mean_confidence_before"] = float(p.max(axis=1).mean())
        metrics[f"{name}.mean_confidence_after"] = float(q.max(axis=1).mean())
    return CalibrationReport(temperature, metrics, probs, labels)


def load_predictions(client: Any, run_id: str, eval_sets: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in eval_sets:
            path = client.download_artifacts(run_id, f"evaluation/{name}/predictions.csv", tmp)
            out[name] = pd.read_csv(path)
    return out


def log_report(client: Any, run_id: str, report: CalibrationReport) -> None:
    for key, value in report.metrics.items():
        client.log_metric(run_id, f"calibration.{key}", value)
    for name in report.probs:
        p, y = report.probs[name], report.labels[name]
        fig = reliability_diagram(
            {
                "before": p,
                f"after (T={report.temperature:.2f})": apply_temperature(p, report.temperature),
            },
            y,
            title=f"Reliability diagram ({name})",
        )
        client.log_figure(run_id, fig, f"calibration/reliability_{name}.png")


def print_report(report: CalibrationReport) -> None:
    m = report.metrics
    print(f"Temperature (fitted on val): {report.temperature:.3f}")
    for name in report.probs:
        print(
            f"  {name:5s} ECE {m[f'{name}.ece_before']:.4f} -> {m[f'{name}.ece_after']:.4f}"
            f"   NLL {m[f'{name}.nll_before']:.4f} -> {m[f'{name}.nll_after']:.4f}"
            f"   mean confidence {m[f'{name}.mean_confidence_before']:.3f}"
            f" -> {m[f'{name}.mean_confidence_after']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-id")
    target.add_argument("--champion", action="store_true", help="the run behind @champion")
    parser.add_argument("--eval-set", nargs="+", default=["val", "test"])
    parser.add_argument(
        "--register", action="store_true", help="package model + temperature and promote it"
    )
    args = parser.parse_args()
    if "val" not in args.eval_set:
        parser.error("--eval-set must include val (the temperature is fitted on it)")

    from neuroscan.tracking import REGISTERED_MODEL, get_client

    client = get_client()
    run_id = args.run_id
    if args.champion:
        run_id = client.get_model_version_by_alias(REGISTERED_MODEL, "champion").run_id
    report = calibrate(load_predictions(client, run_id, args.eval_set))
    log_report(client, run_id, report)
    print_report(report)

    if args.register:
        from neuroscan.serving.packaged import register_packaged_model

        version = register_packaged_model(client, run_id, temperature=report.temperature)
        print(f"Registered calibrated model as version {version}; alias 'champion' moved to it")


if __name__ == "__main__":
    main()
