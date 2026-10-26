"""Evaluate packaged models on the external PMRAM test set.

    python -m neuroscan.evaluate_external                    # current and first champion
    python -m neuroscan.evaluate_external --models models:/neuroscan-classifier@champion

Images that duplicate training-dataset images are excluded first. Each model is the
full served package (preprocessing, calibration, OOD detector, uncertainty rule), so
the results show both classification quality and how often the system refuses or
defers on scans from other hospitals. Metrics are logged into each model's run as
``external.pmram.*``.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neuroscan.config import CLASS_NAMES
from neuroscan.data.external import find_overlap, scan
from neuroscan.data.preprocess import preprocess_file
from neuroscan.evaluate import bootstrap_ci, point_metrics

DEFAULT_MODELS = ["models:/neuroscan-classifier@champion", "models:/neuroscan-classifier/7"]
STATUSES = ["ok", "uncertain", "rejected_ood", "rejected_input"]
PREFIX = "external.pmram"


def predict_all(predictor: Any, df: pd.DataFrame, root: Path, batch: int = 32) -> pd.DataFrame:
    mode = predictor.config.get("preprocessing", "standard")
    size = int(predictor.config["image_size"])
    parts = []
    for start in range(0, len(df), batch):
        paths = df["path"].iloc[start : start + batch]
        x = np.stack([preprocess_file(root / p, size, mode) for p in paths])
        parts.append(predictor.predict(None, x))
    out = pd.concat(parts, ignore_index=True)
    return pd.concat([df.reset_index(drop=True), out], axis=1)


def summarize(preds: pd.DataFrame) -> dict[str, float]:
    y_true = preds["label_idx"].to_numpy()
    y_pred = preds["prediction"].map(CLASS_NAMES.index).to_numpy()
    m = {"n_images": float(len(preds))}
    forced = point_metrics(y_true, y_pred)
    m.update({f"forced.{k}": v for k, v in forced.items()})
    m.update({f"forced.{k}": v for k, v in bootstrap_ci(y_true, y_pred).items()})
    for status in STATUSES:
        m[f"status.{status}"] = float((preds["status"] == status).mean())
    ok = (preds["status"] == "ok").to_numpy()
    m["answered.share"] = float(ok.mean())
    if ok.any():
        m["answered.accuracy"] = float((y_true[ok] == y_pred[ok]).mean())
    return m


def table(results: dict[str, dict[str, float]]) -> str:
    lines = [
        "| Model | Images | Accuracy, forced (95% CI) | Macro F1, forced | Tumor sensitivity "
        "| Refused (OOD) | Uncertain | Answered | Accuracy when answered |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name} | {int(m['n_images'])} "
            f"| {m['forced.accuracy']:.1%} ({m['forced.accuracy_ci_low']:.1%}–"
            f"{m['forced.accuracy_ci_high']:.1%}) | {m['forced.macro_f1']:.3f} "
            f"| {m['forced.tumor_sensitivity']:.1%} | {m['status.rejected_ood']:.1%} "
            f"| {m['status.uncertain']:.1%} | {m['answered.share']:.1%} "
            f"| {m.get('answered.accuracy', float('nan')):.1%} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--external-dir", type=Path, default=Path("data/external/pmram"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    import mlflow

    import neuroscan.models  # noqa: F401
    from neuroscan.reports.plots import confusion_matrix_figure
    from neuroscan.tracking import get_client

    client = get_client()
    df = scan(args.external_dir)
    if df.empty:
        raise SystemExit(f"No images in {args.external_dir}; run python -m neuroscan.data.external")
    overlap = find_overlap(df, args.external_dir, args.raw_dir)
    clean = df[~overlap.to_numpy()].reset_index(drop=True)
    print(
        f"PMRAM: {len(df)} images; {int(overlap.sum())} duplicate training-dataset images "
        f"and are excluded; {len(clean)} evaluated"
    )
    print(clean["label"].value_counts().reindex(CLASS_NAMES).to_dict())

    results: dict[str, dict[str, float]] = {}
    for uri in args.models:
        local = mlflow.artifacts.download_artifacts(
            artifact_uri=uri, dst_path=tempfile.mkdtemp(prefix="neuroscan-model-")
        )
        predictor = mlflow.pyfunc.load_model(local).unwrap_python_model()
        name, _, ref = uri.removeprefix("models:/").partition("@")
        mv = (
            client.get_model_version_by_alias(name, ref)
            if ref
            else client.get_model_version(*uri.removeprefix("models:/").split("/"))
        )
        label = f"v{mv.version} ({predictor.config.get('preprocessing', 'standard')})"
        preds = predict_all(predictor, clean, args.external_dir)
        metrics = summarize(preds)
        metrics["overlap_excluded"] = float(overlap.sum())
        results[label] = metrics
        for key, value in metrics.items():
            client.log_metric(mv.run_id, f"{PREFIX}.v{mv.version}.{key}", value)
        y_pred = preds["prediction"].map(CLASS_NAMES.index).to_numpy()
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(preds["label_idx"], y_pred, labels=range(len(CLASS_NAMES)))
        fig = confusion_matrix_figure(cm, list(CLASS_NAMES), f"(PMRAM, model v{mv.version})")
        client.log_figure(mv.run_id, fig, f"external/pmram/v{mv.version}/confusion_matrix.png")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            preds.to_csv(path, index=False)
            client.log_artifact(mv.run_id, str(path), f"external/pmram/v{mv.version}")
        del predictor
        print(f"{label}: done")

    print()
    print(table(results))


if __name__ == "__main__":
    main()
