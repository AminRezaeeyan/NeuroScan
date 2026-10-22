"""Import a run bundle into the local MLflow store as a regular run.

    python -m neuroscan.tracking.import_run runs/baseline-cnn

Keeps the original start/end time, logs config as params, per-epoch history as
metrics, figures and bundle files as artifacts, and registers the model as a new
version of ``neuroscan-classifier``. Importing the same bundle twice is refused
unless ``--force`` replaces the earlier import.
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from neuroscan import bundle
from neuroscan.config import flatten
from neuroscan.tracking import EXPERIMENT, REGISTERED_MODEL, get_client

MAX_BATCH = 1000  # MLflow's per-request metric limit


@dataclass
class ImportResult:
    run_id: str
    model_version: str | None


def bundle_id(meta: dict[str, Any]) -> str:
    return f"{meta['run_name']}@{meta['start_ms']}"


def find_existing(client: Any, meta: dict[str, Any]) -> list[Any]:
    exp = client.get_experiment_by_name(EXPERIMENT)
    return client.search_runs(
        [exp.experiment_id], filter_string=f"tags.bundle_id = '{bundle_id(meta)}'"
    )


def epoch_metrics(history: dict[str, list[Any]], meta: dict[str, Any]) -> list[Any]:
    """Per-epoch metrics, timestamps spread evenly between the run's start and end."""
    from mlflow.entities import Metric

    epochs = max((len(v) for v in history.values()), default=0)
    start, end = meta["start_ms"], meta["end_ms"]
    stamps: list[int] = np.linspace(start, end, epochs + 1)[1:].astype(int).tolist()
    metrics = []
    for key, values in history.items():
        if key == "stage":
            continue
        for step, value in enumerate(values):
            metrics.append(Metric(key, float(value), int(stamps[step]), step))
    return metrics


def import_bundle(
    run_dir: Path,
    tracking_uri: str | None = None,
    artifact_root: str | Path | None = None,
    register: bool = True,
    force: bool = False,
) -> ImportResult:
    import keras
    import mlflow
    from mlflow.models import infer_signature

    import neuroscan.models  # noqa: F401 - registers custom layers for loading

    meta = bundle.read_json(run_dir / bundle.META_FILE)
    if "end_ms" not in meta or not (run_dir / bundle.MODEL_FILE).exists():
        raise ValueError(f"{run_dir} is not a finished run bundle")
    config = yaml.safe_load((run_dir / bundle.CONFIG_FILE).read_text())
    history = bundle.read_json(run_dir / bundle.HISTORY_FILE)

    client = get_client(tracking_uri, artifact_root)
    existing = find_existing(client, meta)
    if existing and not force:
        raise FileExistsError(
            f"Bundle {bundle_id(meta)} already imported as run {existing[0].info.run_id}; "
            "pass --force to replace it"
        )
    for old in existing:
        client.delete_run(old.info.run_id)

    gpu = meta.get("gpu", "unknown")
    tags = {
        "bundle_id": bundle_id(meta),
        "git_commit": meta["git_commit"],
        "split_version": meta["split_hash"],
        "split_file": meta.get("split_file", ""),
        "model": meta["model"],
        "trained_on": "cpu" if gpu == "cpu" else f"colab-{gpu}",
        "tensorflow": meta.get("tensorflow", ""),
        "keras": meta.get("keras", ""),
        "dry_run": str(meta["run_name"].endswith("-dryrun")).lower(),
    }
    if "stage" in history:
        tags["stages"] = ",".join(dict.fromkeys(history["stage"]))

    exp = client.get_experiment_by_name(EXPERIMENT)
    run = client.create_run(
        exp.experiment_id, run_name=meta["run_name"], start_time=meta["start_ms"], tags=tags
    )
    rid = run.info.run_id
    try:
        params = {k: "None" if v is None else v for k, v in flatten(config).items()}
        for key, value in params.items():
            client.log_param(rid, key, value)
        for key in ("epochs_trained", "best_epoch", "best_stage", "num_train", "num_val"):
            if key in meta:
                client.log_param(rid, f"result.{key}", meta[key])

        metrics = epoch_metrics(history, meta)
        for i in range(0, len(metrics), MAX_BATCH):
            client.log_batch(rid, metrics=metrics[i : i + MAX_BATCH])
        if "best_val_macro_f1" in meta:
            client.log_metric(rid, "best_val_macro_f1", meta["best_val_macro_f1"])

        figures = run_dir / bundle.FIGURES_DIR
        if figures.is_dir() and any(figures.iterdir()):
            client.log_artifacts(rid, str(figures), "figures")
        for name in (bundle.CONFIG_FILE, bundle.META_FILE, bundle.HISTORY_FILE):
            client.log_artifact(rid, str(run_dir / name), "bundle")

        model = keras.models.load_model(run_dir / bundle.MODEL_FILE)
        size = model.input_shape[1]
        example = np.full((1, size, size, 3), 128.0, dtype=np.float32)
        signature = infer_signature(example, model.predict(example, verbose=0))
        with mlflow.start_run(run_id=rid):
            # The input example is logged separately: mlflow.keras.log_model (3.17) writes
            # it before creating the model folder and fails.
            info = mlflow.keras.log_model(
                model,
                name="model",
                signature=signature,
                registered_model_name=REGISTERED_MODEL if register else None,
            )
            with tempfile.TemporaryDirectory() as tmp:
                np.save(Path(tmp) / "input_example.npy", example)
                mlflow.log_artifact(str(Path(tmp) / "input_example.npy"), "model_input")
        version = str(info.registered_model_version) if register else None
        if version:
            for key, value in (
                ("run_name", meta["run_name"]),
                ("split_version", meta["split_hash"]),
            ):
                client.set_model_version_tag(REGISTERED_MODEL, version, key, value)
    except Exception:
        client.set_terminated(rid, status="FAILED")
        raise
    client.set_terminated(rid, status="FINISHED", end_time=meta["end_ms"])
    return ImportResult(rid, version)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("bundles", type=Path, nargs="+", help="run bundle folder(s)")
    parser.add_argument("--no-register", action="store_true", help="don't register the model")
    parser.add_argument("--force", action="store_true", help="replace an earlier import")
    args = parser.parse_args()
    for run_dir in args.bundles:
        result = import_bundle(run_dir, register=not args.no_register, force=args.force)
        version = f", model version {result.model_version}" if result.model_version else ""
        print(f"{run_dir.name}: run {result.run_id}{version}")


if __name__ == "__main__":
    main()
