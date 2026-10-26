"""Local MLflow tracking: one experiment, one registered model.

The tracking store is ``sqlite:///mlflow.db`` with artifacts in ``./mlartifacts``
(override with ``MLFLOW_TRACKING_URI`` / ``NEUROSCAN_ARTIFACT_ROOT``). Browse with
``make mlflow``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

EXPERIMENT = "neuroscan"
REGISTERED_MODEL = "neuroscan-classifier"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_ARTIFACT_ROOT = "mlartifacts"

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def run_preprocessing(client: Any, run_id: str) -> str:
    """The preprocessing mode a run was trained with ("standard" for older runs)."""
    return str(client.get_run(run_id).data.params.get("data.preprocessing", "standard"))


def get_client(tracking_uri: str | None = None, artifact_root: str | Path | None = None) -> Any:
    """Point MLflow at the local store and make sure the experiment exists."""
    import mlflow
    from mlflow import MlflowClient

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    root = Path(artifact_root or os.environ.get("NEUROSCAN_ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    if client.get_experiment_by_name(EXPERIMENT) is None:
        client.create_experiment(EXPERIMENT, artifact_location=root.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return client
