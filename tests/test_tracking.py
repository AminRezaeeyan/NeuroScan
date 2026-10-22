"""Import a real (tiny) run bundle into a temporary SQLite MLflow store."""

import numpy as np
import pytest

from neuroscan.config import load_config
from neuroscan.tracking import EXPERIMENT, REGISTERED_MODEL
from neuroscan.tracking.import_run import import_bundle
from neuroscan.train import MONITOR, train


@pytest.fixture
def run_dir(raw_dir, split_csv, tiny_config_file, tmp_path):
    return train(load_config(tiny_config_file), raw_dir, split_csv, tmp_path / "runs")


@pytest.fixture
def store(tmp_path):
    return {
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "artifact_root": tmp_path / "artifacts",
    }


def test_import_creates_complete_run(run_dir, store):
    import mlflow
    from mlflow import MlflowClient

    result = import_bundle(run_dir, **store)
    client = MlflowClient(store["tracking_uri"])
    run = client.get_run(result.run_id)

    assert run.info.status == "FINISHED"
    assert run.info.run_name == "tiny"
    assert client.get_experiment(run.info.experiment_id).name == EXPERIMENT
    assert run.data.params["model.name"] == "cnn_gap"
    assert run.data.params["train.epochs"] == "2"
    assert int(run.data.params["result.params"]) > 0
    assert run.data.tags["model"] == "cnn_gap"
    assert run.data.tags["trained_on"] == "cpu"
    assert len(run.data.tags["split_version"]) == 12
    assert run.data.tags["git_commit"]

    curve = client.get_metric_history(result.run_id, MONITOR)
    assert [m.step for m in curve] == [0, 1]
    assert "best_val_macro_f1" in run.data.metrics

    artifacts = {a.path for a in client.list_artifacts(result.run_id, "figures")}
    assert "figures/learning_curves.png" in artifacts
    bundle_files = {a.path for a in client.list_artifacts(result.run_id, "bundle")}
    assert {"bundle/config.yaml", "bundle/meta.json", "bundle/history.json"} <= bundle_files

    assert [a.path for a in client.list_artifacts(result.run_id, "model_input")] == [
        "model_input/input_example.npy"
    ]
    assert result.model_version == "1"
    version = client.get_model_version(REGISTERED_MODEL, "1")
    assert version.run_id == result.run_id
    model = mlflow.keras.load_model(f"models:/{REGISTERED_MODEL}/1")
    probs = model.predict(np.full((1, 64, 64, 3), 128.0, np.float32), verbose=0)
    assert probs.shape == (1, 4)


def test_reimport_is_refused_unless_forced(run_dir, store):
    from mlflow import MlflowClient

    first = import_bundle(run_dir, register=False, **store)
    with pytest.raises(FileExistsError, match="already imported"):
        import_bundle(run_dir, register=False, **store)
    second = import_bundle(run_dir, register=False, force=True, **store)
    assert (
        MlflowClient(store["tracking_uri"]).get_run(first.run_id).info.lifecycle_stage == "deleted"
    )
    assert second.run_id != first.run_id


def test_unfinished_bundle_is_rejected(run_dir, store):
    (run_dir / "model.keras").unlink()
    with pytest.raises(ValueError, match="not a finished run"):
        import_bundle(run_dir, **store)
