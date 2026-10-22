"""End-to-end training on a tiny synthetic dataset: bundle layout and contents."""

import json

import keras
import pytest
import yaml

from neuroscan import bundle
from neuroscan.config import load_config
from neuroscan.train import MONITOR, dry_run_overrides, parse_set, train


def test_train_writes_complete_bundle(raw_dir, split_csv, tiny_config_file, tmp_path):
    cfg = load_config(tiny_config_file)
    run_dir = train(cfg, raw_dir, split_csv, tmp_path / "runs")

    for name in (bundle.CONFIG_FILE, bundle.META_FILE, bundle.HISTORY_FILE, bundle.MODEL_FILE):
        assert (run_dir / name).is_file(), name
    assert (run_dir / bundle.FIGURES_DIR / "learning_curves.png").is_file()
    assert not (run_dir / bundle.CHECKPOINT_DIR).exists()

    meta = json.loads((run_dir / bundle.META_FILE).read_text())
    assert meta["model"] == "cnn_gap"
    assert meta["split_hash"] == bundle.file_hash(split_csv)
    assert meta["end_ms"] >= meta["start_ms"]
    assert meta["epochs_trained"] == 2
    assert meta["num_train"] == 8 and meta["num_val"] == 4
    assert meta["params"] == keras.models.load_model(run_dir / bundle.MODEL_FILE).count_params()

    history = json.loads((run_dir / bundle.HISTORY_FILE).read_text())
    assert len(history[MONITOR]) == 2
    assert {"loss", "val_loss", "accuracy", "learning_rate", "stage"} <= history.keys()

    model = keras.models.load_model(run_dir / bundle.MODEL_FILE)
    assert model.output_shape == (None, 4)
    assert yaml.safe_load((run_dir / bundle.CONFIG_FILE).read_text())["run_name"] == "tiny"


def test_finished_run_is_not_overwritten(raw_dir, split_csv, tiny_config_file, tmp_path):
    cfg = load_config(tiny_config_file, {"train.epochs": 1})
    train(cfg, raw_dir, split_csv, tmp_path / "runs")
    with pytest.raises(FileExistsError):
        train(cfg, raw_dir, split_csv, tmp_path / "runs")
    train(cfg, raw_dir, split_csv, tmp_path / "runs", overwrite=True)


@pytest.mark.slow
def test_two_stage_transfer_model(raw_dir, split_csv, tmp_path):
    cfg = load_config(
        "configs/efficientnet_b0.yaml",
        {
            "model.weights": None,
            "data.image_size": 64,
            "data.batch_size": 4,
            "train.epochs": 2,
            "train.head_epochs": 1,
        },
    )
    run_dir = train(cfg, raw_dir, split_csv, tmp_path / "runs")
    history = json.loads((run_dir / bundle.HISTORY_FILE).read_text())
    assert history["stage"] == ["head", "finetune"]


def test_dry_run_overrides():
    assert (
        dry_run_overrides(__import__("pathlib").Path("configs/baseline_cnn.yaml"))["train.epochs"]
        == 1
    )
    two = dry_run_overrides(__import__("pathlib").Path("configs/resnet50.yaml"))
    assert two["train.epochs"] == 2 and two["train.head_epochs"] == 1


def test_parse_set():
    assert parse_set(["train.epochs=3", "model.weights=null"]) == {
        "train.epochs": 3,
        "model.weights": None,
    }


def test_interrupted_run_resumes(raw_dir, split_csv, tiny_config_file, tmp_path, monkeypatch):
    from neuroscan import train as train_module

    cfg = load_config(tiny_config_file, {"train.epochs": 3})
    original = train_module.MacroF1.on_epoch_end

    def crash_on_second_epoch(self, epoch, logs=None):
        if epoch == 1:
            raise RuntimeError("simulated disconnect")
        original(self, epoch, logs)

    monkeypatch.setattr(train_module.MacroF1, "on_epoch_end", crash_on_second_epoch)
    with pytest.raises(RuntimeError, match="simulated disconnect"):
        train(cfg, raw_dir, split_csv, tmp_path / "runs")
    start_ms = json.loads((tmp_path / "runs/tiny" / bundle.META_FILE).read_text())["start_ms"]

    monkeypatch.setattr(train_module.MacroF1, "on_epoch_end", original)
    run_dir = train(cfg, raw_dir, split_csv, tmp_path / "runs")
    meta = json.loads((run_dir / bundle.META_FILE).read_text())
    history = json.loads((run_dir / bundle.HISTORY_FILE).read_text())
    assert meta["start_ms"] == start_ms  # same run, not a new one
    assert len(history[MONITOR]) == 3
