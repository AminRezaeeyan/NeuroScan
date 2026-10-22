import pytest

from neuroscan.config import flatten, load_config


def test_load_and_override(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("run_name: x\nmodel:\n  name: cnn_gap\ntrain:\n  epochs: 5\n")
    cfg = load_config(path, overrides={"train.epochs": 1, "data.batch_size": 8})
    assert cfg.model.name == "cnn_gap"
    assert cfg.train.epochs == 1
    assert cfg.data.batch_size == 8
    assert cfg.data.image_size == 224  # default kept


def test_round_trip(tmp_path):
    cfg = load_config("configs/resnet50.yaml")
    cfg.save(tmp_path / "out.yaml")
    assert load_config(tmp_path / "out.yaml") == cfg


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("train:\n  epoch: 5\n")
    with pytest.raises(ValueError, match="epoch"):
        load_config(path)


def test_flatten():
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}
