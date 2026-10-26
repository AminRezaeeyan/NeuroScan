"""Model-builder smoke tests: build only, no training."""

import keras
import numpy as np
import pytest

from neuroscan.config import CLASS_NAMES, Config, load_config
from neuroscan.models import BUILDERS, build_model
from neuroscan.models.common import FEATURES_LAYER, get_backbone, set_backbone_trainable

CONFIGS = {
    "baseline_cnn": "configs/baseline_cnn.yaml",
    "cnn_gap": "configs/cnn_gap.yaml",
    "efficientnet_b0": "configs/efficientnet_b0.yaml",
    "resnet50": "configs/resnet50.yaml",
}


def _config(name: str) -> Config:
    # No ImageNet download in tests.
    return load_config(CONFIGS[name], overrides={"model.weights": None})


def test_every_builder_has_a_config():
    assert set(CONFIGS) == set(BUILDERS)


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_builder_shapes(name):
    model = build_model(_config(name))
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, len(CLASS_NAMES))
    assert model.optimizer is not None
    model.get_layer(FEATURES_LAYER)

    probs = model.predict(np.full((2, 224, 224, 3), 128.0, np.float32), verbose=0)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


def test_gap_variant_is_much_smaller():
    baseline = build_model(_config("baseline_cnn")).count_params()
    gap = build_model(_config("cnn_gap")).count_params()
    assert baseline > 9_000_000
    assert gap < 500_000


@pytest.mark.parametrize("name", ["efficientnet_b0", "resnet50"])
def test_transfer_models_start_frozen_and_unfreeze_top(name):
    cfg = _config(name)
    model = build_model(cfg)
    backbone = get_backbone(model)
    assert not backbone.trainable

    set_backbone_trainable(model, cfg.model.fine_tune_from)
    trainable = [layer for layer in backbone.layers if layer.trainable]
    assert trainable
    assert all(layer.name.startswith(("block6", "block7", "top", "conv5")) for layer in trainable)
    assert not any(isinstance(layer, keras.layers.BatchNormalization) for layer in trainable)

    with pytest.raises(ValueError, match="No backbone layer"):
        set_backbone_trainable(model, "no_such_block")


def test_unknown_model_name():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model(load_config(CONFIGS["baseline_cnn"], overrides={"model.name": "vgg"}))


@pytest.mark.parametrize(
    ("name", "expected_file"),
    [
        ("efficientnet_b0", "efficientnetb0_notop.h5"),
        ("resnet50", "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"),
    ],
)
def test_imagenet_weights_file_name(name, expected_file, monkeypatch):
    """The builders must request Keras' real ImageNet weights files (no network needed)."""
    from keras.src.utils import file_utils

    requested = []

    def fake_get_file(fname=None, *args, **kwargs):
        requested.append(fname)
        raise RuntimeError("stop before downloading")

    monkeypatch.setattr(file_utils, "get_file", fake_get_file)
    with pytest.raises(RuntimeError, match="stop before downloading"):
        build_model(load_config(CONFIGS[name]))
    assert requested == [expected_file]
