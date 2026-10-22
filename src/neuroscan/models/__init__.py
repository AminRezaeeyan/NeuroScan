"""Model registry: one builder module per model, picked by ``cfg.model.name``."""

from __future__ import annotations

from collections.abc import Callable

import keras

from neuroscan.config import Config
from neuroscan.models import baseline_cnn, cnn_gap, efficientnet, resnet

BUILDERS: dict[str, Callable[[Config], keras.Model]] = {
    "baseline_cnn": baseline_cnn.build_model,
    "cnn_gap": cnn_gap.build_model,
    "efficientnet_b0": efficientnet.build_model,
    "resnet50": resnet.build_model,
}

TRANSFER_MODELS = {"efficientnet_b0", "resnet50"}


def build_model(cfg: Config) -> keras.Model:
    try:
        builder = BUILDERS[cfg.model.name]
    except KeyError:
        choices = sorted(BUILDERS)
        raise ValueError(f"Unknown model {cfg.model.name!r}; choose from {choices}") from None
    return builder(cfg)
