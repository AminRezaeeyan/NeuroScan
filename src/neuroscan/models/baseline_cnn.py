"""Baseline CNN: 4 conv blocks, Flatten -> Dense 512."""

from __future__ import annotations

import keras

from neuroscan.config import Config
from neuroscan.models.common import (
    FEATURES_LAYER,
    classifier_head,
    compile_model,
    input_layer,
)


def conv_blocks(x: keras.KerasTensor) -> keras.KerasTensor:
    for filters in (32, 64, 128, 128):
        x = keras.layers.Conv2D(filters, 3, activation="relu")(x)
        x = keras.layers.MaxPooling2D(2)(x)
    return x


def build_model(cfg: Config) -> keras.Model:
    inputs = input_layer(cfg)
    x = keras.layers.Rescaling(1.0 / 255)(inputs)
    x = conv_blocks(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(cfg.model.dense_units, activation="relu", name=FEATURES_LAYER)(x)
    outputs = classifier_head(x, cfg)
    model = keras.Model(inputs, outputs, name="baseline_cnn")
    return compile_model(model, cfg.train.learning_rate)
