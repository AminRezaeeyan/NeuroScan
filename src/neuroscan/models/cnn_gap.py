"""Baseline CNN with GlobalAveragePooling instead of Flatten (~9M fewer parameters)."""

from __future__ import annotations

import keras

from neuroscan.config import Config
from neuroscan.models.baseline_cnn import conv_blocks
from neuroscan.models.common import (
    FEATURES_LAYER,
    classifier_head,
    compile_model,
    input_layer,
)


def build_model(cfg: Config) -> keras.Model:
    inputs = input_layer(cfg)
    x = keras.layers.Rescaling(1.0 / 255)(inputs)
    x = conv_blocks(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(cfg.model.dense_units, activation="relu", name=FEATURES_LAYER)(x)
    outputs = classifier_head(x, cfg)
    model = keras.Model(inputs, outputs, name="cnn_gap")
    return compile_model(model, cfg.train.learning_rate)
