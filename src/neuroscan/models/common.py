"""Helpers shared by the model builders."""

from __future__ import annotations

import keras

from neuroscan.config import CLASS_NAMES, Config

FEATURES_LAYER = "features"
BACKBONE_LAYER = "backbone"


def input_layer(cfg: Config) -> keras.KerasTensor:
    """All models take raw RGB pixels in [0, 255]; each does its own scaling inside."""
    size = cfg.data.image_size
    return keras.Input(shape=(size, size, 3), name="image")


def classifier_head(x: keras.KerasTensor, cfg: Config) -> keras.KerasTensor:
    x = keras.layers.Dropout(cfg.model.dropout)(x)
    return keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="probs")(x)


def compile_model(model: keras.Model, learning_rate: float) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def set_backbone_trainable(model: keras.Model, fine_tune_from: str) -> None:
    """Unfreeze the backbone from the first layer named ``fine_tune_from*`` ("" = all of it).

    BatchNorm layers stay frozen: updating their statistics with small batches
    hurts fine-tuning (standard Keras transfer-learning advice).
    """
    backbone = model.get_layer(BACKBONE_LAYER)
    backbone.trainable = True
    layers = backbone.layers
    cutoff = 0
    if fine_tune_from:
        matches = [i for i, layer in enumerate(layers) if layer.name.startswith(fine_tune_from)]
        if not matches:
            raise ValueError(f"No backbone layer starts with {fine_tune_from!r}")
        cutoff = matches[0]
    for i, layer in enumerate(layers):
        is_bn = isinstance(layer, keras.layers.BatchNormalization)
        layer.trainable = i >= cutoff and not is_bn
