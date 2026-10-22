"""EfficientNetB0, ImageNet-pretrained. Built with the backbone frozen (stage 1)."""

from __future__ import annotations

import keras

from neuroscan.config import Config
from neuroscan.models.common import (
    BACKBONE_LAYER,
    FEATURES_LAYER,
    classifier_head,
    compile_model,
    input_layer,
)


def build_model(cfg: Config) -> keras.Model:
    inputs = input_layer(cfg)
    # Keras EfficientNet rescales internally and expects [0, 255] input.
    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights=cfg.model.weights,
        input_shape=inputs.shape[1:],
        name=BACKBONE_LAYER,
    )
    backbone.trainable = False
    # training=False keeps BatchNorm in inference mode, also after unfreezing.
    x = backbone(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D(name=FEATURES_LAYER)(x)
    outputs = classifier_head(x, cfg)
    model = keras.Model(inputs, outputs, name="efficientnet_b0")
    return compile_model(model, cfg.train.learning_rate)
