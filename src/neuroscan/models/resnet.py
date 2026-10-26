"""ResNet50, ImageNet-pretrained. Built with the backbone frozen (stage 1)."""

from __future__ import annotations

import keras

from neuroscan.config import Config
from neuroscan.models.common import (
    FEATURES_LAYER,
    classifier_head,
    compile_model,
    input_layer,
)

# "caffe" preprocessing used by keras.applications.resnet50: RGB -> BGR, minus ImageNet means.
_IMAGENET_BGR_MEAN = (103.939, 116.779, 123.68)


@keras.saving.register_keras_serializable(package="neuroscan")
class ResNetPreprocess(keras.layers.Layer):
    """``resnet50.preprocess_input`` as a layer, so the saved model takes raw pixels."""

    def call(self, inputs):
        bgr = keras.ops.flip(inputs, axis=-1)
        return bgr - keras.ops.convert_to_tensor(_IMAGENET_BGR_MEAN, dtype=inputs.dtype)


def build_model(cfg: Config) -> keras.Model:
    inputs = input_layer(cfg)
    x = ResNetPreprocess(name="resnet_preprocess")(inputs)
    backbone = keras.applications.ResNet50(
        include_top=False,
        weights=cfg.model.weights,
        input_shape=inputs.shape[1:],
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name=FEATURES_LAYER)(x)
    outputs = classifier_head(x, cfg)
    model = keras.Model(inputs, outputs, name="resnet50")
    return compile_model(model, cfg.train.learning_rate)
