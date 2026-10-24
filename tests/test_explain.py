"""Grad-CAM and the brain-mask sanity check."""

import keras
import numpy as np
import pytest

from neuroscan.config import load_config
from neuroscan.explain import brain_mask, gradcam, heatmap_stats, target_layer_index
from neuroscan.models import build_model


@pytest.mark.parametrize(
    ("config", "overrides"),
    [
        ("configs/cnn_gap.yaml", {"data.image_size": 64}),
        ("configs/baseline_cnn.yaml", {"data.image_size": 64}),
        ("configs/efficientnet_b0.yaml", {"data.image_size": 64, "model.weights": None}),
        ("configs/resnet50.yaml", {"data.image_size": 64, "model.weights": None}),
    ],
)
def test_gradcam_shapes_and_predicted_class(config, overrides):
    model = build_model(load_config(config, overrides))
    images = np.random.default_rng(0).uniform(0, 255, (3, 64, 64, 3)).astype(np.float32)
    cams, pred = gradcam(model, images)
    assert cams.shape == (3, 64, 64)
    assert cams.min() >= 0 and cams.max() <= 1 + 1e-6
    np.testing.assert_array_equal(pred, model.predict(images, verbose=0).argmax(axis=1))
    target = model.layers[target_layer_index(model)]
    assert isinstance(target, keras.Model | keras.layers.Conv2D)


def disk_image(size=96, radius=30):
    yy, xx = np.mgrid[:size, :size]
    head = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < radius**2
    img = np.where(head, 120.0, 0.0)
    img[size // 2, size // 2] = 0.0  # a dark hole inside (e.g. a ventricle) must stay in the mask
    return np.repeat(img[..., None], 3, axis=-1), head


def test_brain_mask_finds_the_head():
    image, head = disk_image()
    mask = brain_mask(image)
    assert (mask == head).mean() > 0.98
    assert mask[48, 48]


def test_heatmap_stats():
    image, head = disk_image()
    mask = brain_mask(image)
    inside = np.zeros((96, 96))
    inside[48, 50] = 1.0
    outside = np.zeros((96, 96))
    outside[2, 2] = 1.0
    stats = heatmap_stats(inside, mask)
    assert stats["peak_in_brain"] == 1.0 and stats["mass_in_brain"] == 1.0
    assert stats["mask_area"] == pytest.approx(head.mean(), abs=0.02)
    assert heatmap_stats(outside, mask)["peak_in_brain"] == 0.0
