"""TFLite export of a tiny model: agreement with Keras and the INT8 size reduction."""

import numpy as np

from neuroscan.config import load_config
from neuroscan.models import build_model
from neuroscan.optimize import TFLiteModel, benchmark, markdown_table, to_tflite


def test_tflite_matches_keras_and_int8_is_smaller():
    model = build_model(load_config("configs/cnn_gap.yaml", {"data.image_size": 64}))
    images = np.random.default_rng(0).uniform(0, 255, (6, 64, 64, 3)).astype(np.float32)
    keras_probs = model.predict(images, verbose=0)

    lite = TFLiteModel(to_tflite(model))
    np.testing.assert_allclose(lite.predict(images), keras_probs, atol=1e-5)

    float_size = len(to_tflite(model))
    int8_size = len(to_tflite(model, quantize=True))
    assert int8_size < float_size / 2


def test_benchmark_rows():
    model = build_model(load_config("configs/cnn_gap.yaml", {"data.image_size": 64}))
    images = np.random.default_rng(1).uniform(0, 255, (8, 64, 64, 3)).astype(np.float32)
    labels = np.array([0, 1, 2, 3] * 2)
    rows, variants = benchmark(model, images, labels, keras_size_mb=1.0)
    assert [r.name for r in rows] == ["keras", "tflite_float32", "tflite_int8"]
    assert rows[1].agreement == 1.0 and rows[1].max_abs_diff < 1e-4
    assert set(variants) == {"tflite_float32", "tflite_int8"}
    assert markdown_table(rows).count("\n") == 4
