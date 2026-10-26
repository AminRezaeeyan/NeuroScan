"""Export the champion to TensorFlow Lite and benchmark it against Keras.

    python -m neuroscan.optimize --champion

Builds two TFLite models from the champion's Keras network: float32, and INT8
weights (dynamic-range quantization: weights stored as int8, activations computed
in float, no calibration data needed). For each of Keras, TFLite float32 and TFLite
INT8 it measures single-image CPU latency (p50/p95), model size, agreement with
Keras and test-split accuracy / macro F1, and logs the table into the run.

TFLite covers classification only. The served model keeps Keras because Grad-CAM
and the OOD detector need gradients and intermediate features.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score

AGREEMENT_SAMPLES = 64
Predictor = Callable[[np.ndarray], np.ndarray]


def to_tflite(model: keras.Model, quantize: bool = False) -> bytes:
    """Convert via a SavedModel export (the Keras 3 path TFLite supports)."""
    with tempfile.TemporaryDirectory() as tmp:
        model.export(tmp, format="tf_saved_model", verbose=False)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmp)
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        return converter.convert()


class TFLiteModel:
    """Single-image TFLite inference: (H, W, 3) or (1, H, W, 3) float32 -> probabilities."""

    def __init__(self, content: bytes, threads: int | None = None) -> None:
        self.interpreter = tf.lite.Interpreter(model_content=content, num_threads=threads)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]["index"]
        self.output = self.interpreter.get_output_details()[0]["index"]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        x = np.asarray(image, dtype=np.float32).reshape(1, *image.shape[-3:])
        self.interpreter.set_tensor(self.input, x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output)[0]

    def predict(self, images: np.ndarray) -> np.ndarray:
        return np.stack([self(img) for img in images])


def keras_predictor(model: keras.Model) -> Predictor:
    infer = tf.function(lambda x: model(x, training=False))

    def run(image: np.ndarray) -> np.ndarray:
        return infer(tf.constant(image[None], tf.float32)).numpy()[0]

    return run


def latency_ms(
    fn: Predictor, image: np.ndarray, runs: int = 50, warmup: int = 5
) -> tuple[float, float]:
    times = []
    for i in range(warmup + runs):
        start = time.perf_counter()
        fn(image)
        if i >= warmup:
            times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


@dataclass
class Row:
    name: str
    size_mb: float
    p50_ms: float
    p95_ms: float
    max_abs_diff: float
    agreement: float
    accuracy: float
    macro_f1: float


def benchmark(
    model: keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    keras_size_mb: float,
    keras_probs: np.ndarray | None = None,
) -> tuple[list[Row], dict[str, bytes]]:
    """Compare Keras, TFLite float32 and TFLite INT8 on ``images``."""
    run_keras = keras_predictor(model)
    if keras_probs is None:
        keras_probs = np.stack([run_keras(img) for img in images])
    variants = {"tflite_float32": to_tflite(model), "tflite_int8": to_tflite(model, quantize=True)}

    rows = []
    k_pred = keras_probs.argmax(axis=1)
    p50, p95 = latency_ms(run_keras, images[0])
    rows.append(
        Row(
            "keras",
            keras_size_mb,
            p50,
            p95,
            0.0,
            1.0,
            float((k_pred == labels).mean()),
            float(f1_score(labels, k_pred, average="macro")),
        )
    )
    n = min(AGREEMENT_SAMPLES, len(images))
    for name, content in variants.items():
        lite = TFLiteModel(content)
        probs = lite.predict(images)
        pred = probs.argmax(axis=1)
        p50, p95 = latency_ms(lite, images[0])
        rows.append(
            Row(
                name,
                len(content) / 1e6,
                p50,
                p95,
                float(np.abs(probs[:n] - keras_probs[:n]).max()),
                float((pred == k_pred).mean()),
                float((pred == labels).mean()),
                float(f1_score(labels, pred, average="macro")),
            )
        )
    return rows, variants


def markdown_table(rows: list[Row]) -> str:
    lines = [
        "| Runtime | Size | CPU p50 | CPU p95 | Max prob. diff vs Keras | Same class as Keras "
        "| Test accuracy | Test macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r.name} | {r.size_mb:.1f} MB | {r.p50_ms:.1f} ms | {r.p95_ms:.1f} ms "
            f"| {r.max_abs_diff:.1e} | {r.agreement:.1%} | {r.accuracy:.1%} | {r.macro_f1:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-id")
    target.add_argument("--champion", action="store_true", help="the run behind @champion")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split-csv", type=Path, default=Path("splits/v1.csv"))
    args = parser.parse_args()

    import mlflow

    import neuroscan.models  # noqa: F401
    from neuroscan.calibrate import PROB_COLUMNS
    from neuroscan.data.datasets import load_split
    from neuroscan.data.preprocess import preprocess_file
    from neuroscan.tracking import REGISTERED_MODEL, get_client, run_preprocessing

    client = get_client()
    run_id = args.run_id
    if args.champion:
        run_id = client.get_model_version_by_alias(REGISTERED_MODEL, "champion").run_id
    model = mlflow.keras.load_model(f"runs:/{run_id}/model")

    split = load_split(args.split_csv)
    test = split[split["split"] == "test"].reset_index(drop=True)
    mode = run_preprocessing(client, run_id)
    images = np.stack([preprocess_file(args.raw_dir / p, mode=mode) for p in test["path"]])
    labels = test["label_idx"].to_numpy()

    with tempfile.TemporaryDirectory() as tmp:
        # Keras probabilities from the logged evaluation (same images, no re-run).
        path = client.download_artifacts(run_id, "evaluation/test/predictions.csv", tmp)
        logged = pd.read_csv(path).set_index("path").loc[test["path"]]
        keras_probs = logged[PROB_COLUMNS].to_numpy(dtype=np.float32)
        # The model's own weights; saved .keras/.weights.h5 files also hold the optimizer
        # state (Keras 3), which inference doesn't need.
        keras_size = sum(int(np.prod(v.shape)) * 4 for v in model.variables) / 1e6

        rows, variants = benchmark(model, images, labels, keras_size, keras_probs)
        for row in rows:
            for key in (
                "size_mb",
                "p50_ms",
                "p95_ms",
                "max_abs_diff",
                "agreement",
                "accuracy",
                "macro_f1",
            ):
                client.log_metric(run_id, f"optimize.{row.name}.{key}", getattr(row, key))
        table = markdown_table(rows)
        (Path(tmp) / "benchmark.md").write_text(table + "\n")
        client.log_artifact(run_id, str(Path(tmp) / "benchmark.md"), "optimize")
        for name, content in variants.items():
            (Path(tmp) / f"{name}.tflite").write_bytes(content)
            client.log_artifact(run_id, str(Path(tmp) / f"{name}.tflite"), "optimize")
    print(f"Test split: {len(test)} images\n")
    print(table)


if __name__ == "__main__":
    main()
