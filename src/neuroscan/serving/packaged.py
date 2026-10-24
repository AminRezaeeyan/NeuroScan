"""The deployable model: Keras network + calibration (and later OOD) as one MLflow pyfunc.

Input: a batch of preprocessed images, float32 ``(N, H, W, 3)`` in [0, 255], as made
by ``neuroscan.data.preprocess``. Output: one row per image with the calibrated
probability of each class, the predicted class and its confidence.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel

from neuroscan.config import CLASS_NAMES

PACKAGED_ARTIFACT = "packaged_model"
CONFIG_FILE = "serving_config.json"


def calibrated_softmax(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


class NeuroScanModel(PythonModel):
    def load_context(self, context: Any) -> None:
        import keras

        import neuroscan.models  # noqa: F401 - registers custom layers

        self.model = keras.models.load_model(context.artifacts["keras_model"])
        self.config = json.loads(Path(context.artifacts["config"]).read_text())

    def predict(self, context, model_input, params=None):
        # No annotations: MLflow validates input against annotated predict() signatures.
        x = np.asarray(model_input, dtype=np.float32)
        raw = self.model.predict(x, verbose=0)
        probs = calibrated_softmax(raw, self.config["temperature"])
        names = self.config["class_names"]
        out = pd.DataFrame(probs, columns=names)
        out["prediction"] = [names[i] for i in probs.argmax(axis=1)]
        out["confidence"] = probs.max(axis=1)
        return out


def serving_config(temperature: float, image_size: int, source_version: str | None) -> dict:
    return {
        "class_names": list(CLASS_NAMES),
        "image_size": image_size,
        "temperature": temperature,
        "source_model_version": source_version,
    }


def register_packaged_model(
    client: Any, run_id: str, temperature: float, alias: str | None = "champion"
) -> str:
    """Log the packaged model into ``run_id``, register it and point ``alias`` at it."""
    import keras
    import tensorflow as tf
    from mlflow.models import infer_signature

    import neuroscan.models  # noqa: F401
    from neuroscan.tracking import REGISTERED_MODEL

    run = client.get_run(run_id)
    source = client.search_model_versions(f"name='{REGISTERED_MODEL}' and run_id='{run_id}'")
    source_version = min((str(v.version) for v in source), key=int, default=None)
    model = mlflow.keras.load_model(f"runs:/{run_id}/model")
    size = int(model.input_shape[1])
    config = serving_config(temperature, size, source_version)

    with tempfile.TemporaryDirectory() as tmp:
        keras_path = Path(tmp) / "model.keras"
        model.save(keras_path)
        config_path = Path(tmp) / CONFIG_FILE
        config_path.write_text(json.dumps(config, indent=2))

        example = np.full((1, size, size, 3), 128.0, dtype=np.float32)
        wrapper = NeuroScanModel()
        wrapper.model, wrapper.config = model, config
        signature = infer_signature(example, wrapper.predict(None, example))

        with mlflow.start_run(run_id=run_id):
            info = mlflow.pyfunc.log_model(
                name=PACKAGED_ARTIFACT,
                python_model=NeuroScanModel(),
                artifacts={"keras_model": str(keras_path), "config": str(config_path)},
                signature=signature,
                registered_model_name=REGISTERED_MODEL,
                pip_requirements=[f"tensorflow=={tf.__version__}", f"keras=={keras.__version__}"],
            )
    # Reopening the run moved its end time to now; restore the training end time.
    client.set_terminated(run_id, status="FINISHED", end_time=run.info.end_time)

    version = str(info.registered_model_version)
    tags = {
        "packaged": "true",
        "calibrated": "true",
        "temperature": f"{temperature:.4f}",
        "run_name": run.info.run_name,
        "split_version": run.data.tags.get("split_version", ""),
    }
    if source_version:
        tags["source_version"] = source_version
    for key, value in tags.items():
        client.set_model_version_tag(REGISTERED_MODEL, version, key, value)
    if alias:
        client.set_registered_model_alias(REGISTERED_MODEL, alias, version)
    return version
