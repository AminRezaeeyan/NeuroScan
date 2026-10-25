"""The deployable model: network, calibration and rejection rules as one MLflow pyfunc.

Input: a batch of preprocessed images, float32 ``(N, H, W, 3)`` in [0, 255], as made
by ``neuroscan.data.preprocess``. Output: one row per image with the calibrated
probability of each class, the predicted class, its confidence and a ``status``:

- ``ok``: answer the prediction;
- ``uncertain``: in distribution, but confidence below the review threshold;
- ``rejected_ood``: the features are far from every training class (not a brain MRI
  like the training data);
- ``rejected_input``: failed a cheap input check (e.g. a color photo).
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

        model = keras.models.load_model(context.artifacts["keras_model"])
        config = json.loads(Path(context.artifacts["config"]).read_text())
        detector = None
        if "ood_detector" in context.artifacts:
            from neuroscan.ood import MahalanobisDetector

            detector = MahalanobisDetector.load(context.artifacts["ood_detector"])
        self.setup(model, config, detector)

    def setup(self, model: Any, config: dict, detector: Any = None) -> None:
        from neuroscan.ood import feature_model

        self.model, self.config, self.detector = model, config, detector
        self.features = None
        if detector is not None:
            import tensorflow as tf

            # Compiled graph: calling the Keras model eagerly is ~10x slower per image.
            features = feature_model(model)
            self.features = tf.function(
                lambda x: features(x, training=False), reduce_retracing=True
            )

    def predict(self, context, model_input, params=None):
        # No annotations: MLflow validates input against annotated predict() signatures.
        from neuroscan.serving.checks import check_array

        x = np.asarray(model_input, dtype=np.float32)
        if self.features is not None:
            feats, raw = (np.asarray(t) for t in self.features(x))
            ood_score = self.detector.score(feats)
            ood = ood_score > self.detector.threshold
        else:
            raw = self.model.predict(x, verbose=0)
            ood_score = np.full(len(x), np.nan)
            ood = np.zeros(len(x), dtype=bool)
        probs = calibrated_softmax(raw, self.config["temperature"])
        names = self.config["class_names"]
        out = pd.DataFrame(probs, columns=names)
        out["prediction"] = [names[i] for i in probs.argmax(axis=1)]
        out["confidence"] = probs.max(axis=1)
        out["ood_score"] = ood_score

        bad_input = np.array([check_array(img) is not None for img in x], dtype=bool)
        uncertain = out["confidence"].to_numpy() < self.config.get("uncertain_threshold", 0.0)
        out["status"] = np.select(
            [bad_input, ood, uncertain], ["rejected_input", "rejected_ood", "uncertain"], "ok"
        )
        return out


def serving_config(
    temperature: float,
    image_size: int,
    source_version: str | None,
    uncertain_threshold: float = 0.0,
    preprocessing: str = "standard",
) -> dict:
    return {
        "class_names": list(CLASS_NAMES),
        "image_size": image_size,
        "preprocessing": preprocessing,
        "temperature": temperature,
        "uncertain_threshold": uncertain_threshold,
        "source_model_version": source_version,
    }


def register_packaged_model(
    client: Any,
    run_id: str,
    temperature: float,
    alias: str | None = "champion",
    ood_path: Path | None = None,
    uncertain_threshold: float = 0.0,
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
    preprocessing = run.data.params.get("data.preprocessing", "standard")
    config = serving_config(temperature, size, source_version, uncertain_threshold, preprocessing)
    detector = None
    if ood_path is not None:
        from neuroscan.ood import MahalanobisDetector

        detector = MahalanobisDetector.load(ood_path)

    with tempfile.TemporaryDirectory() as tmp:
        keras_path = Path(tmp) / "model.keras"
        model.save(keras_path)
        config_path = Path(tmp) / CONFIG_FILE
        config_path.write_text(json.dumps(config, indent=2))

        example = np.full((1, size, size, 3), 128.0, dtype=np.float32)
        wrapper = NeuroScanModel()
        wrapper.setup(model, config, detector)
        signature = infer_signature(example, wrapper.predict(None, example))
        artifacts = {"keras_model": str(keras_path), "config": str(config_path)}
        if ood_path is not None:
            artifacts["ood_detector"] = str(ood_path)

        with mlflow.start_run(run_id=run_id):
            info = mlflow.pyfunc.log_model(
                name=PACKAGED_ARTIFACT,
                python_model=NeuroScanModel(),
                artifacts=artifacts,
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
        "ood_rejection": str(detector is not None).lower(),
        "uncertain_threshold": f"{uncertain_threshold:.2f}",
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
