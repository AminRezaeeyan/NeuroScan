"""Log the first published model (trained before this pipeline existed) as a reference run.

    python -m neuroscan.tracking.log_legacy data/legacy/best_model.keras

The model is logged unchanged (no retraining, not registered) so ``neuroscan.evaluate``
can measure it with the same code as every new model. It was trained on the Kaggle
Training folder of an earlier dataset version, so only Kaggle Testing images are fair
evaluation data for it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from neuroscan.tracking import EXPERIMENT, get_client

RUN_NAME = "legacy-kaggle-model"


def log_legacy(model_path: Path, tracking_uri: str | None = None) -> str:
    import keras
    import mlflow
    from mlflow.models import infer_signature

    client = get_client(tracking_uri)
    exp = client.get_experiment_by_name(EXPERIMENT)
    existing = client.search_runs(
        [exp.experiment_id], filter_string=f"attributes.run_name = '{RUN_NAME}'"
    )
    if existing:
        raise FileExistsError(f"{RUN_NAME} already logged as run {existing[0].info.run_id}")

    model = keras.models.load_model(model_path)
    size = model.input_shape[1]
    example = np.full((1, size, size, 3), 128.0, dtype=np.float32)
    signature = infer_signature(example, model.predict(example, verbose=0))
    tags = {
        "model": "baseline_cnn",
        "trained_on": "kaggle",
        "split_version": "kaggle-folders-earlier-version",
        "legacy": "true",
    }
    with mlflow.start_run(run_name=RUN_NAME, tags=tags) as run:
        mlflow.log_params(
            {
                "model.name": "baseline_cnn",
                "params": model.count_params(),
                "source": model_path.name,
            }
        )
        mlflow.keras.log_model(model, name="model", signature=signature)
    return str(run.info.run_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("model_path", type=Path)
    args = parser.parse_args()
    run_id = log_legacy(args.model_path)
    print(f"Logged {RUN_NAME} as run {run_id}")
    print(
        "Evaluate with: python -m neuroscan.evaluate --run-id "
        f"{run_id} --eval-set kaggle-testing kaggle-testing-clean"
    )


if __name__ == "__main__":
    main()
