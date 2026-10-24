"""Grad-CAM heatmaps and a sanity check of where they point.

    python -m neuroscan.explain --champion            # test split, logs into the run
    python -m neuroscan.explain --run-id <id> --eval-set val

For each image, Grad-CAM weights the last feature map by the gradient of the
predicted class's pre-softmax score. The check measures how often the heatmap's
peak, and what share of its mass, falls inside a simple brain mask (intensity
threshold + largest connected component). That is a sanity check, not proof that
the model uses the right evidence: a tumor near the skull or a mask error can put
a correct heatmap outside the mask.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import ndimage

from neuroscan.config import CLASS_NAMES
from neuroscan.data.preprocess import preprocess_file

MASK_THRESHOLD = 0.08  # fraction of the image's max intensity


def target_layer_index(model: keras.Model) -> int:
    """Index of the layer whose output is the last spatial feature map."""
    for i in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[i]
        if isinstance(layer, keras.Model):  # pretrained backbone
            return i
        if isinstance(layer, keras.layers.Conv2D):
            return i
    raise ValueError("No convolutional layer or backbone found")


def gradcam(model: keras.Model, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(heatmaps in [0, 1] at input resolution, predicted class per image).

    Assumes a sequential chain of layers (true for all NeuroScan models) whose last
    layer is a softmax Dense; its pre-activation output is the class score.
    """
    target = target_layer_index(model)
    layers = model.layers[1:]  # skip the InputLayer
    dense = layers[-1]
    x = tf.convert_to_tensor(images, dtype=tf.float32)
    features = None
    with tf.GradientTape() as tape:
        h = x
        for layer in layers[:-1]:
            h = layer(h, training=False)
            if layer is model.layers[target]:
                features = h
                tape.watch(features)
        logits = tf.matmul(h, dense.kernel) + dense.bias
        pred = tf.argmax(logits, axis=1)
        score = tf.gather(logits, pred, axis=1, batch_dims=1)
    grads = tape.gradient(score, features)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * features, axis=-1))[..., None]
    cam = tf.image.resize(cam, images.shape[1:3], method="bilinear")[..., 0].numpy()
    peak = cam.reshape(len(cam), -1).max(axis=1)[:, None, None]
    return cam / np.maximum(peak, 1e-8), pred.numpy()


def brain_mask(image: np.ndarray) -> np.ndarray:
    """Largest bright connected region with holes filled (image: (H, W, 3) in [0, 255])."""
    gray = image.mean(axis=-1)
    fg = gray > MASK_THRESHOLD * max(float(gray.max()), 1.0)
    fg = ndimage.binary_opening(fg, iterations=2)
    labels, n = ndimage.label(fg)
    if n == 0:
        return np.ones_like(fg)
    sizes = ndimage.sum(fg, labels, range(1, n + 1))
    return ndimage.binary_fill_holes(labels == (int(np.argmax(sizes)) + 1))


def heatmap_stats(cam: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    total = float(cam.sum())
    return {
        "mask_area": float(mask.mean()),  # chance level for peak_in_brain
        "peak_in_brain": float(mask[peak]),
        "mass_in_brain": float(cam[mask].sum() / total) if total > 0 else float("nan"),
    }


def explain(
    model: keras.Model, preds: pd.DataFrame, raw_dir: Path, batch_size: int = 32
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Per-image heatmap stats for ``preds`` (evaluation predictions) and the heatmaps."""
    rows, cams = [], {}
    for start in range(0, len(preds), batch_size):
        part = preds.iloc[start : start + batch_size]
        images = np.stack([preprocess_file(raw_dir / p) for p in part["path"]])
        heat, _ = gradcam(model, images)
        for (_, row), image, cam in zip(part.iterrows(), images, heat, strict=True):
            stats = heatmap_stats(cam, brain_mask(image))
            rows.append({"path": row["path"], "label": row["label"], "pred": row["pred"], **stats})
            cams[row["path"]] = cam.astype(np.float16)
    df = pd.DataFrame(rows)
    df["correct"] = df["label"] == df["pred"]
    return df, cams


def summarize(stats: pd.DataFrame) -> dict[str, float]:
    out = {
        "mask_area": float(stats["mask_area"].mean()),
        "peak_in_brain": float(stats["peak_in_brain"].mean()),
        "mass_in_brain": float(stats["mass_in_brain"].mean()),
    }
    for name, part in (("correct", stats[stats.correct]), ("wrong", stats[~stats.correct])):
        if len(part):
            out[f"{name}.peak_in_brain"] = float(part["peak_in_brain"].mean())
    for name in CLASS_NAMES:
        part = stats[stats.label == name]
        if len(part):
            out[f"{name}.peak_in_brain"] = float(part["peak_in_brain"].mean())
    return out


def pick_examples(stats: pd.DataFrame, per_class: int = 2, seed: int = 0) -> pd.DataFrame:
    """Per class: up to ``per_class`` correct and ``per_class`` wrong predictions."""
    picks = []
    for name in CLASS_NAMES:
        for correct in (True, False):
            part = stats[(stats.label == name) & (stats.correct == correct)]
            picks.append(part.sample(min(per_class, len(part)), random_state=seed))
    return pd.concat(picks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-id")
    target.add_argument("--champion", action="store_true", help="the run behind @champion")
    parser.add_argument("--eval-set", default="test")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    import mlflow

    import neuroscan.models  # noqa: F401
    from neuroscan.reports.plots import gradcam_grid
    from neuroscan.tracking import REGISTERED_MODEL, get_client

    client = get_client()
    run_id = args.run_id
    if args.champion:
        run_id = client.get_model_version_by_alias(REGISTERED_MODEL, "champion").run_id
    model = mlflow.keras.load_model(f"runs:/{run_id}/model")
    with tempfile.TemporaryDirectory() as tmp:
        path = client.download_artifacts(run_id, f"evaluation/{args.eval_set}/predictions.csv", tmp)
        preds = pd.read_csv(path)

    stats, cams = explain(model, preds, args.raw_dir)
    summary = summarize(stats)
    prefix = f"explain.{args.eval_set}"
    for key, value in summary.items():
        client.log_metric(run_id, f"{prefix}.{key}", value)
    examples = pick_examples(stats)
    images = {p: preprocess_file(args.raw_dir / p) for p in examples["path"]}
    fig = gradcam_grid(examples, images, cams)
    client.log_figure(run_id, fig, f"explain/{args.eval_set}/gradcam_grid.png")
    with tempfile.TemporaryDirectory() as tmp:
        stats.to_csv(Path(tmp) / "heatmap_stats.csv", index=False)
        client.log_artifact(
            run_id, str(Path(tmp) / "heatmap_stats.csv"), f"explain/{args.eval_set}"
        )

    print(f"{args.eval_set}: {len(stats)} images")
    print(f"  heatmap peak inside the brain mask: {summary['peak_in_brain']:.1%}")
    print(f"  heatmap mass inside the brain mask: {summary['mass_in_brain']:.1%}")
    print(f"  chance level (mask area):           {summary['mask_area']:.1%}")
    for key in ("correct", "wrong", *CLASS_NAMES):
        if f"{key}.peak_in_brain" in summary:
            print(f"    peak in brain, {key:11s} {summary[f'{key}.peak_in_brain']:.1%}")


if __name__ == "__main__":
    main()
