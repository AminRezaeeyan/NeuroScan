"""Out-of-distribution (OOD) rejection with a training-free Mahalanobis detector.

    python -m neuroscan.ood --champion                 # fit, evaluate, log
    python -m neuroscan.ood --champion --register      # also package and promote

The detector uses the champion's penultimate features (the ``features`` layer):
class means and one shared, shrunk covariance (Ledoit-Wolf) from the training
images; an image's score is its smallest Mahalanobis distance to a class mean.
The threshold is the 98th percentile of validation MRI scores, so about 2% of real
MRIs are rejected and no OOD image is used to choose it. The OOD images in
``--ood-dir`` (``python -m neuroscan.data.ood_download``) only test it.

Maximum softmax probability (after temperature scaling) is reported on the same
data as a reference detector. Predictions that pass the OOD check but have a
calibrated confidence below the ``uncertain`` threshold are flagged for review;
that threshold is the lowest one at which validation accuracy on the remaining
predictions reaches ``--target-accuracy``.
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

from neuroscan.data.preprocess import decode_and_resize
from neuroscan.models.common import FEATURES_LAYER
from neuroscan.serving.checks import check_array

REJECT_QUANTILE = 0.98
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class MahalanobisDetector:
    means: np.ndarray  # (classes, D)
    precision: np.ndarray  # (D, D)
    threshold: float = float("inf")

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray) -> MahalanobisDetector:
        classes = np.unique(labels)
        means = np.stack([features[labels == c].mean(axis=0) for c in classes])
        centered = np.concatenate([features[labels == c] - means[i] for i, c in enumerate(classes)])
        cov = LedoitWolf().fit(centered).covariance_
        return cls(means.astype(np.float64), np.linalg.inv(cov).astype(np.float64))

    def score(self, features: np.ndarray) -> np.ndarray:
        """Smallest squared Mahalanobis distance to any class mean (higher = more OOD)."""
        f = features.astype(np.float64)
        dists = []
        for mean in self.means:
            d = f - mean
            dists.append(np.einsum("nd,de,ne->n", d, self.precision, d))
        return np.min(np.stack(dists, axis=1), axis=1)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, means=self.means, precision=self.precision, threshold=self.threshold
        )

    @classmethod
    def load(cls, path: str | Path) -> MahalanobisDetector:
        z = np.load(path)
        return cls(z["means"], z["precision"], float(z["threshold"]))


def feature_model(model: keras.Model) -> keras.Model:
    """Same inputs; outputs (penultimate features, class probabilities)."""
    return keras.Model(model.inputs, [model.get_layer(FEATURES_LAYER).output, model.output])


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def load_images(
    paths: list[Path], size: int, mode: str = "standard", batch_size: int = 32
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices([str(p) for p in paths])
    ds = ds.map(lambda p: decode_and_resize(tf.io.read_file(p), size, mode), tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def extract(
    model: keras.Model, paths: list[Path], mode: str = "standard"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(features, probabilities, input-check rejected flags) for image files."""
    fm = feature_model(model)
    size = int(model.input_shape[1])
    feats, probs, rejected = [], [], []
    for batch in load_images(paths, size, mode):
        f, p = fm(batch, training=False)
        feats.append(np.asarray(f))
        probs.append(np.asarray(p))
        rejected += [check_array(img) is not None for img in batch.numpy()]
    return np.concatenate(feats), np.concatenate(probs), np.array(rejected, dtype=bool)


def fpr_at_95_tpr(in_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Share of OOD images accepted when the threshold accepts 95% of real MRIs."""
    threshold = np.quantile(in_scores, 0.95)
    return float((ood_scores <= threshold).mean())


def detection_metrics(in_scores: np.ndarray, ood_scores: np.ndarray) -> dict[str, float]:
    y = np.r_[np.zeros(len(in_scores)), np.ones(len(ood_scores))]
    return {
        "auroc": float(roc_auc_score(y, np.r_[in_scores, ood_scores])),
        "fpr_at_95tpr": fpr_at_95_tpr(in_scores, ood_scores),
    }


def uncertain_threshold(
    confidence: np.ndarray, correct: np.ndarray, target_accuracy: float
) -> float:
    """Lowest confidence cut at which accuracy on predictions above it reaches the target."""
    for t in np.round(np.arange(0.40, 1.0, 0.01), 2):
        kept = confidence >= t
        if kept.any() and correct[kept].mean() >= target_accuracy:
            return float(t)
    return 1.0


def max_softmax_score(probs: np.ndarray, temperature: float) -> np.ndarray:
    from neuroscan.calibrate import apply_temperature

    return -apply_temperature(probs, temperature).max(axis=1)  # higher = more OOD


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-id")
    target.add_argument("--champion", action="store_true", help="the run behind @champion")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split-csv", type=Path, default=Path("splits/v1.csv"))
    parser.add_argument("--ood-dir", type=Path, default=Path("data/ood"))
    parser.add_argument("--target-accuracy", type=float, default=0.99)
    parser.add_argument("--register", action="store_true", help="package and promote")
    args = parser.parse_args()

    import mlflow

    import neuroscan.models  # noqa: F401
    from neuroscan.data.datasets import load_split
    from neuroscan.reports.plots import ood_histogram
    from neuroscan.tracking import REGISTERED_MODEL, get_client, run_preprocessing

    client = get_client()
    run_id = args.run_id
    if args.champion:
        run_id = client.get_model_version_by_alias(REGISTERED_MODEL, "champion").run_id
    metrics_in_run = client.get_run(run_id).data.metrics
    temperature = metrics_in_run.get("calibration.temperature")
    if temperature is None:
        raise SystemExit("Calibrate this run first: python -m neuroscan.calibrate --run-id ...")
    model = mlflow.keras.load_model(f"runs:/{run_id}/model")
    mode = run_preprocessing(client, run_id)

    split = load_split(args.split_csv)
    data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    labels: dict[str, np.ndarray] = {}
    for name in ("train", "val", "test"):
        part = split[split["split"] == name]
        data[name] = extract(model, [args.raw_dir / p for p in part["path"]], mode)
        labels[name] = part["label_idx"].to_numpy()
        print(f"{name}: {len(part)} MRIs")
    sources = (
        sorted(p.name for p in args.ood_dir.iterdir() if p.is_dir())
        if args.ood_dir.is_dir()
        else []
    )
    for source in sources:
        data[f"ood/{source}"] = extract(model, list_images(args.ood_dir / source), mode)
        print(f"ood/{source}: {len(data[f'ood/{source}'][0])} images")
    if not sources:
        print(f"No OOD images in {args.ood_dir}: fitting and thresholds only")

    detector = MahalanobisDetector.fit(data["train"][0], labels["train"])
    scores = {k: detector.score(v[0]) for k, v in data.items()}
    msp = {k: max_softmax_score(v[1], temperature) for k, v in data.items()}
    detector.threshold = float(np.quantile(scores["val"], REJECT_QUANTILE))

    from neuroscan.calibrate import apply_temperature

    val_probs = apply_temperature(data["val"][1], temperature)
    unc = uncertain_threshold(
        val_probs.max(axis=1), val_probs.argmax(axis=1) == labels["val"], args.target_accuracy
    )

    metrics: dict[str, float] = {"threshold": detector.threshold, "uncertain_threshold": unc}
    for name in ("val", "test"):
        accepted = (scores[name] <= detector.threshold) & ~data[name][2]
        probs = apply_temperature(data[name][1], temperature)
        confident = accepted & (probs.max(axis=1) >= unc)
        correct = probs.argmax(axis=1) == labels[name]
        metrics[f"{name}.mri_rejected"] = float(1 - accepted.mean())
        metrics[f"{name}.uncertain"] = float((accepted & ~confident).mean())
        metrics[f"{name}.accuracy_when_answered"] = float(correct[confident].mean())
    for source in sources:
        key = f"ood/{source}"
        for det, sc in (("mahalanobis", scores), ("msp", msp)):
            for m, v in detection_metrics(sc["test"], sc[key]).items():
                metrics[f"{source}.{det}.{m}"] = v
        rejected = (scores[key] > detector.threshold) | data[key][2]
        metrics[f"{source}.rejected"] = float(rejected.mean())
        metrics[f"{source}.input_check_rejected"] = float(data[key][2].mean())
    if sources:
        all_ood = np.concatenate([scores[f"ood/{s}"] for s in sources])
        all_msp = np.concatenate([msp[f"ood/{s}"] for s in sources])
        for det, sc, ood in (("mahalanobis", scores, all_ood), ("msp", msp, all_msp)):
            for m, v in detection_metrics(sc["test"], ood).items():
                metrics[f"all.{det}.{m}"] = v

    for key, value in metrics.items():
        client.log_metric(run_id, f"ood.{key}", value)
    fig = ood_histogram(
        {"test MRIs": scores["test"], **{s: scores[f"ood/{s}"] for s in sources}},
        detector.threshold,
    )
    client.log_figure(run_id, fig, "ood/score_histogram.png")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ood_detector.npz"
        detector.save(path)
        client.log_artifact(run_id, str(path), "ood")
        print_summary(metrics, sources)
        if args.register:
            from neuroscan.serving.packaged import register_packaged_model

            version = register_packaged_model(
                client,
                run_id,
                temperature=temperature,
                ood_path=path,
                uncertain_threshold=unc,
            )
            print(f"Registered model with OOD rejection as version {version}; @champion moved")


def print_summary(m: dict[str, float], sources: list[str]) -> None:
    print(f"\nOOD threshold (98th pct of val MRI scores): {m['threshold']:.1f}")
    print(f"Uncertain below calibrated confidence {m['uncertain_threshold']:.2f}")
    for name in ("val", "test"):
        print(
            f"  {name}: real MRIs rejected {m[f'{name}.mri_rejected']:.1%}, "
            f"flagged uncertain {m[f'{name}.uncertain']:.1%}, "
            f"accuracy when answered {m[f'{name}.accuracy_when_answered']:.1%}"
        )
    if sources:
        print("\n  source           rejected  input check  AUROC maha / msp   FPR@95TPR maha / msp")
        for s in [*sources, "all"]:
            rej = f"{m[f'{s}.rejected']:.1%}" if f"{s}.rejected" in m else "  -  "
            chk = (
                f"{m[f'{s}.input_check_rejected']:.1%}"
                if f"{s}.input_check_rejected" in m
                else "  -  "
            )
            print(
                f"  {s:16s} {rej:>8s}  {chk:>11s}   "
                f"{m[f'{s}.mahalanobis.auroc']:.3f} / {m[f'{s}.msp.auroc']:.3f}     "
                f"{m[f'{s}.mahalanobis.fpr_at_95tpr']:.1%} / {m[f'{s}.msp.fpr_at_95tpr']:.1%}"
            )


if __name__ == "__main__":
    main()
