"""How much of the class is visible in simple pixel statistics of the model's input?

    python -m neuroscan.shortcut                       # both preprocessing modes
    python -m neuroscan.shortcut --modes standard

A random forest sees only global statistics of the preprocessed 224x224 image
(brightness, contrast, intensity percentiles, background share, sharpness), never
the anatomy. Its 5-fold cross-validated accuracy on the training split measures how
much source information the preprocessing leaves for a model to exploit; chance is
25%. Comparing modes shows whether shortcut-reduced preprocessing removes it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from neuroscan.data.preprocess import PREPROCESSING_MODES, preprocess_file

PIXEL_FEATURES = [
    "mean",
    "std",
    "p05",
    "p50",
    "p95",
    "background_frac",
    "sharpness",
    "center_mean",
    "border_mean",
]


def pixel_stats(image: np.ndarray) -> dict[str, float]:
    gray = image.mean(axis=-1)
    p05, p50, p95 = np.percentile(gray, [5, 50, 95])
    laplacian = np.abs(
        4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    )
    h, w = gray.shape
    center = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    border = np.concatenate([gray[: h // 10].ravel(), gray[-h // 10 :].ravel()])
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "background_frac": float((gray < 10).mean()),
        "sharpness": float(laplacian.mean()),
        "center_mean": float(center.mean()),
        "border_mean": float(border.mean()),
    }


def probe_accuracy(df: pd.DataFrame, raw_dir: Path, mode: str, seed: int = 0) -> dict[str, float]:
    """Cross-validated pixel-probe accuracy; folds keep duplicate groups together."""
    feats = pd.DataFrame([pixel_stats(preprocess_file(raw_dir / p, mode=mode)) for p in df["path"]])
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    forest = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=2)
    pred = cross_val_predict(forest, feats[PIXEL_FEATURES], df["label"], cv=cv, groups=df["group"])
    out = {"accuracy": float((pred == df["label"].to_numpy()).mean())}
    for name in sorted(df["label"].unique()):
        mask = df["label"].to_numpy() == name
        out[f"{name}.recall"] = float((pred[mask] == name).mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split-csv", type=Path, default=Path("splits/v1.csv"))
    parser.add_argument("--modes", nargs="+", default=list(PREPROCESSING_MODES))
    args = parser.parse_args()

    split = pd.read_csv(args.split_csv)
    train = split[split["split"] == "train"].reset_index(drop=True)
    print(f"Pixel-statistics probe, 5-fold CV on {len(train)} training images (chance 25%)\n")
    print("| Preprocessing | Probe accuracy | Recall: glioma / meningioma / no tumor / pituitary |")
    print("| --- | ---: | --- |")
    for mode in args.modes:
        r = probe_accuracy(train, args.raw_dir, mode)
        recalls = " / ".join(
            f"{r[f'{c}.recall']:.0%}" for c in ("glioma", "meningioma", "notumor", "pituitary")
        )
        print(f"| {mode} | {r['accuracy']:.1%} | {recalls} |")


if __name__ == "__main__":
    main()
