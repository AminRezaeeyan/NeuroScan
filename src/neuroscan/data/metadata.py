"""Per-image file and acquisition properties, and the metadata-only shortcut probe.

The probe is a random forest that sees only these properties (no anatomy). If it
predicts the class well, the label is partly recoverable from how an image was
produced, and a CNN can learn that shortcut too.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

PROBE_FEATURES = [
    "width",
    "height",
    "aspect",
    "file_kb",
    "mean_intensity",
    "std_intensity",
    "background_frac",
    "max_channel_diff",
    "is_png",
]


def image_metadata(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    with Image.open(path) as img:
        width, height, fmt, mode = img.width, img.height, img.format, img.mode
        arr = np.asarray(img.convert("RGB"), dtype=np.int16)
    gray = arr.mean(axis=2)
    return {
        "width": width,
        "height": height,
        "img_format": fmt,
        "color_mode": mode,
        "file_kb": len(data) / 1024,
        "md5": hashlib.md5(data, usedforsecurity=False).hexdigest(),  # nosec B324
        "max_channel_diff": int(np.abs(arr - arr[..., :1]).max()),
        "mean_intensity": float(gray.mean()),
        "std_intensity": float(gray.std()),
        "background_frac": float((gray < 10).mean()),
    }


def compute_metadata(
    df: pd.DataFrame, raw_dir: str | Path, cache: str | Path | None = None
) -> pd.DataFrame:
    """``df`` joined with one row of metadata per image (cached as CSV, keyed by path)."""
    raw_dir = Path(raw_dir)
    cached = pd.DataFrame()
    if cache is not None and Path(cache).exists():
        cached = pd.read_csv(cache)
    have = set(cached["path"]) if len(cached) else set()
    missing = [p for p in df["path"] if p not in have]
    if missing:
        new = pd.DataFrame([{"path": p, **image_metadata(raw_dir / p)} for p in missing])
        cached = pd.concat([cached, new], ignore_index=True) if len(cached) else new
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            cached.to_csv(cache, index=False)
    out = df.merge(cached, on="path", how="left")
    out["aspect"] = out["width"] / out["height"]
    out["is_png"] = (out["img_format"] == "PNG").astype(int)
    return out


def fit_probe(meta: pd.DataFrame, seed: int = 0) -> RandomForestClassifier:
    probe = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    probe.fit(meta[PROBE_FEATURES], meta["label"])
    return probe
