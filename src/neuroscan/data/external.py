"""External test set: PMRAM, brain MRIs from hospitals in Bangladesh.

    python -m neuroscan.data.external --out data/external/pmram     # download (~168 MB)

PMRAM (Kaggle ``orvile/pmram-bangladeshi-brain-cancer-mri-dataset``, CC BY-SA 4.0)
has the same four classes as the training data but comes from different hospitals.
Only its 1,600 original images (400 per class) are kept; the dataset's augmented
copies of those would count the same scan several times.

``find_overlap`` checks it against the training dataset with the same duplicate
detector used for the split, so images that also exist in the training data can be
excluded before evaluation.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from neuroscan.config import CLASS_NAMES
from neuroscan.data.download import IMAGE_SUFFIXES

DATASET = "orvile/pmram-bangladeshi-brain-cancer-mri-dataset"
LABEL_KEYWORDS = {
    "glioma": "glioma",
    "menin": "meningioma",
    "pituitar": "pituitary",
    "normal": "notumor",
    "no tumor": "notumor",
    "notumor": "notumor",
}


def label_from_path(path: str) -> str | None:
    folder = path.lower().rsplit("/", 2)[-2] if "/" in path else ""
    for key, label in LABEL_KEYWORDS.items():
        if key in folder:
            return label
    return None


def original_images(root: Path) -> pd.DataFrame:
    """Original (non-augmented) images with labels from their folder names."""
    rows = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.suffix.lower() not in IMAGE_SUFFIXES or "augment" in rel.lower():
            continue
        label = label_from_path(rel)
        if label is not None:
            rows.append({"source_path": rel, "label": label})
    return pd.DataFrame(rows, columns=["source_path", "label"])


def download(out: Path) -> pd.DataFrame:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        api.dataset_download_files(DATASET, path=tmp, unzip=True, quiet=False)
        images = original_images(Path(tmp))
        if images.empty:
            raise RuntimeError(f"No original images found in {DATASET}")
        for label, part in images.groupby("label"):
            dest = out / str(label)
            dest.mkdir(parents=True, exist_ok=True)
            for i, rel in enumerate(part["source_path"]):
                src = Path(tmp) / rel
                shutil.copy(src, dest / f"{i:04d}{src.suffix.lower()}")
    return scan(out)


def scan(root: Path) -> pd.DataFrame:
    """One row per image: path (relative to root), label, label_idx."""
    rows = [
        {"path": p.relative_to(root).as_posix(), "label": name, "label_idx": idx}
        for idx, name in enumerate(CLASS_NAMES)
        for p in sorted((root / name).glob("*"))
        if p.suffix.lower() in IMAGE_SUFFIXES
    ]
    return pd.DataFrame(rows, columns=["path", "label", "label_idx"])


def find_overlap(
    external: pd.DataFrame,
    external_dir: Path,
    raw_dir: Path,
    raw_cache: Path | None = Path("data/cache/signatures.npz"),
) -> pd.Series:
    """For each external image, whether it is a duplicate of a training-dataset image.

    Uses the training split's duplicate detector (identical pixels, pHash, rank
    correlation) on the two datasets together.
    """
    from neuroscan.data.datasets import scan_raw
    from neuroscan.data.dedup import compute_signatures, find_groups

    internal, internal_vec = compute_signatures(scan_raw(raw_dir), raw_dir, raw_cache)
    ext, ext_vec = compute_signatures(external[["path", "label"]], external_dir, None)
    both = pd.concat(
        [internal.assign(source="internal"), ext.assign(source="external")], ignore_index=True
    )
    both["path"] = both["source"] + ":" + both["path"]
    groups, _ = find_groups(both, np.concatenate([internal_vec, ext_vec]))
    has_internal = groups.groupby("group")["source"].transform(lambda s: (s == "internal").any())
    flags = has_internal[groups["source"] == "external"].to_numpy()
    return pd.Series(flags, index=external.index, name="overlaps_training_data")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/external/pmram"))
    args = parser.parse_args()
    if args.out.is_dir() and any(args.out.iterdir()):
        df = scan(args.out)
        print(f"Already present: {args.out}")
    else:
        df = download(args.out)
    print(df["label"].value_counts().reindex(CLASS_NAMES).to_string())
    print(f"{len(df)} original images in {args.out}")


if __name__ == "__main__":
    main()
