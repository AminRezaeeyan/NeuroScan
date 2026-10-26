"""Download the Kaggle Brain Tumor MRI dataset.

Credentials: ``~/.kaggle/access_token`` (or ``KAGGLE_API_TOKEN``) locally, a Colab
secret in Colab. Usage: ``python -m neuroscan.data.download --out data/raw``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from neuroscan.config import CLASS_NAMES

DATASET = "masoudnickparvar/brain-tumor-mri-dataset"
SPLITS = ("Training", "Testing")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def count_images(root: Path) -> dict[str, dict[str, int]]:
    return {
        split: {name: len(list_images(root / split / name)) for name in CLASS_NAMES}
        for split in SPLITS
    }


def is_complete(root: Path) -> bool:
    return all((root / split / name).is_dir() for split in SPLITS for name in CLASS_NAMES)


def download(out: Path, force: bool = False) -> Path:
    if is_complete(out) and not force:
        print(f"Dataset already present in {out}")
        return out
    # Imported here: importing kaggle authenticates immediately.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} to {out} ...")
    api.dataset_download_files(DATASET, path=str(out), unzip=True, quiet=False)
    if not is_complete(out):
        raise RuntimeError(f"Unexpected archive layout in {out}; expected {SPLITS} x {CLASS_NAMES}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = download(args.out, args.force)
    counts = count_images(root)
    for split, per_class in counts.items():
        print(f"{split:9s} {per_class}  total={sum(per_class.values())}")
    print(f"All images: {sum(sum(c.values()) for c in counts.values())}")


if __name__ == "__main__":
    main()
