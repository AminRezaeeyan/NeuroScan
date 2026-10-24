"""Build the out-of-distribution (OOD) test set from public Kaggle datasets.

    python -m neuroscan.data.ood_download --out data/ood

Downloads each source, samples ``--per-source`` images into ``<out>/<source>/`` and
deletes the rest. These images are only used to *test* OOD rejection; the rejection
threshold is chosen on validation MRIs alone.
"""

from __future__ import annotations

import argparse
import random
import shutil
import tempfile
from pathlib import Path

SOURCES = {
    # source name: (Kaggle dataset, path keywords that exclude non-image files such as masks)
    "natural_photos": ("prasunroy/natural-images", ()),
    "chest_xray": ("khoongweihao/covid19-xray-dataset-train-test-sets", ()),
    "ct": ("luisblanche/covidct", ()),
    "spine_mri": ("anoukstein/spider-mri-spine-t2-png", ("mask", "label", "seg")),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def candidate_images(root: Path, exclude: tuple[str, ...]) -> list[Path]:
    files = [
        p
        for p in root.rglob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES
        and not any(k in p.relative_to(root).as_posix().lower() for k in exclude)
    ]
    return sorted(files)


def sample_into(files: list[Path], dest: Path, n: int, seed: int) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    chosen = random.Random(seed).sample(files, min(n, len(files)))  # noqa: S311 - sampling
    for i, src in enumerate(chosen):
        shutil.copy(src, dest / f"{i:04d}{src.suffix.lower()}")
    return len(chosen)


def download_source(name: str, out: Path, n: int, seed: int) -> int:
    from kaggle.api.kaggle_api_extended import KaggleApi

    dataset, exclude = SOURCES[name]
    api = KaggleApi()
    api.authenticate()
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        print(f"{name}: downloading {dataset} ...")
        api.dataset_download_files(dataset, path=tmp, unzip=True, quiet=False)
        files = candidate_images(Path(tmp), exclude)
        if not files:
            raise RuntimeError(f"{dataset}: no image files found")
        return sample_into(files, out / name, n, seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/ood"))
    parser.add_argument("--per-source", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="*", choices=sorted(SOURCES), help="subset of sources")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.only or SOURCES:
        if (args.out / name).is_dir() and any((args.out / name).iterdir()):
            print(f"{name}: already present, skipping")
            continue
        count = download_source(name, args.out, args.per_source, args.seed)
        print(f"{name}: {count} images in {args.out / name}")


if __name__ == "__main__":
    main()
