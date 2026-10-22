"""Build the fixed train/val/test split, keeping each duplicate group in one split.

    python -m neuroscan.data.splits --out splits/v1.csv

The Kaggle Training/Testing folders are ignored: they share near-duplicates.
Pre-augmented files (``*-aug-*``, 203 meningioma images) are excluded: they are
flipped, rotated or zoomed copies of other images that pixel matching can't reliably
trace to their source, and training augments on the fly anyway.
Duplicates and near-duplicates are grouped with ``neuroscan.data.dedup``, and the
groups are assigned with StratifiedGroupKFold (20 folds: 3 test, 3 val, 14 train ~ 70/15/15).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from neuroscan.config import CLASS_NAMES
from neuroscan.data.datasets import scan_raw
from neuroscan.data.dedup import (
    CORR_THRESHOLD,
    PHASH_THRESHOLD,
    compute_signatures,
    find_groups,
)

N_FOLDS = 20
TEST_FOLDS = {0, 1, 2}
VAL_FOLDS = {3, 4, 5}
SPLIT_COLUMNS = ["path", "label", "group", "kaggle_split", "split"]
AUGMENTED_PATTERN = "-aug-"


def assign_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """``df`` needs ``label`` and ``group``; returns a copy with a ``split`` column."""
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=df.index)
    for k, (_, idx) in enumerate(cv.split(df, df["label"], groups=df["group"])):
        fold.iloc[idx] = k
    split = fold.map(lambda k: "test" if k in TEST_FOLDS else "val" if k in VAL_FOLDS else "train")
    return df.assign(split=split)


def check_split(df: pd.DataFrame) -> None:
    """Raises if a group crosses splits or a split misses a class."""
    crossing = df.groupby("group")["split"].nunique()
    if (crossing > 1).any():
        raise ValueError(f"{int((crossing > 1).sum())} duplicate groups cross splits")
    for name, part in df.groupby("split"):
        missing = set(CLASS_NAMES) - set(part["label"])
        if missing:
            raise ValueError(f"Split {name!r} has no images of {sorted(missing)}")


def build_split(
    raw_dir: Path,
    cache: Path | None = None,
    phash_threshold: int = PHASH_THRESHOLD,
    corr_threshold: float = CORR_THRESHOLD,
    seed: int = 42,
) -> pd.DataFrame:
    df, vectors = compute_signatures(scan_raw(raw_dir), raw_dir, cache)
    groups, _ = find_groups(df, vectors, phash_threshold, corr_threshold)
    groups = groups[~groups["path"].str.contains(AUGMENTED_PATTERN, regex=False)]
    split = assign_splits(groups, seed)
    check_split(split)
    return split.sort_values("path")[SPLIT_COLUMNS].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("splits/v1.csv"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache/signatures.npz"))
    parser.add_argument("--phash-threshold", type=int, default=PHASH_THRESHOLD)
    parser.add_argument("--corr-threshold", type=float, default=CORR_THRESHOLD)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split = build_split(
        args.raw_dir, args.cache, args.phash_threshold, args.corr_threshold, args.seed
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(args.out, index=False)

    table = pd.crosstab(split["split"], split["label"], margins=True)
    print(table.loc[["train", "val", "test", "All"]])
    share = split["split"].value_counts(normalize=True).round(3).to_dict()
    print(f"Share: {share}")
    print(f"Groups: {split['group'].nunique()} (none crosses splits). Wrote {args.out}")
    excluded = len(scan_raw(args.raw_dir)) - len(split)
    print(f"Excluded pre-augmented files: {excluded}")


if __name__ == "__main__":
    main()
