"""Find exact and near-duplicate images and group them.

    python -m neuroscan.data.dedup --raw-dir data/raw --out data/duplicates.csv

Two images are linked when any of these holds:

* **exact**: identical decoded pixels (MD5);
* **phash**: perceptual-hash Hamming distance <= ``--phash-threshold`` (re-encoded,
  resized or slightly cropped copies);
* **corr**: rank correlation of the 64x64 grayscale images >= ``--corr-threshold``.
  Rank correlation ignores any monotonic brightness/contrast change, so it finds
  contrast-altered copies and adjacent slices of the same scan that pHash misses.

Links are merged transitively into groups, which the split treats as single units.
The thresholds were set by inspecting example pairs at each distance: pHash 6 and
correlation 0.95 already pair different scans, and 0.95 chains into one huge group.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterator
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata

from neuroscan.data.datasets import scan_raw

PHASH_THRESHOLD = 4
CORR_THRESHOLD = 0.96
CORR_SIZE = 64


def image_signature(path: Path) -> tuple[str, int, np.ndarray]:
    """(MD5 of the grayscale pixels and shape, 64-bit pHash, unit-norm rank vector)."""
    with Image.open(path) as img:
        gray = img.convert("L")
    pixels = np.asarray(gray)
    digest = hashlib.md5(  # nosec B324 - duplicate detection, not security
        pixels.tobytes() + str(pixels.shape).encode(), usedforsecurity=False
    ).hexdigest()
    phash = int(str(imagehash.phash(gray)), 16)
    small = np.asarray(
        gray.resize((CORR_SIZE, CORR_SIZE), Image.Resampling.BILINEAR), dtype=np.float32
    )
    ranks = rankdata(small.ravel()).astype(np.float32)
    ranks -= ranks.mean()
    ranks /= np.linalg.norm(ranks) + 1e-12  # dot product of two vectors = rank correlation
    return digest, phash, ranks


def compute_signatures(
    df: pd.DataFrame, raw_dir: Path, cache: Path | None = None
) -> tuple[pd.DataFrame, np.ndarray]:
    """Adds ``md5`` and ``phash`` columns and returns the rank vectors (cached as .npz)."""
    if cache is not None and cache.exists():
        stored = np.load(cache, allow_pickle=False)
        if list(stored["paths"]) == list(df["path"]):
            out = df.assign(md5=stored["md5"], phash=stored["phash"].astype(np.uint64))
            return out, stored["vectors"]
    sigs = [image_signature(raw_dir / p) for p in df["path"]]
    md5 = np.array([s[0] for s in sigs])
    phash = np.array([s[1] for s in sigs], dtype=np.uint64)
    vectors = np.stack([s[2] for s in sigs])
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, paths=df["path"].to_numpy(dtype=str), md5=md5, phash=phash, vectors=vectors
        )
    return df.assign(md5=md5, phash=phash), vectors


def _bits(hashes: np.ndarray) -> np.ndarray:
    """uint64 hashes -> (N, 64) bit matrix."""
    as_bytes = hashes.astype(">u8").view(np.uint8).reshape(-1, 8)
    return np.unpackbits(as_bytes, axis=1)


def phash_pairs(phashes: np.ndarray, threshold: int, chunk: int = 512) -> Iterator[tuple[int, int]]:
    """All (i, j), i < j, with pHash Hamming distance <= threshold."""
    bits = _bits(phashes).astype(np.int16)
    ones = bits.sum(1)
    for start in range(0, len(bits), chunk):
        block = bits[start : start + chunk]
        # Hamming distance = a + b - 2 * (a & b), counted in bits.
        dist = ones[start : start + chunk, None] + ones[None, :] - 2 * (block @ bits.T)
        for r, c in zip(*np.nonzero(dist <= threshold), strict=True):
            if start + r < c:
                yield start + int(r), int(c)


def corr_pairs(
    vectors: np.ndarray, threshold: float, chunk: int = 1024
) -> Iterator[tuple[int, int]]:
    """All (i, j), i < j, with rank correlation >= threshold."""
    for start in range(0, len(vectors), chunk):
        sims = vectors[start : start + chunk] @ vectors.T
        for r, c in zip(*np.nonzero(sims >= threshold), strict=True):
            if start + r < c:
                yield start + int(r), int(c)


def phash_distance(a: int, b: int) -> int:
    return (int(a) ^ int(b)).bit_count()


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        a, b = self.find(i), self.find(j)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def find_groups(
    df: pd.DataFrame,
    vectors: np.ndarray,
    phash_threshold: int = PHASH_THRESHOLD,
    corr_threshold: float = CORR_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (df with ``group`` and ``group_size``, linked pairs with their scores).

    ``group`` is the smallest row index in the group, so it is stable for a fixed dataset.
    """
    df = df.reset_index(drop=True)
    links: dict[tuple[int, int], set[str]] = {}
    for _, idx in df.groupby("md5").indices.items():
        for k in idx[1:]:
            links.setdefault((int(idx[0]), int(k)), set()).add("exact")
    for pair in phash_pairs(df["phash"].to_numpy(dtype=np.uint64), phash_threshold):
        links.setdefault(pair, set()).add("phash")
    for pair in corr_pairs(vectors, corr_threshold):
        links.setdefault(pair, set()).add("corr")

    uf = UnionFind(len(df))
    for i, j in links:
        uf.union(i, j)
    out = df.assign(group=[uf.find(i) for i in range(len(df))])
    out["group_size"] = out.groupby("group")["path"].transform("size")

    phashes = df["phash"].to_numpy(dtype=np.uint64)
    pairs = pd.DataFrame(
        [
            {
                "path_i": df.at[i, "path"],
                "path_j": df.at[j, "path"],
                "reason": "+".join(sorted(reasons)),
                "phash_distance": phash_distance(phashes[i], phashes[j]),
                "corr": round(float(vectors[i] @ vectors[j]), 4),
            }
            for (i, j), reasons in sorted(links.items())
        ],
        columns=["path_i", "path_j", "reason", "phash_distance", "corr"],
    )
    return out, pairs


def summarize(groups: pd.DataFrame) -> dict[str, object]:
    dup = groups[groups["group_size"] > 1]
    per_group = dup.groupby("group").agg(
        splits=("kaggle_split", lambda s: "+".join(sorted(set(s)))),
        labels=("label", lambda s: "+".join(sorted(set(s)))),
        size=("path", "size"),
    )
    cross = per_group[per_group["splits"] == "Testing+Training"]
    test_rows = groups[groups["kaggle_split"] == "Testing"]
    leaked_test = test_rows[test_rows["group"].isin(cross.index)]
    return {
        "images": len(groups),
        "images_in_duplicate_groups": len(dup),
        "duplicate_groups": len(per_group),
        "largest_group": int(per_group["size"].max()) if len(per_group) else 0,
        "groups_within_training": int((per_group["splits"] == "Training").sum()),
        "groups_within_testing": int((per_group["splits"] == "Testing").sum()),
        "groups_across_training_testing": len(cross),
        "testing_images_with_a_training_duplicate": len(leaked_test),
        "testing_images_with_a_training_duplicate_per_class": leaked_test["label"]
        .value_counts()
        .to_dict(),
        "duplicates_per_class": dup["label"].value_counts().to_dict(),
        "groups_with_conflicting_labels": int(per_group["labels"].str.contains(r"\+").sum()),
    }


def save_examples(pairs: pd.DataFrame, raw_dir: Path, path: Path, seed: int = 0) -> None:
    """Training/Testing pairs for each link type, plus a near miss that is not linked."""
    from neuroscan.reports.plots import duplicate_examples

    cross = pairs[pairs["path_i"].str[:4] != pairs["path_j"].str[:4]]
    chosen = []
    for reason, caption in (
        ("exact", "identical pixels"),
        ("phash", "pHash {d}"),
        ("corr", "contrast changed, corr {c:.2f}"),
    ):
        rows = cross[cross["reason"].str.split("+").map(lambda r, k=reason: k in r)]
        if reason != "exact":
            rows = rows[~rows["reason"].str.contains("exact")]
        if reason == "corr":
            rows = rows[rows["reason"] == "corr"]  # found by correlation only
        for _, r in rows.sample(min(2, len(rows)), random_state=seed).iterrows():
            top, bottom = sorted((r["path_i"], r["path_j"]), key=lambda p: not p.startswith("Tr"))
            chosen.append((top, bottom, caption.format(d=r["phash_distance"], c=r["corr"])))
    fig = duplicate_examples(chosen, raw_dir, "Near-duplicates across Kaggle Training and Testing")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/duplicates.csv"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache/signatures.npz"))
    parser.add_argument("--phash-threshold", type=int, default=PHASH_THRESHOLD)
    parser.add_argument("--corr-threshold", type=float, default=CORR_THRESHOLD)
    parser.add_argument("--figure", type=Path, help="save example pairs to this PNG")
    args = parser.parse_args()

    df, vectors = compute_signatures(scan_raw(args.raw_dir), args.raw_dir, args.cache)
    groups, pairs = find_groups(df, vectors, args.phash_threshold, args.corr_threshold)
    dup = groups[groups["group_size"] > 1].sort_values(["group", "path"])
    cols = ["path", "label", "kaggle_split", "group", "group_size", "md5"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dup[cols].to_csv(args.out, index=False)
    pairs_path = args.out.with_name("duplicate_pairs.csv")
    pairs.to_csv(pairs_path, index=False)

    for key, value in summarize(groups).items():
        print(f"{key:52s} {value}")
    print(f"Links by reason: {pairs['reason'].value_counts().to_dict()}")
    print(f"Wrote {args.out} and {pairs_path}")
    if args.figure:
        save_examples(pairs, args.raw_dir, args.figure)
        print(f"Wrote {args.figure}")


if __name__ == "__main__":
    main()
