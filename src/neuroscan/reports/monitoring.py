"""Dashboard figure from the API's prediction log (NEUROSCAN_PREDICTION_LOG).

    python -m neuroscan.reports.monitoring predictions.jsonl --out docs/figures/monitoring.png

Shows status shares over time, the calibrated confidence distribution of answered
predictions, and latency. A shift in these (more rejections, lower confidence)
is the cheapest early warning that incoming images differ from the training data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

STATUS_COLORS = {
    "ok": "#2f9e6e",
    "uncertain": "#e0a030",
    "rejected_ood": "#d9534f",
    "rejected_input": "#8e6cc0",
}


def dashboard(log: pd.DataFrame, bins: int = 20) -> Figure:
    log = log.sort_values("time").reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))

    window = max(len(log) // bins, 1)
    shares = (
        pd.get_dummies(log["status"])
        .reindex(columns=list(STATUS_COLORS), fill_value=0)
        .astype(float)
        .rolling(window, min_periods=1)
        .mean()
    )
    axes[0].stackplot(
        range(len(shares)),
        *[shares[c] for c in STATUS_COLORS],
        labels=list(STATUS_COLORS),
        colors=list(STATUS_COLORS.values()),
        alpha=0.85,
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("request")
    axes[0].set_title(f"Status share (rolling {window} requests)")
    axes[0].legend(fontsize=8, loc="lower left")

    answered = log["confidence"].dropna()
    axes[1].hist(answered, bins=20, range=(0, 1), color="#3c8dbc")
    axes[1].set_xlabel("calibrated confidence")
    axes[1].set_title(f"Confidence of answered predictions (n={len(answered)})")

    axes[2].hist(log["latency_ms"], bins=30, color="#7f8c8d")
    p95 = log["latency_ms"].quantile(0.95)
    axes[2].axvline(p95, color="black", ls="--", lw=1, label=f"p95 {p95:.0f} ms")
    axes[2].set_xlabel("latency (ms)")
    axes[2].set_title("Request latency")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    versions = ", ".join(f"v{v}" for v in sorted(log["model_version"].astype(str).unique()))
    fig.suptitle(f"NeuroScan monitoring: {len(log)} predictions, model {versions}")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/figures/monitoring.png"))
    args = parser.parse_args()
    log = pd.read_json(args.log, lines=True)
    dashboard(log).savefig(args.out, dpi=120)
    print(f"Wrote {args.out} from {len(log)} predictions")


if __name__ == "__main__":
    main()
