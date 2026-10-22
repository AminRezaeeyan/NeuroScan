"""Matplotlib figures shared by training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


def learning_curves(history: Mapping[str, Sequence[float]], stage_boundary: int = 0) -> Figure:
    """Loss, accuracy, validation macro F1 and learning rate per epoch."""
    panels = [
        ("Loss", ["loss", "val_loss"]),
        ("Accuracy", ["accuracy", "val_accuracy"]),
        ("Validation macro F1", ["val_macro_f1"]),
        ("Learning rate", ["learning_rate"]),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4))
    for ax, (title, keys) in zip(axes, panels, strict=True):
        for key in keys:
            if key in history:
                values = history[key]
                ax.plot(range(1, len(values) + 1), values, marker=".", label=key)
        if stage_boundary:
            ax.axvline(
                stage_boundary + 0.5, color="grey", ls="--", lw=1, label="fine-tuning starts"
            )
        if title == "Learning rate":
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
