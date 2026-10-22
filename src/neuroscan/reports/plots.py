"""Matplotlib figures shared by training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402


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


def duplicate_examples(pairs: list[tuple[str, str, str]], raw_dir: Path, title: str = "") -> Figure:
    """Columns of (top image, bottom image, caption) pairs."""
    fig, axes = plt.subplots(2, len(pairs), figsize=(2.3 * len(pairs), 5), squeeze=False)
    for k, (top, bottom, caption) in enumerate(pairs):
        for row, rel in enumerate((top, bottom)):
            ax = axes[row, k]
            with Image.open(raw_dir / rel) as img:
                ax.imshow(img.convert("L"), cmap="gray")
            ax.set_title(
                rel.replace("Training/", "Train/").replace("Testing/", "Test/"), fontsize=7
            )
            ax.axis("off")
        axes[1, k].text(
            0.5, -0.12, caption, transform=axes[1, k].transAxes, ha="center", fontsize=9
        )
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def confusion_matrix_figure(cm: np.ndarray, labels: Sequence[str], title: str = "") -> Figure:
    """Counts with row-normalized recall in brackets."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if norm[i, j] > 0.6 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.0%})", ha="center", va="center", color=color)
    ax.set_xticks(range(len(labels)), labels, rotation=20)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix {title}".strip())
    fig.tight_layout()
    return fig


def misclassified_grid(errors: pd.DataFrame, raw_dir: Path, max_images: int = 16) -> Figure:
    """Misclassified images with true label, prediction and confidence."""
    shown = errors.sort_values("confidence", ascending=False).head(max_images)
    cols = 4
    rows = max(1, int(np.ceil(len(shown) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, row) in zip(axes.flat, shown.iterrows(), strict=False):
        with Image.open(raw_dir / row["path"]) as img:
            ax.imshow(img.convert("L"), cmap="gray")
        ax.set_title(
            f"true {row['label']}\npred {row['pred']} ({row['confidence']:.0%})", fontsize=9
        )
    title = "Most confident mistakes" if len(shown) else "No misclassified images"
    fig.suptitle(title)
    fig.tight_layout()
    return fig
