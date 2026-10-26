"""Export README figures and the results table from MLflow into docs/.

    python -m neuroscan.reports.figures

Writes docs/figures/model_comparison.png, the champion's learning curves,
confusion matrix and most confident mistakes, and docs/results.md (comparison
table and how the selection rule decided). Everything comes from tracked runs,
so rerunning after new evaluations refreshes the docs.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402

from neuroscan.tracking import REGISTERED_MODEL, get_client  # noqa: E402
from neuroscan.tracking.select import (  # noqa: E402
    BASELINE_MODEL,
    choose,
    comparison_table,
    load_candidates,
)

LABELS = {
    "baseline_cnn": "Baseline CNN",
    "cnn_gap": "CNN + GAP",
    "efficientnet_b0": "EfficientNetB0",
    "resnet50": "ResNet50",
    "efficientnet_b0+shortcut_reduced": "EfficientNetB0, shortcut-reduced",
}
CHAMPION_ARTIFACTS = {
    "figures/learning_curves.png": "champion_learning_curves.png",
    "evaluation/test/confusion_matrix.png": "champion_confusion_matrix.png",
    "evaluation/test/misclassified.png": "champion_misclassified.png",
    "calibration/reliability_test.png": "champion_reliability.png",
    "explain/test/gradcam_grid.png": "champion_gradcam.png",
    "ood/score_histogram.png": "champion_ood_scores.png",
}


def save_compact(src: Path, dst: Path, max_bytes: int = 600_000) -> None:
    """Copy a PNG, reducing it to a 256-color palette if it's large (photos + heatmaps)."""
    shutil.copy(src, dst)
    if dst.stat().st_size > max_bytes:
        with Image.open(src) as img:
            img.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(
                dst, optimize=True
            )


def comparison_figure(df: pd.DataFrame, champion: str) -> Figure:
    """Test macro F1 with 95% intervals, shortcut gap and CPU latency per model."""
    order = df.sort_values("test_f1").index
    labels = [LABELS.get(m, m) + (" ★" if m == champion else "") for m in order]
    colors = ["#2b6cb0" if m == champion else "#a0aec0" for m in order]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)

    f1 = df.loc[order]
    err = [f1["test_f1"] - f1["test_f1_low"], f1["test_f1_high"] - f1["test_f1"]]
    axes[0].barh(labels, f1["test_f1"], xerr=err, color=colors, capsize=3)
    axes[0].set_xlim(0.8, 1.0)
    axes[0].set_title("Test macro F1 (95% CI)")

    axes[1].barh(labels, f1["test_shortcut_gap"] * 100, color=colors)
    axes[1].set_title("Shortcut gap (points, lower is better)")

    axes[2].barh(labels, f1["latency"], color=colors)
    axes[2].set_title("CPU latency p50 (ms)")
    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Four models on the duplicate-aware split (each duplicate group counted once)")
    fig.tight_layout()
    return fig


def calibration_table(client: Any, df: pd.DataFrame) -> list[str]:
    """Temperature and test ECE/NLL before and after scaling, per model (if calibrated)."""
    rows = [
        "| Model | Temperature | Test ECE before → after | Test NLL before → after |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model, run_id in df["run_id"].items():
        m = client.get_run(run_id).data.metrics
        if "calibration.temperature" not in m:
            continue
        rows.append(
            f"| {LABELS.get(str(model), str(model))} | {m['calibration.temperature']:.2f} "
            f"| {m['calibration.test.ece_before']:.3f} → {m['calibration.test.ece_after']:.3f} "
            f"| {m['calibration.test.nll_before']:.3f} → {m['calibration.test.nll_after']:.3f} |"
        )
    return rows if len(rows) > 2 else []


def explain_lines(client: Any, run_id: str) -> list[str]:
    m = client.get_run(run_id).data.metrics
    if "explain.test.peak_in_brain" not in m:
        return []
    return [
        "## Where the champion looks (Grad-CAM, test split)",
        "",
        f"- Heatmap peak inside the brain mask: {m['explain.test.peak_in_brain']:.1%} "
        f"(chance, i.e. mask area: {m['explain.test.mask_area']:.1%}).",
        f"- Share of heatmap mass inside the brain mask: {m['explain.test.mass_in_brain']:.1%}.",
        "- Peak inside the brain for wrong predictions: "
        f"{m['explain.test.wrong.peak_in_brain']:.1%}.",
        "",
        "A sanity check, not proof of correct reasoning: the mask is a simple intensity threshold.",
        "",
        "![Grad-CAM](figures/champion_gradcam.png)",
        "",
    ]


def ood_lines(client: Any, run_id: str) -> list[str]:
    m = client.get_run(run_id).data.metrics
    if "ood.threshold" not in m:
        return []
    sources = sorted(
        {k.split(".")[1] for k in m if k.startswith("ood.") and k.endswith(".rejected")}
        - {"val", "test", "all"}
    )
    lines = [
        "## Refusing what the model shouldn't answer (test split)",
        "",
        f"- Real test MRIs rejected as out of distribution: {m['ood.test.mri_rejected']:.1%} "
        "(threshold set so about 2% of validation MRIs are rejected).",
        f"- Flagged uncertain (calibrated confidence < {m['ood.uncertain_threshold']:.2f}): "
        f"{m['ood.test.uncertain']:.1%}; accuracy on the answered rest: "
        f"{m['ood.test.accuracy_when_answered']:.1%}.",
        "",
        "| OOD source | Rejected | AUROC (Mahalanobis / max softmax) "
        "| FPR at 95% TPR (Mahalanobis / max softmax) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for s in sources:
        maha, msp = f"ood.{s}.mahalanobis", f"ood.{s}.msp"
        lines.append(
            f"| {s.replace('_', ' ')} | {m[f'ood.{s}.rejected']:.1%} "
            f"| {m[f'{maha}.auroc']:.3f} / {m[f'{msp}.auroc']:.3f} "
            f"| {m[f'{maha}.fpr_at_95tpr']:.1%} / {m[f'{msp}.fpr_at_95tpr']:.1%} |"
        )
    lines += ["", "![OOD scores](figures/champion_ood_scores.png)", ""]
    return lines


def results_markdown(
    df: pd.DataFrame, steps: list[str], champion: str, extra: list[str] | None = None
) -> str:
    return "\n".join(
        [
            "# Model comparison",
            "",
            "Generated by `python -m neuroscan.reports.figures` from the tracked MLflow runs.",
            "All four models use `splits/v1.csv`; metrics count each duplicate group once.",
            "",
            comparison_table(df),
            "",
            "## Selection ([rule](model_selection.md), validation metrics only)",
            "",
            *steps,
            "",
            f"**Champion:** {LABELS.get(champion, champion)}. "
            f"**Baseline:** {LABELS[BASELINE_MODEL]}.",
            "",
            "![Model comparison](figures/model_comparison.png)",
            "",
            *(extra or []),
        ]
    )


def export(client: Any, docs: Path) -> str:
    df = load_candidates(client)
    decision = choose(df)
    figures = docs / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig = comparison_figure(df, decision.winner)
    fig.savefig(figures / "model_comparison.png", dpi=130)
    plt.close(fig)

    champion_run = df.loc[decision.winner, "run_id"]
    available = {
        a.path
        for folder in {Path(k).parent.as_posix() for k in CHAMPION_ARTIFACTS}
        for a in client.list_artifacts(champion_run, folder)
    }
    with tempfile.TemporaryDirectory() as tmp:
        for artifact, name in CHAMPION_ARTIFACTS.items():
            if artifact not in available:
                continue
            local = client.download_artifacts(champion_run, artifact, tmp)
            save_compact(Path(local), figures / name)

    extra = []
    calibration = calibration_table(client, df)
    if calibration:
        extra += [
            "## Calibration (temperature fitted on validation)",
            "",
            *calibration,
            "",
            "The champion is served with its temperature applied.",
            "",
            "![Reliability diagram](figures/champion_reliability.png)",
            "",
        ]
    extra += explain_lines(client, str(champion_run))
    extra += ood_lines(client, str(champion_run))
    (docs / "results.md").write_text(results_markdown(df, decision.steps, decision.winner, extra))
    return decision.winner


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args()
    client = get_client()
    winner = export(client, args.docs)
    alias = client.get_model_version_by_alias(REGISTERED_MODEL, "champion")
    print(f"Champion by rule: {winner}; registry champion is version {alias.version}")
    print(f"Wrote {args.docs / 'results.md'} and figures in {args.docs / 'figures'}")


if __name__ == "__main__":
    main()
