"""Compare the trained models and pick the champion with the rule in docs/model_selection.md.

    python -m neuroscan.tracking.select            # print the comparison and the decision
    python -m neuroscan.tracking.select --apply    # also set the champion/baseline aliases

Every candidate must have been evaluated on val and test first
(``python -m neuroscan.evaluate --run-id <id> --eval-set val test``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import pandas as pd

from neuroscan.tracking import EXPERIMENT, REGISTERED_MODEL, get_client

CANDIDATE_MODELS = ("baseline_cnn", "cnn_gap", "efficientnet_b0", "resnet50")
BASELINE_MODEL = "baseline_cnn"
SENSITIVITY_MARGIN = 0.01  # 1 percentage point
LATENCY_TOLERANCE = 0.10  # within 10% counts as equally fast

SELECTION_METRICS = {
    "f1": "val.grouped.macro_f1",
    "f1_low": "val.grouped.macro_f1_ci_low",
    "f1_high": "val.grouped.macro_f1_ci_high",
    "sensitivity": "val.grouped.tumor_sensitivity",
    "latency": "val.latency_cpu_p50_ms",
}
REPORT_METRICS = {
    "test_f1": "test.grouped.macro_f1",
    "test_f1_low": "test.grouped.macro_f1_ci_low",
    "test_f1_high": "test.grouped.macro_f1_ci_high",
    "test_accuracy": "test.grouped.accuracy",
    "test_sensitivity": "test.grouped.tumor_sensitivity",
    "test_specificity": "test.grouped.tumor_specificity",
    "test_shortcut_gap": "test.shortcut.accuracy_gap",
}


@dataclass
class Decision:
    winner: str
    steps: list[str]


def choose(candidates: pd.DataFrame) -> Decision:
    """Apply the selection rule. ``candidates`` is indexed by model name with the
    columns of SELECTION_METRICS plus ``params``."""
    steps = []
    ranked = candidates.sort_values("f1", ascending=False)
    leader = ranked.iloc[0]
    steps.append(
        "1. Ranked by validation macro F1: "
        + ", ".join(f"{m} {r.f1:.4f}" for m, r in ranked.iterrows())
        + f". Leader: {leader.name}."
    )

    tied = ranked[ranked["f1_high"] >= leader["f1_low"]]
    steps.append(
        f"2. 95% intervals overlapping the leader's ({leader.f1_low:.4f}-{leader.f1_high:.4f}): "
        + ", ".join(tied.index)
        + "."
    )

    best_sens = tied["sensitivity"].max()
    safe = tied[tied["sensitivity"] >= best_sens - SENSITIVITY_MARGIN]
    dropped = sorted(set(tied.index) - set(safe.index))
    steps.append(
        f"3. Best tumor sensitivity in the tie {best_sens:.4f}; kept within 1 point: "
        + ", ".join(safe.index)
        + (f" (dropped {', '.join(dropped)})." if dropped else ".")
    )

    fastest = safe["latency"].min()
    quick = safe[safe["latency"] <= fastest * (1 + LATENCY_TOLERANCE)]
    winner = str(quick.sort_values(["params", "latency"]).index[0])
    steps.append(
        "4. CPU latency p50: "
        + ", ".join(f"{m} {r.latency:.1f} ms" for m, r in safe.iterrows())
        + f". Within 10% of the fastest: {', '.join(quick.index)}; "
        + f"fewest parameters among them: {winner}."
    )
    return Decision(winner, steps)


def load_candidates(client: Any) -> pd.DataFrame:
    exp = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.dry_run = 'false' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )
    rows = {}
    for run in runs:
        model = run.data.tags.get("model")
        preprocessing = run.data.params.get("data.preprocessing", "standard")
        name = model if preprocessing == "standard" else f"{model}+{preprocessing}"
        # Latest run per architecture and preprocessing.
        if model in CANDIDATE_MODELS and name not in rows:
            metrics = run.data.metrics
            row: dict[str, Any] = {"run_id": run.info.run_id, "run_name": run.info.run_name}
            row["params"] = int(run.data.params.get("result.params", 0))
            for col, key in {**SELECTION_METRICS, **REPORT_METRICS}.items():
                row[col] = metrics.get(key)
            rows[name] = row
    df = pd.DataFrame.from_dict(rows, orient="index")
    if df.empty:
        raise LookupError("No finished, non-dry-run candidate runs in MLflow")
    missing = df[df[list(SELECTION_METRICS)].isna().any(axis=1)]
    if len(missing):
        raise LookupError(
            "Evaluate these runs on val and test first: "
            + ", ".join(f"{m} ({r.run_id})" for m, r in missing.iterrows())
        )
    return df


def comparison_table(df: pd.DataFrame) -> str:
    """Markdown table, baseline first, then by validation macro F1."""
    order = [BASELINE_MODEL] if BASELINE_MODEL in df.index else []
    order += [m for m in df.sort_values("f1", ascending=False).index if m != BASELINE_MODEL]
    lines = [
        "| Model | Params | Val macro F1 (95% CI) | Test macro F1 (95% CI) | Test accuracy "
        "| Tumor sensitivity | Tumor specificity | Shortcut gap | CPU latency |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    def pct(x: Any) -> str:
        return "–" if x is None or pd.isna(x) else f"{x:.1%}"

    def ci(v: Any, lo: Any, hi: Any) -> str:
        return "–" if v is None or pd.isna(v) else f"{v:.3f} ({lo:.3f}–{hi:.3f})"

    for m in order:
        r = df.loc[m]
        gap: str = "–" if pd.isna(r.test_shortcut_gap) else f"{r.test_shortcut_gap * 100:+.1f} pt"
        lines.append(
            f"| {m} | {r.params / 1e6:.2f}M | {ci(r.f1, r.f1_low, r.f1_high)} "
            f"| {ci(r.test_f1, r.test_f1_low, r.test_f1_high)} | {pct(r.test_accuracy)} "
            f"| {pct(r.test_sensitivity)} | {pct(r.test_specificity)} | {gap} "
            f"| {r.latency:.1f} ms |"
        )
    return "\n".join(lines)


def set_aliases(client: Any, df: pd.DataFrame, winner: str) -> dict[str, str]:
    aliases = {"champion": winner}
    if BASELINE_MODEL in df.index:
        aliases["baseline"] = BASELINE_MODEL
    out = {}
    for alias, model in aliases.items():
        run_id = df.loc[model, "run_id"]
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL}' and run_id='{run_id}'")
        if not versions:
            raise LookupError(f"No registered version for {model} (run {run_id})")
        version = max(int(v.version) for v in versions)
        client.set_registered_model_alias(REGISTERED_MODEL, alias, str(version))
        out[alias] = f"{model} (version {version})"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--apply", action="store_true", help="set the registry aliases")
    args = parser.parse_args()

    client = get_client()
    df = load_candidates(client)
    absent = sorted(set(CANDIDATE_MODELS) - set(df.index))
    if absent:
        print(f"Note: no evaluated run yet for {', '.join(absent)}\n")
    print(comparison_table(df))
    decision = choose(df)
    print("\nSelection (docs/model_selection.md, validation metrics only):")
    for step in decision.steps:
        print(f"  {step}")
    print(f"\nChampion: {decision.winner}")
    if args.apply:
        for alias, target in set_aliases(client, df, decision.winner).items():
            print(f"Alias {alias!r} -> {target}")


if __name__ == "__main__":
    main()
