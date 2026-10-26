"""The model-selection rule on made-up validation numbers."""

import pandas as pd

from neuroscan.tracking.select import choose, comparison_table


def candidates(**rows):
    cols = ["f1", "f1_low", "f1_high", "sensitivity", "latency", "params"]
    return pd.DataFrame.from_dict(
        {k: dict(zip(cols, v, strict=True)) for k, v in rows.items()}, orient="index"
    )


def test_clear_winner_is_chosen():
    df = candidates(
        baseline_cnn=(0.80, 0.77, 0.83, 0.93, 40, 9.7e6),
        efficientnet_b0=(0.92, 0.90, 0.94, 0.97, 30, 4.0e6),
    )
    assert choose(df).winner == "efficientnet_b0"


def test_tie_goes_to_the_faster_model():
    df = candidates(
        resnet50=(0.93, 0.91, 0.95, 0.97, 120, 23.6e6),
        efficientnet_b0=(0.92, 0.90, 0.94, 0.97, 30, 4.0e6),
    )
    assert choose(df).winner == "efficientnet_b0"


def test_tie_never_trades_away_tumor_sensitivity():
    df = candidates(
        resnet50=(0.93, 0.91, 0.95, 0.98, 120, 23.6e6),
        cnn_gap=(0.92, 0.90, 0.94, 0.95, 10, 0.26e6),  # faster, but 3 points less sensitive
    )
    decision = choose(df)
    assert decision.winner == "resnet50"
    assert "dropped cnn_gap" in decision.steps[2]


def test_equal_latency_goes_to_fewer_parameters():
    df = candidates(
        baseline_cnn=(0.90, 0.88, 0.92, 0.96, 31, 9.7e6),
        cnn_gap=(0.90, 0.88, 0.92, 0.96, 30, 0.26e6),
        resnet50=(0.91, 0.89, 0.93, 0.96, 29, 23.6e6),
    )
    assert choose(df).winner == "cnn_gap"


def test_table_lists_baseline_first():
    df = candidates(
        efficientnet_b0=(0.92, 0.90, 0.94, 0.97, 30, 4.0e6),
        baseline_cnn=(0.80, 0.77, 0.83, 0.93, 40, 9.7e6),
    )
    for col in (
        "test_f1",
        "test_f1_low",
        "test_f1_high",
        "test_accuracy",
        "test_sensitivity",
        "test_specificity",
        "test_shortcut_gap",
    ):
        df[col] = 0.5
    rows = comparison_table(df).splitlines()
    assert rows[2].startswith("| baseline_cnn")
    assert rows[3].startswith("| efficientnet_b0")
