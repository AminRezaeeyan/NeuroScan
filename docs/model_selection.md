# Model selection rule

This rule was written and committed **before any of the compared models was trained**,
so the choice can't be tuned to the test results. `python -m neuroscan.tracking.select`
applies it to the runs in MLflow.

## Candidates

The four architectures, each trained once with its config in `configs/` on the same split
(`splits/v1.csv`), seed and augmentation:

| Config | Model |
| --- | --- |
| `baseline_cnn.yaml` | 4 conv blocks, Flatten → Dense(512) |
| `cnn_gap.yaml` | same conv blocks, GlobalAveragePooling → Dense(128) |
| `efficientnet_b0.yaml` | EfficientNetB0, ImageNet weights, fine-tuned in two stages |
| `resnet50.yaml` | ResNet50, ImageNet weights, fine-tuned in two stages |

For each model, the latest finished run that is not a dry run is used.

**Addition (after the first comparison, before any shortcut-reduced model was trained).**
A run trained with different preprocessing is a separate candidate, named
`<model>+<preprocessing>`; the first is `efficientnet_b0+shortcut_reduced`
(`configs/efficientnet_b0_shortcut_reduced.yaml`: the champion's recipe with head
cropping and per-image intensity normalization). The rule below is unchanged and is
applied to all candidates together.

## Rule

Only **validation** metrics are used. Test metrics are reported, never used to choose.
All metrics count each duplicate group once (`val.grouped.*`), because near-duplicates
are not independent evidence.

1. **Rank** the candidates by validation macro F1 (`val.grouped.macro_f1`). The top one is
   the leader.
2. **Statistical tie:** a candidate is tied with the leader when their 95% intervals for
   validation macro F1 overlap. The tied set always includes the leader.
3. **Safety first:** within the tied set, drop every candidate whose validation tumor
   sensitivity (`val.grouped.tumor_sensitivity`) is more than 1 percentage point below the
   best tumor sensitivity in the tied set. Missing a tumor is the costliest error.
4. **Cheapest wins:** among the remaining candidates, choose the lowest single-image CPU
   latency (`val.latency_cpu_p50_ms`); if two are within 10% of each other, the one with
   fewer parameters.

The winner gets the registry alias `champion`; the baseline CNN gets `baseline`.

## What this rule does not do

- It does not use the test split, the Kaggle Testing folder or the legacy model.
- It does not change after results are known. If it has to change, the change is a new
  commit with the reason, and both the old and new choices are reported.
