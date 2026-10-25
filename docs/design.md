# Design decisions

This document records the decisions behind NeuroScan: what problem each one solved,
what was chosen, what else was considered, and what it cost. It is meant to be read
alongside the code; every number comes from a tracked MLflow run or a committed file.

The project started as a Kaggle notebook: a 4-block CNN with a reported 98.32% test
accuracy, wrapped in a Flask page. The goal of the rework was not a higher number but a
number that can be defended, and a system that behaves sensibly when its input is not
what it expects.

Contents

1. [Guiding principles](#1-guiding-principles)
2. [Project structure and tooling](#2-project-structure-and-tooling)
3. [Data](#3-data)
4. [Evaluation methodology](#4-evaluation-methodology)
5. [Experiment tracking and training workflow](#5-experiment-tracking-and-training-workflow)
6. [Models and training](#6-models-and-training)
7. [Model selection](#7-model-selection)
8. [Calibration](#8-calibration)
9. [Explainability](#9-explainability)
10. [Refusing inputs: out-of-distribution and uncertainty](#10-refusing-inputs-out-of-distribution-and-uncertainty)
11. [Serving](#11-serving)
12. [Security and privacy](#12-security-and-privacy)
13. [Containerization](#13-containerization)
14. [Performance](#14-performance)
15. [Testing and CI](#15-testing-and-ci)
16. [What is deliberately not done](#16-what-is-deliberately-not-done)
17. [Open problems and next steps](#17-open-problems-and-next-steps)

---

## 1. Guiding principles

- **Distrust good numbers.** A 98% score on a public medical dataset is a reason to look
  for leakage and shortcuts, not to celebrate. Every result is reported next to the check
  that could have invalidated it.
- **Decide before looking.** The split, the metrics and the model-selection rule were
  fixed and committed before any candidate model was trained.
- **Track everything.** Every training run, including the baseline and the original model,
  is an MLflow run with its config, code version, data version and artifacts.
- **Fail safely.** A medical-imaging demo must be able to say "I don't know" and "this is
  not something I was built for". Refusal is a feature, measured like accuracy.
- **Reproducible from a clean machine.** One command per step, pinned dependencies, no
  paths that only exist on one computer.

---

## 2. Project structure and tooling

### 2.1 From notebook to package

**Context.** All training, evaluation and plotting lived in one Kaggle notebook with
`/kaggle/input` paths. Nothing could be tested or rerun outside Kaggle.

**Decision.** An installable package, `src/neuroscan/`, with one module per job
(`data/`, `models/`, `train.py`, `evaluate.py`, `calibrate.py`, `explain.py`, `ood.py`,
`optimize.py`, `tracking/`, `serving/`, `reports/`). Every step is a command
(`python -m neuroscan.<module>`) and a `make` target. Configuration is YAML parsed into
dataclasses (`config.py`) that reject unknown keys, so a typo in a config fails loudly.

**Alternatives.** Keeping notebooks as the main interface (not testable, hidden state);
a heavier config framework such as Hydra (more power than a four-model project needs).

**The notebook stays** as `notebooks/01_data_exploration.ipynb`, for data exploration
only. It imports the package's own loaders and metadata code, so the exploration and the
pipeline cannot drift apart. Notebooks are committed without outputs (`nbstripout` in
pre-commit, checked in CI).

### 2.2 Tooling

| Tool | Why |
| --- | --- |
| `uv` + `uv.lock` | Fast, fully locked environments; the same lock drives local work, CI and Docker. |
| `ruff` | Linting and formatting in one tool, including security rules (`S`). |
| `mypy` | Type checks for `src/`. |
| `bandit` | Static security scan; zero findings is part of CI. |
| `pre-commit` | Runs whitespace, YAML, large-file, ruff, mypy and nbstripout checks before every commit. |
| `Makefile` | One documented entry point per step (`make data`, `make split`, `make eval`, ...). |

**Dependency groups.** MLflow is an optional `tracking` extra and the API stack a
`serving` extra. Training on Colab installs neither, which keeps that install small and
avoids version clashes with Colab's preinstalled packages.

**TensorFlow is a range (`>=2.18,<2.21`), not a pin**, so Colab can keep its
preinstalled build instead of downloading another one for every session. The lock file
still pins the exact local version (2.20.0).

### 2.3 Model files leave git

The original repository committed the model twice (`best_model.keras` via LFS and a zip
copy), about 190 MB. Models now live only in the MLflow registry. The files were removed
from the tree (not from history: rewriting published history would break existing
clones). The original model is kept locally, git-ignored, because it is re-evaluated as a
reference point (section 4.5).

---

## 3. Data

The full findings are in [data_audit.md](data_audit.md); this section covers the
decisions they led to.

### 3.1 Dataset version

The Kaggle dataset had been updated since the original notebook: 7,200 images (1,400 per
class in `Training`, 400 per class in `Testing`) instead of 7,023. All work uses the
current version; the original model's results are therefore not directly comparable to
its published number, and that is stated wherever the two appear together.

### 3.2 Download

`make data` uses the Kaggle API with paths from config, never `/kaggle/input`. The same
command runs locally and in Colab, where the token comes from Colab Secrets.

### 3.3 Exploring before modeling

The exploration notebook looks for problems a model could exploit, not only for class
balance. The key finding: **file and acquisition properties alone predict the class.**
A random forest on resolution, file size, format, brightness, contrast and background
share (no anatomy) reaches 79.4% accuracy (chance: 25%) and recognizes "no tumor" at
about 99% recall, because most "no tumor" images come from a different source (small,
often PNG, brighter) than the tumor classes (mostly 512x512 JPEG).

**Decision.** Measure this shortcut in every evaluation (section 4.4), report it as a
limitation, and try to reduce it in a dedicated retraining run (section 17).

### 3.4 Duplicate detection

**Context.** Public MRI datasets often contain the same slice several times, resized,
re-encoded or with altered contrast, and the Kaggle `Training`/`Testing` folders are not
guaranteed to be disjoint. A model evaluated on near-copies of its training images is
partly measuring memory.

**Decision.** Two images are linked if any of these holds, and links are merged
transitively into groups (`neuroscan.data.dedup`):

1. identical decoded pixels (MD5 of the grayscale pixels, so re-saved files match);
2. perceptual hash (pHash) Hamming distance <= 4;
3. rank correlation of the 64x64 grayscale images >= 0.96.

**How the thresholds were chosen.** By looking at example pairs at each distance, before
building the split:

- pHash catches resized and re-encoded copies, but misses copies whose contrast was
  changed. Raising its threshold to 6 started pairing different scans.
- Rank correlation is unaffected by brightness and contrast changes, because it compares
  the order of pixel intensities, not their values. Copies score 0.96 to 0.99, different
  patients typically 0.7 to 0.9. At 0.95 the links chain more than 600 images into one
  group; at 0.96 the largest group has 61 images, which matches what is visible.
- Images are reduced to 64x64 for the correlation: enough to recognize the same slice,
  insensitive to JPEG noise, and fast enough to compare all 26 million pairs in about
  40 seconds.

**Result.** 4,333 of 7,200 images belong to 1,162 duplicate groups. **883 of the 1,600
Kaggle `Testing` images have a near-duplicate in `Training`** (377 of the 400 "no tumor"
test images). 27 groups contain conflicting labels, mostly glioma versus meningioma.

### 3.5 Duplicates are grouped, not deleted

**Decision.** Duplicates stay in the data, but every group is kept within one split.

**Why not delete them.** Many "near-duplicates" are adjacent slices of the same scan,
which are genuinely different images; contrast-altered copies act as free augmentation in
training. What matters for honest evaluation is that no test image has a copy in
training, and grouping guarantees that.

**The consequence** for evaluation is handled in section 4.2: within a test set, a group
of near-identical images should not count as many independent results.

### 3.6 Pre-augmented files are excluded

203 meningioma files (`*-aug-*`) are flipped, rotated or zoomed copies of other images.
Pixel-based matching cannot reliably trace them to their source, so they could leak
across splits undetected. They are excluded; training applies its own augmentation anyway.

### 3.7 One fixed split

**Decision.** `splits/v1.csv`: 70/15/15 train/validation/test, built with
`StratifiedGroupKFold` (20 folds: 3 test, 3 validation, 14 train, seed 42) so that classes
stay balanced and no duplicate group crosses splits. The Kaggle folders are ignored.

| Split | Images | Groups |
| --- | ---: | ---: |
| train | 4,897 | |
| validation | 1,050 | |
| test | 1,050 | |
| total | 6,997 | 3,891 |

The split file is committed, and its hash (`0816b6f44c96`) is recorded in every training
run as `split_version`. A test pins its per-class counts, so an accidental change fails CI.
The test split is used once per model, after training, and never for any choice.

### 3.8 Shared preprocessing

Every model takes raw RGB pixels in [0, 255] at 224x224 and scales them inside the
network (a `Rescaling` layer, EfficientNet's built-in normalization, or a ResNet
preprocessing layer). Preprocessing is therefore only decode, convert to 3 channels and
bilinear resize, written once in TensorFlow (`data/preprocess.py`) and used by training,
evaluation and the API. A parity test checks that the training pipeline and the API
produce identical tensors for PNG, JPEG, BMP, grayscale, RGBA and odd sizes.

---

## 4. Evaluation methodology

### 4.1 Metrics

**Decision.** Report per-class precision, recall and F1, **macro** averages, and the
clinically central pair: **tumor sensitivity** (share of tumor images, any type,
predicted as some tumor) and **tumor specificity**.

**Why.** The original notebook reported weighted precision, recall, F1 and an F2 derived
from them; with balanced classes, weighted recall equals accuracy, so four "98.32%"
figures were one number. Macro averages weigh each class equally. Tumor sensitivity
answers the question a clinician would ask first: how often is a tumor missed?

### 4.2 Each duplicate group counts once

**Context.** Within the test split, a group of ten near-identical images counts as ten
results. If the model gets that slice right, accuracy gains ten correct answers from what
is really one piece of evidence, and confidence intervals come out too narrow.

**Decision.** Every metric is computed twice: per image, and with each image weighted by
1 / (size of its group), so every group counts once. Intervals for the group-weighted
metrics use a **cluster bootstrap** that resamples whole groups. Headline numbers use the
group-weighted version. A test checks that the cluster bootstrap gives wider intervals
when duplicates repeat the same outcome.

### 4.3 Confidence intervals

95% bootstrap intervals (1,000 resamples) for accuracy, macro F1 and tumor sensitivity.
With about 1,000 test images, a difference of one or two points between models is often
within the noise; the intervals make that visible and feed the selection rule.

### 4.4 The shortcut check

**Decision.** Every evaluation trains the metadata-only probe from section 3.3 on the
training split and reports the model's accuracy separately on test images the probe gets
right and wrong. If the model relied only on anatomy, the two would be similar; a large
gap means it leans on the same source cues as the probe. This turns a dataset caveat into
a number per model.

### 4.5 The original model as a reference point

The original Kaggle model was logged to MLflow unchanged (`legacy-kaggle-model`) and
evaluated by the same script on the Kaggle `Testing` folder, both complete and restricted
to images with no duplicate in `Training`:

| Evaluation set | Images | Accuracy (95% CI) | Macro F1, groups counted once | Tumor sensitivity |
| --- | ---: | --- | --- | ---: |
| Published claim (earlier dataset version) | 1,311 | 98.3% | | |
| Kaggle `Testing`, current version | 1,497 | 94.1% (92.9–95.2) | 0.931 | 95.2% |
| `Testing` images with no duplicate in `Training` | 641 | 88.0% (85.3–90.5) | 0.778 (0.727–0.820) | 92.2% |

The drop from 94.1% to 88.0% is the duplicate effect: once the model cannot recognize
images it has seen, it misses about one tumor in thirteen and finds only 73% of gliomas.
These numbers are not comparable with the new models' test results (different test
sets); they show why the clean split was necessary.

### 4.6 Latency is measured on the compiled graph

The first latency measurements called the Keras model eagerly, which adds per-layer
Python overhead and made EfficientNetB0 look ten times slower than it is (326 ms instead
of 30 ms). Latency is now measured through `tf.function`, as a server runs a model.

---

## 5. Experiment tracking and training workflow

### 5.1 Local MLflow, training on Colab

**Context.** Training needs a GPU, which the development laptop does not have; tracking
needs to persist and be browsable, which a Colab session cannot offer.

**Decision.** MLflow runs only locally, with a SQLite backend and a local artifact folder
(`make mlflow` opens the UI). Colab never talks to MLflow: each training run writes a
self-contained **run bundle** to Google Drive, which is imported locally afterwards.

**Alternatives.** A hosted tracking server (an account, credentials in notebooks, and a
public endpoint for a personal project); logging to a file store on Drive (fragile
concurrent access, no registry).

### 5.2 The run bundle

```text
config.yaml     the exact config used
meta.json       model, git commit, split hash, start/end time, GPU, Python/TF/Keras versions
history.json    per-epoch loss, accuracy, validation macro F1, learning rate, stage
model.keras     the best checkpoint by validation macro F1
figures/        learning curves
```

`python -m neuroscan.tracking.import_run <bundle>` turns it into a regular MLflow run with
the original start and end times, every config value as a parameter, per-epoch metrics,
tags (`git_commit`, `split_version`, `trained_on`, ...) and the model registered as a new
version of `neuroscan-classifier`. Importing the same bundle twice is refused unless
`--force` replaces the earlier import.

### 5.3 Getting exact code into Colab without publishing it

**Context.** The work happens on a branch that is not pushed while in progress, so Colab
cannot `git clone` it.

**Decision.** `make colab-package` archives the current commit with `git archive` and
stamps the commit hash into the archive. It refuses to run with uncommitted changes to
packaged files, so every run points to an exact commit. The archive is uploaded to Drive
and installed in Colab. Inside Colab, the code reads its commit from the stamp, since
there is no `.git` folder.

### 5.4 Surviving Colab disconnects

- Checkpoints, Keras `BackupAndRestore` state and a CSV log are written to Drive during
  training; rerunning the same command resumes from the last finished epoch.
- If the code changed between attempts, the run records the commit that produced the
  final weights (`git_commit`) and keeps the first one (`first_git_commit`).
- If a session ends after the last epoch but before the bundle is written,
  `python -m neuroscan.train --finalize <run-dir>` builds the bundle from the checkpoints.
  This happened once (ResNet50).
- Each epoch's wall-clock end time is logged, so the run's end time does not depend on
  file times, which a zip downloaded from Drive shifts.

### 5.5 The pipeline was proven before any real training

A dry run (one epoch on 32 images) went through the whole path: Colab, bundle, import,
MLflow run and registered model version. Only then were models trained. The dry run and
its model version were deleted afterwards, so the real baseline became version 1.

---

## 6. Models and training

### 6.1 Four candidates

| Config | Model | Parameters | Role |
| --- | --- | ---: | --- |
| `baseline_cnn.yaml` | 4 conv blocks, Flatten → Dense(512) | 9.68M | The original architecture, retrained on the clean split |
| `cnn_gap.yaml` | Same conv blocks, GlobalAveragePooling → Dense(128) | 0.26M | Is the 9.4M-parameter dense layer needed? |
| `efficientnet_b0.yaml` | EfficientNetB0, ImageNet weights | 4.05M | Small transfer-learning model |
| `resnet50.yaml` | ResNet50, ImageNet weights | 23.6M | Large transfer-learning model |

Every builder exposes `build_model(cfg)` and a layer named `features` (the penultimate
representation), which the out-of-distribution detector uses later. The rebuilt baseline
has exactly the original model's 9,680,580 parameters.

### 6.2 Shared training rules

All four use the same split, image size (224), seed (42), batch size (32) and
augmentation (horizontal flip, rotation and zoom of ±10%, brightness ±15%). The baseline
keeps every hyperparameter of the original model: Adam at 0.001, up to 60 epochs, early
stopping patience 10 with the best weights restored, learning-rate reduction by half
after 7 stagnant epochs down to 1e-5. CNN + GAP uses the same settings, so only the head
differs.

**One deliberate change from the original:** early stopping, checkpointing and
learning-rate reduction monitor **validation macro F1** for all four models (the
original used validation accuracy and loss). A single rule for every model makes the
comparison cleaner; with balanced classes, macro F1 and accuracy rank epochs almost
identically.

### 6.3 Two-stage fine-tuning for the pretrained models

1. **Head only**, 10 epochs at 0.001, backbone frozen: the new classifier learns without
   destroying pretrained features.
2. **Fine-tuning** at 0.0001 (10x lower), early stopping patience 8, up to 40 epochs in
   total. Only the top of the backbone is unfrozen, chosen by block name rather than layer
   count (a count cut EfficientNet mid-block): EfficientNet from `block6a` (about 3.1M of
   4.1M parameters trainable), ResNet50 from `conv5` (about 15M of 23.6M).

BatchNormalization layers stay frozen and run in inference mode, the standard practice
for fine-tuning with small batches. ResNet's `preprocess_input` is implemented as a layer
inside the model, so every saved model takes the same raw [0, 255] input.

### 6.4 Results

Test split, each duplicate group counted once:

| Model | Test macro F1 (95% CI) | Tumor sensitivity | Shortcut gap | CPU latency |
| --- | --- | ---: | ---: | ---: |
| Baseline CNN | 0.954 (0.933–0.972) | 99.0% | +10.9 pt | 12.5 ms |
| CNN + GAP | 0.879 (0.847–0.906) | 97.3% | +22.6 pt | 10.0 ms |
| **EfficientNetB0** | 0.955 (0.936–0.972) | 99.0% | +10.5 pt | 30.1 ms |
| ResNet50 | 0.967 (0.952–0.981) | 99.8% | +4.5 pt | 90.7 ms |

Observations:

- **The three strong models cannot be separated on accuracy**: their intervals overlap.
- **The large dense layer was doing useful work**: removing it cost 7.5 points of macro F1
  with non-overlapping intervals, for little latency gain (the convolutions dominate).
- **Global average pooling doubled the shortcut gap**: it summarizes overall brightness
  and texture, exactly the cues the metadata probe uses.
- **ResNet50 leans on source cues the least** (4.5 points), relevant for the
  shortcut-reduction work in section 17.
- EfficientNet and ResNet reached the 40-epoch limit while validation F1 was still rising
  slightly. All models had the same budget, so the comparison is fair, but they may be
  slightly under-trained.

---

## 7. Model selection

**Decision.** The rule in [model_selection.md](model_selection.md) was committed before
any candidate was trained, and `python -m neuroscan.tracking.select` applies it to MLflow:

1. Rank by **validation** macro F1 (groups counted once).
2. Candidates whose 95% interval overlaps the leader's are tied.
3. Within the tie, drop any candidate more than 1 point below the best tumor sensitivity:
   missing a tumor is the costliest error.
4. Among the rest, take the lowest CPU latency; within 10%, the fewest parameters.

**Outcome.** ResNet50 led on validation, but EfficientNetB0 and the baseline were tied
with it. The baseline was dropped on sensitivity (98.6% versus 100%). EfficientNetB0 is
three times faster and six times smaller than ResNet50 at equal sensitivity, so it became
the **champion**. The registry aliases are `champion` (EfficientNetB0) and `baseline`
(the retrained original architecture).

**Why the rule matters more than the winner.** With overlapping intervals, any choice
made after seeing test results could be tuned to them. A rule fixed in advance, using
validation data only, makes the choice defensible; the test numbers are reported, never
used.

---

## 8. Calibration

**Context.** All four models were overconfident: the champion averaged 98.1% confidence
on test images while being right 96.0% of the time. A demo that shows probabilities must
mean them.

**Decision.** Temperature scaling (`neuroscan.calibrate`): one scalar T, fitted on the
validation split by minimizing negative log-likelihood; probabilities become
softmax(logits / T). It never changes the predicted class.

| Model | Temperature | Test ECE before → after | Test NLL before → after |
| --- | ---: | ---: | ---: |
| EfficientNetB0 (champion) | 2.02 | 0.024 → 0.011 | 0.138 → 0.111 |
| ResNet50 | 2.41 | 0.019 → 0.014 | 0.123 → 0.090 |
| Baseline CNN | 2.39 | 0.031 → 0.023 | 0.178 → 0.138 |
| CNN + GAP | 1.33 | 0.025 → 0.041 | 0.281 → 0.281 |

Expected calibration error halves for the champion. For CNN + GAP it gets worse: that
model was barely overconfident, and ECE on about 1,000 images is noisy; the result is
reported as measured.

**Packaging.** The Keras network and its temperature are registered together as one
MLflow `pyfunc` model, so anything that loads the champion gets calibrated probabilities
without knowing about temperatures. The same package later gained the OOD detector.

---

## 9. Explainability

**Decision.** Grad-CAM on the last feature map (`neuroscan.explain`), with a quantitative
sanity check instead of hand-picked examples: how often the heatmap's peak, and what
share of its mass, falls inside a simple brain mask (intensity threshold, largest
connected component, holes filled).

| Check (test split, 1,050 images) | Result | Chance level |
| --- | ---: | ---: |
| Heatmap peak inside the brain | 99.6% | 54% (mask area) |
| Heatmap mass inside the brain | 85.8% | 54% |
| Peak inside the brain, wrong predictions | 97.6% | |

In the example grid, correct predictions focus on the lesion, and **glioma/meningioma
confusions still look at the tumor**: the model finds the lesion but misjudges its type.
This is a sanity check, not proof of correct reasoning; the mask is a simple threshold
and a heatmap inside the brain can still be driven by the wrong features.

---

## 10. Refusing inputs: out-of-distribution and uncertainty

### 10.1 The problem

A classifier always picks one of its classes. Before this work, a photo of a cat would
have been labeled "glioma, 97%". The model must refuse inputs it was not built for and
flag cases it is unsure about.

### 10.2 Four statuses

Every prediction carries a status:

| Status | Meaning | Decided by |
| --- | --- | --- |
| `rejected_input` | Not a plausible MRI slice file | Cheap input checks |
| `rejected_ood` | Unlike every training class | Mahalanobis detector |
| `uncertain` | In distribution, but not confident enough | Calibrated confidence threshold |
| `ok` | Answered | |

### 10.3 Input checks

Before the model: the file must decode as an image, be 64 to 8,000 pixels per side, and
be grayscale. "Grayscale" is defined from the data: no training image has more than 4.4%
of pixels whose RGB channels differ by over 30, so images above 10% are rejected. These
checks cost nothing and stop 83% of natural photos on their own.

### 10.4 Mahalanobis detector

**Decision.** A training-free detector on the champion's penultimate features: class
means and one shared covariance from the training images, with Ledoit-Wolf shrinkage
(the plain covariance of 1,280-dimensional features from a few thousand images is
unstable). An image's score is its smallest Mahalanobis distance to a class mean.

**Threshold.** The 98th percentile of validation MRI scores: about 2% of real MRIs are
refused, a known and accepted cost. **No OOD image is used to choose it**; tuning it on
the OOD test images would be the same mistake as training on the test set.

**Evaluation.** An OOD test set from public datasets, 300 images per source (188 for chest
X-rays): natural photos, chest X-rays, CT slices and lumbar spine MRI (MRI, but the wrong
anatomy). Maximum softmax probability (after temperature scaling) is the reference
detector.

| Source | Refused | AUROC, Mahalanobis / max softmax | FPR at 95% TPR, Mahalanobis / max softmax |
| --- | ---: | --- | --- |
| Natural photos | 100% | 1.000 / 0.965 | 0.0% / 25.7% |
| Chest X-ray | 100% | 1.000 / 0.961 | 0.0% / 31.9% |
| Spine MRI | 100% | 0.999 / 0.866 | 0.0% / 80.3% |
| CT | 94.3% | 0.994 / 0.947 | 1.3% / 39.0% |
| All | | 0.998 / 0.932 | 0.4% / 45.5% |

Using the classifier's own confidence would let 80% of spine MRIs through at a setting
that keeps 95% of real brain MRIs; the feature-space detector lets none through. CT is the
hardest case (17 of 300 accepted): cross-sectional, grayscale and close to MRI in feature
space.

**What "out of distribution" means here.** The detector refuses images unlike the
training data, which is not the same as "not a brain MRI". A genuine sagittal brain MRI
with text overlays and much more neck than the training images was refused (score 4,517
versus a threshold of 3,918). Scans from other hospitals or scanners may be refused more
often; this is listed as a limitation.

### 10.5 The uncertain threshold

Chosen on validation: the lowest calibrated confidence (0.86) at which accuracy on the
predictions above it reaches 99%. On the test split, with all rules applied:

| | |
| --- | ---: |
| Real MRIs refused as out of distribution | 2.2% |
| Flagged uncertain | 8.7% |
| Accuracy on the answered rest | 98.8% (96.0% when answering everything) |

### 10.6 One package

The network, temperature, detector parameters and both thresholds are registered as one
MLflow model version (v7, with alias `champion`). Moving the alias releases or rolls back
all of them together.

---

## 11. Serving

### 11.1 One FastAPI service, one page

**Decision.** A FastAPI app (`serving/api.py`) serves the JSON API and a single static
page. The original design had a separate Flask site calling the API; for a demo, one
process with no inter-service traffic is simpler and has fewer moving parts.

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/predict` | One image: status, class, calibrated probabilities, OOD score, the 224x224 input and a Grad-CAM overlay |
| `POST /v1/predict/batch` | Up to 16 images, no heatmaps |
| `GET /v1/info` | Served model version, thresholds and its headline test metrics |
| `GET /health`, `GET /ready` | Liveness; readiness only once the model is loaded (503 before) |
| `GET /metrics` | Prometheus-format counters (section 14.3) |
| `GET /docs` | OpenAPI documentation |

### 11.2 The model comes from the registry

The service loads `models:/neuroscan-classifier@champion` at startup (configurable with
`NEUROSCAN_MODEL_URI`). Releasing a model means moving the alias in MLflow; no image
rebuild, no file copying. Every response carries the model version that produced it.

The model is copied to a temporary folder before loading: MLflow writes a metadata file
next to the artifacts when loading in place, which fails on a read-only registry mount.

### 11.3 The demo page

A single HTML page with its own CSS and JavaScript and no external assets: drag-and-drop,
paste or one of five held-out test images (one per class plus a real "hard case" that the
model sends to review). It shows the status, calibrated probabilities, the Grad-CAM
overlay with an opacity control, and the OOD distance against its threshold. It follows
the system's light or dark theme, works at phone width, and the research-only notice is
always visible. Status explanations are written for non-experts ("This image doesn't look
like the brain MRIs the model was trained on, so it won't guess").

![Demo page](figures/ui_dark.png)

---

## 12. Security and privacy

The original web app had a hard-coded secret key, used the uploaded file name directly
in the save path (path traversal), kept every upload forever in a publicly served folder,
ran with the Werkzeug debugger enabled (remote code execution if exposed), had no upload
limit and returned raw exception text. All of these are gone:

| Concern | Now |
| --- | --- |
| Upload storage | Processed in memory, never written to disk or logs |
| File names | Never used |
| Upload size | 10 MB limit (`413` beyond), configurable |
| File validation | Must decode as an image of sensible size (`400` otherwise); extensions are ignored |
| Errors | Generic codes to clients (`not_an_image`, `internal_error`); details only in server logs |
| Debug mode | None; production server (uvicorn) |
| Browser security | Content-Security-Policy allowing only same-origin scripts and styles, no framing, `nosniff`, no referrer, no caching |
| Medical disclaimer | On every page and in every API response |
| Static analysis | `bandit` reports no issues; CI enforces it |

API tests cover a text file renamed to `.jpg` (400), an oversized upload (413), a
path-traversal file name (harmless) and the security headers.

---

## 13. Containerization

- **Multi-stage build** with `uv`: dependencies install in a build stage; the runtime
  image holds only Python and the virtual environment.
- **The model is not in the image.** The container mounts the MLflow registry
  **read-only** and serves the `champion` alias.
- **Hardened runtime**: non-root user (uid 10001), read-only root filesystem with a
  `tmpfs` for `/tmp`, all Linux capabilities dropped, `no-new-privileges`, and a health
  check on `/ready`. The container does not restart on its own.
- **Offline build.** Packages come from the host's uv cache and the build steps run with
  networking disabled, so building the image downloads no Python packages. This came from
  a practical constraint (limited bandwidth) and also makes builds reproducible. CI, which
  has no such cache, passes an empty cache and allows downloads.
- **Disk hygiene.** The build copies the package cache into BuildKit; `make docker-build`
  prunes the build cache afterwards. Without that, repeated builds filled the root disk.
- MLflow stores absolute artifact paths, so the registry is mounted at the same path
  inside the container. A hosted registry would remove this coupling.

---

## 14. Performance

Details in [performance.md](performance.md).

### 14.1 TFLite instead of ONNX

**Decision.** Benchmark TensorFlow Lite (float32 and INT8 weights) rather than ONNX
Runtime. TFLite ships with TensorFlow, so no new dependencies; the ONNX route needed about
35 MB of extra packages and a converter with uncertain support for TensorFlow 2.20 and
Keras 3.

| Runtime | Size | CPU p50 | Test accuracy |
| --- | ---: | ---: | ---: |
| Keras | 16.2 MB | 18.0 ms | 96.0% |
| TFLite float32 | 16.1 MB | 15.5 ms | 96.0% |
| TFLite INT8 weights | 4.6 MB | 16.5 ms | 95.8% |

TFLite float32 is the same model (max probability difference 5e-5) and 15% faster. INT8
weights make it 3.5x smaller for 0.2 points of accuracy, but not faster on this CPU, since
weight-only quantization dequantizes at run time. **The API keeps Keras**: Grad-CAM and
the OOD detector need gradients and intermediate features. Keras 3 stores optimizer state
even in weights-only files, so Keras sizes count the model's variables directly.

### 14.2 The load test found a real bottleneck

A small load tester (`serving/loadtest.py`, built on the already-present `httpx` instead
of adding Locust) showed about 4 requests per second whatever the concurrency, with a
single-image latency of 260 ms against a model that needs 18 ms. The OOD feature
extraction and Grad-CAM were running eagerly, layer by layer. Compiled with
`tf.function`:

| Requests in parallel | Grad-CAM | Before: req/s, p95 | After: req/s, p95 |
| ---: | --- | ---: | ---: |
| 1 | no | 3.8, 284 ms | 10.9, 111 ms |
| 4 | no | 4.3, 990 ms | 21.4, 218 ms |
| 1 | yes | 1.1, 1,146 ms | 7.3, 171 ms |

Inference also moved to a worker thread, so the event loop keeps answering health checks
during a prediction.

### 14.3 Monitoring

`/metrics` exposes predictions by status and class, and latency and confidence
histograms, in Prometheus text format (written by hand, no new dependency). With
`NEUROSCAN_PREDICTION_LOG` set, the API appends one JSON line per prediction (status,
class, confidence, OOD score, latency, model version; never the image or file name), and
`python -m neuroscan.reports.monitoring` draws a dashboard from it. A rising rejection rate
or falling confidence is the cheapest early warning that incoming images differ from the
training data.

---

## 15. Testing and CI

84 tests run in CI; 8 behavioral tests run locally against the real champion.

| Area | What is checked |
| --- | --- |
| Data | Planted duplicates are found (exact, contrast-changed, re-encoded); distinct images are not grouped; no group crosses splits; the committed split's counts are pinned |
| Preprocessing | Shape, dtype and range for grayscale, RGBA and odd sizes; non-images raise |
| Train/serve parity | The training pipeline and the API produce identical tensors |
| Models | Each builder's shapes, the GAP variant's size, which layers fine-tuning unfreezes, the exact ImageNet weight file each builder requests (without network) |
| Training | A full run writes a complete bundle; resume after a simulated disconnect continues the same run; finalize after a lost session; commit change on resume is recorded |
| Tracking | Import into a temporary MLflow store: params, metrics, tags, artifacts, registered version; re-import refused |
| Evaluation | Metrics by hand, macro versus weighted, group weighting, cluster bootstrap width, evaluation sets |
| Calibration | Temperature recovers known overconfidence, never changes predictions, is fitted on validation only; packaged model round trip |
| OOD | Detector separates far points, save/load, thresholds, input checks, every status of the packaged model |
| API | Statuses, 400/413/503, path traversal, batch, security headers, metrics and outcome log |
| Behavioral (local) | Demo images keep their answers; stable under ±10% brightness; real test MRIs accepted and correct; photos, X-rays, CT and spine MRI refused |

**GitHub Actions** (`.github/workflows/ci.yml`) runs ruff, mypy, bandit, a check that
notebooks have no outputs, and pytest on every pull request and push to `main`, and
builds the Docker image and checks it runs as a non-root user. `make ci` runs the same
checks locally.

---

## 16. What is deliberately not done

- **No public demo.** The service runs locally or in Docker. A public medical demo invites
  misuse and needs abuse protection that is out of scope.
- **No patient-level evaluation.** The dataset has no patient IDs; duplicate grouping
  approximates it but cannot catch every slice of the same patient.
- **No clinical claims.** No "state-of-the-art", "production-ready" or "clinical
  integration" language; the model is a research and education artifact.
- **No history rewrite** to remove the old model files from git.

---

## 17. Open problems and next steps

- **Reduce the source shortcut.** Crop each image to the head and normalize intensity per
  image, so resolution, framing, brightness and contrast carry less information about the
  source; re-run the metadata probe on the preprocessed images, retrain the champion
  architecture on the same split, compare the shortcut gap before and after, and apply the
  same selection rule again. ResNet50's smaller gap makes it a candidate to revisit.
- **Train longer.** The pretrained models hit their epoch limit while improving.
- **External validation.** Test on brain MRIs from another source to measure how the
  model and the OOD detector behave under a real distribution shift.
- **Label noise.** 27 duplicate groups carry conflicting labels; a review of these would
  improve both training and the test set.
