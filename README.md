# NeuroScan

[![CI](https://github.com/AminRezaeeyan/NeuroScan/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AminRezaeeyan/NeuroScan/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Brain tumor classification from 2D MRI slices (glioma, meningioma, pituitary tumor, no
tumor), built to produce numbers that hold up to scrutiny: a duplicate-aware evaluation,
tracked experiments, a model chosen by a rule fixed in advance, calibrated probabilities,
and a service that refuses inputs it wasn't built for.

> **Research demo, not a medical device.** NeuroScan must not be used for diagnosis or
> any decision about a patient.

> **Project history.** The foundation of NeuroScan (the CNN architecture, its training
> recipe and the first web app) was built in February–March 2025 and released as
> [v1.0.0](https://github.com/AminRezaeeyan/NeuroScan/releases/tag/v1.0.0) (2 March 2025)
> and [v1.1.0](https://github.com/AminRezaeeyan/NeuroScan/releases/tag/v1.1.0)
> (23 March 2025). In October 2026, about a year and a half later, I revisited it and
> re-engineered it into the system described here: an audited dataset and leakage-free
> evaluation, tracked experiments, a pre-registered model comparison, calibration,
> out-of-distribution refusal, a secured API and tests with CI. The original model is kept
> as a reference and [re-evaluated below](#the-original-model-measured-honestly); its code
> remains available under the v1 tags.

![NeuroScan demo page](docs/figures/ui_dark.png)

## Highlights

- **An honest test set.** 55% of the dataset's own test images have a near-duplicate in
  its training images. NeuroScan detects duplicates (including contrast-altered copies)
  and builds a split where no duplicate group crosses train, validation and test.
  The original model's 98.3% becomes 88.0% on test images it hasn't effectively seen.
- **Four models, one pre-registered rule.** A baseline CNN, a lighter variant,
  EfficientNetB0 and ResNet50, trained on Colab, tracked in MLflow and compared with 95%
  confidence intervals. The champion was chosen on validation data by a rule committed
  before training.
- **Calibrated, and able to say no.** Temperature scaling cuts calibration error by 40%. A
  Mahalanobis detector refuses photos, X-rays, CT and spine MRI (AUROC 0.999), and
  low-confidence cases are sent to review: the answered 94% of test scans are 98.2%
  accurate.
- **Measured and reduced shortcuts.** File and acquisition properties alone predict the
  class 79% of the time. Every evaluation reports how much a model leans on such cues;
  retraining with head cropping and per-image intensity normalization **cut the
  champion's shortcut gap from 10.5 to 6.2 points without losing accuracy**.
- **Tested on scans from other hospitals.** On 614 new images from Bangladeshi hospitals
  (PMRAM, after removing the 59% that duplicate images in the training dataset), the
  champion keeps **96.7% accuracy** (the previous champion: 91.9%) and answers 85% of
  scans at 98.8% accuracy.
- **Served safely.** FastAPI service and a single-page demo, loaded from the MLflow
  registry by alias, with in-memory uploads, strict security headers and a hardened,
  read-only Docker container.

## Results

All four models were trained with the same split, seed and augmentation, and are
evaluated on the same 1,050 held-out test images, with **each group of near-duplicates
counted once**.

| Model | Params | Test macro F1 (95% CI) | Tumor sensitivity | Shortcut gap | CPU latency |
| --- | ---: | --- | ---: | ---: | ---: |
| Baseline CNN (original architecture) | 9.7M | 0.954 (0.933–0.972) | 99.0% | +10.9 pt | 12.5 ms |
| CNN + global average pooling | 0.26M | 0.879 (0.847–0.906) | 97.3% | +22.6 pt | 10.0 ms |
| EfficientNetB0 | 4.1M | 0.955 (0.936–0.972) | 99.0% | +10.5 pt | 30.1 ms* |
| **EfficientNetB0, shortcut-reduced (champion)** | 4.1M | **0.960 (0.942–0.976)** | **99.6%** | **+6.2 pt** | 18.0 ms* |
| ResNet50 | 23.6M | 0.967 (0.952–0.981) | 99.8% | +4.5 pt | 90.7 ms |

*Shortcut gap: how much more accurate a model is on images whose class can be guessed
from metadata alone. Smaller is better. \*Same architecture; the latency difference is
measurement noise between sessions.*

- The shortcut-reduced EfficientNetB0 is the original EfficientNetB0 recipe with one
  change: each image is cropped to the head and its intensity normalized before resizing.
  It is as accurate as before and leans much less on image-source cues (validation gap
  +1.1 instead of +4.4 points, test +6.2 instead of +10.5).
- Three models are statistically tied with the leader, ResNet50 (overlapping intervals). The
  [selection rule](docs/model_selection.md), fixed before training, keeps tumor
  sensitivity within 1 point and prefers the faster, then smaller model. The champion is
  3x faster and 6x smaller than ResNet50.
- With the refusal rules on, the served model refuses 2.0% of real test MRIs as out of
  distribution, sends 3.6% to review as uncertain, and answers the rest with **98.2%
  accuracy**.

![Model comparison](docs/figures/model_comparison.png)

| Refusing inputs | Refused | AUROC, Mahalanobis / max softmax |
| --- | ---: | --- |
| Natural photos | 100% | 1.000 / 0.988 |
| Chest X-ray | 100% | 1.000 / 0.989 |
| Lumbar spine MRI | 100% | 0.999 / 0.969 |
| CT slices | 98.3% | 0.996 / 0.969 |

### External test: scans from other hospitals

[PMRAM](https://www.kaggle.com/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset)
(CC BY-SA 4.0) collects brain MRIs from hospitals in Bangladesh. Of its 1,505 original
images, **891 (59%) turned out to be copies of images in the public training dataset**,
including every pituitary image, so they were excluded with the same duplicate detector
used for the split. The remaining 614 images (glioma, meningioma, no tumor) are new to
every model.

| Model | Accuracy, forced to answer (95% CI) | Refused | Uncertain | Accuracy when answered |
| --- | --- | ---: | ---: | ---: |
| **EfficientNetB0, shortcut-reduced (champion)** | **96.7% (95.3–98.0)** | 6.8% | 8.6% | **98.8%** |
| EfficientNetB0 (previous champion) | 91.9% (89.7–93.8) | 11.4% | 16.9% | 96.4% |

Reducing reliance on source cues paid off where it matters: on new data the champion is
clearly more accurate, and refuses or defers less. Caveats: three classes only, 614
images, and the remaining images may still share origins with other public collections.

More: [results](docs/results.md) (all tables and figures, generated from MLflow),
[data audit](docs/data_audit.md), [model card](docs/model_card.md),
[performance](docs/performance.md).

### The original model, measured honestly

| Evaluation set | Accuracy (95% CI) | Macro F1 |
| --- | --- | --- |
| Published claim (earlier dataset version) | 98.3% | |
| Kaggle `Testing`, current version | 94.1% (92.9–95.2) | 0.939 |
| Kaggle `Testing` images with no duplicate in `Training` | **88.0% (85.3–90.5)** | **0.801** |

## How it works

```mermaid
flowchart LR
    A[Kaggle dataset] --> B[Duplicate audit<br/>pixels, pHash, rank correlation]
    B --> C[Fixed group-aware split<br/>splits/v1.csv]
    C --> D[Train on Colab<br/>run bundle on Drive]
    D --> E[Import into local MLflow]
    E --> F[Evaluate<br/>intervals, sensitivity, shortcut check]
    F --> G[Pre-registered selection<br/>@champion]
    G --> H[Calibrate + OOD detector<br/>one packaged model]
    H --> I[FastAPI + demo page<br/>Docker]
```

For each uploaded image the service returns one of four statuses:

| Status | When |
| --- | --- |
| `ok` | In distribution and confident: class + calibrated probabilities + Grad-CAM |
| `uncertain` | In distribution, calibrated confidence below 0.75: sent to review |
| `rejected_ood` | Features far from every training class (Mahalanobis distance) |
| `rejected_input` | Not a plausible MRI file: undecodable, odd size, or a color image |

Every decision and its alternatives are explained in **[docs/design.md](docs/design.md)**.

## Quick start

The trained model is not stored in git: it lives in a local MLflow registry. Running the
demo therefore needs the registry, which [docs/reproducing.md](docs/reproducing.md)
builds from scratch (training takes about four Colab sessions).

```bash
make install            # Python 3.12 + uv; installs everything and the git hooks
make data               # download the dataset (Kaggle token in ~/.kaggle/access_token)
make mlflow             # browse experiments on http://127.0.0.1:5000
make serve              # API + demo page on http://127.0.0.1:8000 (needs @champion)
make docker-up          # same, in a hardened container
```

### API

```bash
curl -F "file=@scan.jpg" http://127.0.0.1:8000/v1/predict
```

```json
{
  "model_version": "10",
  "status": "ok",
  "prediction": "meningioma",
  "confidence": 0.992,
  "probabilities": {"glioma": 0.005, "meningioma": 0.992, "notumor": 0.0, "pituitary": 0.002},
  "calibrated": true,
  "ood_score": 2371.5,
  "ood_threshold": 3864.5,
  "latency_ms": "…",
  "input_png_base64": "…",
  "heatmap_png_base64": "…",
  "disclaimer": "Research demo, not a medical device. Do not use for diagnosis."
}
```

| Endpoint | |
| --- | --- |
| `POST /v1/predict` | One image (`?heatmap=false` skips Grad-CAM) |
| `POST /v1/predict/batch` | Up to 16 images |
| `GET /v1/info` | Served model version, thresholds, headline metrics |
| `GET /health`, `GET /ready` | Liveness and readiness |
| `GET /metrics` | Prometheus metrics |
| `GET /docs` | Interactive API documentation |

Errors return `{"error": "<code>"}`: `not_an_image` / `bad_size` (400), `file_too_large`
(413), `model_not_loaded` (503).

## Repository layout

```text
src/neuroscan/
  config.py            YAML configs -> dataclasses
  data/                download, duplicate audit, splits, preprocessing, metadata probe
  models/              one builder per model
  train.py             config-driven training -> run bundle (resumable)
  evaluate.py          metrics, cluster-bootstrap intervals, shortcut check
  calibrate.py         temperature scaling
  explain.py           Grad-CAM + brain-mask check
  ood.py               Mahalanobis detector and thresholds
  shortcut.py          pixel-statistics probe of the model input, per preprocessing mode
  optimize.py          TFLite export and benchmark
  tracking/            run-bundle import, legacy model, model selection
  serving/             FastAPI app, demo page, input checks, packaged model, monitoring
  reports/             figures and dashboards
configs/               one YAML per model
splits/v1.csv          the fixed split (committed)
notebooks/             data exploration, Colab training notebook
tests/                 unit, integration, API and behavioral tests
docs/                  design, data audit, model card, results, performance, reproducing
```

## Testing and CI

```bash
make ci                 # ruff, mypy, bandit, pytest (89 tests)
make test-behavioral    # 8 tests against the real champion (needs the local registry)
```

GitHub Actions runs lint, type checks, the security scan, a notebook-output check and the
test suite on every pull request, and builds the Docker image. Tests cover duplicate
detection, split integrity, train/serve parity, training resume, MLflow import,
calibration, OOD rejection, and API security cases (renamed text files, oversized
uploads, path-traversal file names, security headers).

## Limitations

- **Image-level evaluation; limited external validation.** The dataset has no patient
  IDs, so different slices of one patient can sit in different splits. The external test
  covers 614 images and three classes from one other source.
- **Source shortcuts.** Image source partly reveals the class (a metadata-only probe
  reaches 79%). Head cropping and intensity normalization reduced the champion's reliance
  on it, but cues such as MRI sequence type and the blur of upscaled small images remain:
  a probe on the champion's input pixels still reaches 80%. Tumor-versus-no-tumor results
  in particular are likely optimistic.
- **"Out of distribution" means unlike the training data**, not "not a brain MRI".
  Real brain MRIs framed differently can be refused, and some CT slices are accepted.
- **2D slices only**, four classes, and some label noise between glioma and meningioma.
- **Grad-CAM** shows where the model looked, not why.

How the shortcut was reduced, and what remains: [design.md](docs/design.md#18-reducing-the-source-shortcut).

## Data and license

Dataset: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
by Masoud Nickparvar (CC BY 4.0). The five demo images in
`src/neuroscan/serving/static/samples/` come from its held-out test split.

Code: MIT License, see [LICENSE](LICENSE). Author: Amin Rezaeeyan.
