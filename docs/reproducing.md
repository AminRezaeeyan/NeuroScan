# Reproducing NeuroScan

Every number in this repository can be regenerated with the steps below. Training runs
on Google Colab (free GPU); everything else runs on a laptop CPU.

## 0. Setup

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), a
[Kaggle API token](https://www.kaggle.com/settings/api) and, for training, a Google
account. Docker is optional.

```bash
make install        # uv sync --all-extras + pre-commit hooks
```

Put the Kaggle token in `~/.kaggle/access_token`.

## 1. Data

```bash
make data           # download and unpack the dataset into data/raw/ (~165 MB)
make dedup          # duplicate audit -> data/duplicates.csv, docs/figures/duplicate_pairs.png
make split          # rebuild splits/v1.csv (identical to the committed file)
```

`splits/v1.csv` is committed; rebuilding it is only needed to verify it. The exploration
notebook (`notebooks/01_data_exploration.ipynb`) reproduces the metadata-shortcut and
dataset statistics.

## 2. Experiment tracking

```bash
make mlflow         # MLflow UI on http://127.0.0.1:5000 (SQLite + ./mlartifacts)
```

The experiment is `neuroscan`; the registered model is `neuroscan-classifier`.

## 3. Train on Colab

Colab never talks to MLflow. Each session writes a **run bundle** to Google Drive, which
is imported locally.

1. Commit your changes, then package the exact commit:
   ```bash
   make colab-package          # -> dist/neuroscan-<commit>.tar.gz
   ```
2. In Google Drive, create `My Drive/neuroscan/code/` and upload the archive there.
3. Open `notebooks/colab_train.ipynb` in Colab (File → Upload notebook).
4. Add a Colab secret (key icon in the sidebar) named `KAGGLE_API_TOKEN` and enable
   notebook access.
5. Runtime → Change runtime type → T4 GPU.
6. In the first cell set `PACKAGE` to the archive name, `CONFIG` to one of
   `configs/baseline_cnn.yaml`, `configs/cnn_gap.yaml`, `configs/efficientnet_b0.yaml`,
   `configs/resnet50.yaml`, and `DRY_RUN = False`.
7. Runtime → Run all. If the session disconnects, run all again with the same settings:
   training resumes from the last finished epoch.

Set `DRY_RUN = True` first to push one epoch on 32 images through the whole pipeline.

Each model takes roughly 40–50 minutes on a T4. If a session ends after training but
before the bundle is written, finish it with:

```bash
uv run python -m neuroscan.train --finalize runs/<run-name>
```

## 4. Import and evaluate

Download `My Drive/neuroscan/runs/<run-name>/` into `runs/`, then:

```bash
make import BUNDLE=runs/baseline-cnn     # prints the MLflow run ID and model version
make eval RUN=<run-id> SETS="val test"   # metrics, intervals, shortcut check, figures
```

Repeat for the four models.

To re-evaluate the original Kaggle model as a reference (if you have its file):

```bash
uv run python -m neuroscan.tracking.log_legacy <path/to/best_model.keras>
uv run python -m neuroscan.evaluate --run-id <id> --eval-set kaggle-testing kaggle-testing-clean
```

## 5. Select, calibrate, explain, refuse

```bash
make select              # comparison table + the pre-registered rule, step by step
make select APPLY=1      # set the champion and baseline aliases
make calibrate           # temperature on validation, package, move @champion
make explain             # Grad-CAM grid and brain-mask check (test split)
make ood-data            # public OOD test images into data/ood/ (~920 MB download)
make ood                 # fit the detector, evaluate, package, move @champion
make figures             # docs/results.md and docs/figures/ from MLflow
make external-data       # PMRAM external test set into data/external/ (~168 MB)
make external-eval       # overlap check against training data, then evaluate
```

`make calibrate` and `make ood` each register a new model version and move the
`champion` alias to it; `make ood` needs the calibration metrics in the run.

## 6. Serve

```bash
make serve               # API + demo page on http://127.0.0.1:8000
make docker-up           # same, in a hardened container (offline build); make docker-down
```

The service loads `models:/neuroscan-classifier@champion`. Set `NEUROSCAN_MODEL_URI` to
serve another version, for example `models:/neuroscan-classifier/3`.

## 7. Check and measure

```bash
make ci                  # ruff, mypy, bandit, pytest (what CI runs)
make test-behavioral     # behavioral tests against the local champion
make optimize            # TFLite export + runtime benchmark
make loadtest            # load test a running API (start make serve first)
```

## Resource notes

- The Docker build copies the local uv cache (~3 GB) into BuildKit and prunes it
  afterwards; keep ~8 GB free on the disk holding Docker's data.
- Behavioral tests, evaluation and the load test each load TensorFlow (1.5–2 GB RAM);
  run them one at a time on small machines.
