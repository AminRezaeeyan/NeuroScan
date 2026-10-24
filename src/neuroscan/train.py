"""Train one model from a config and write a run bundle.

    python -m neuroscan.train --config configs/baseline_cnn.yaml --out runs/
    python -m neuroscan.train --config configs/baseline_cnn.yaml --out runs/ --dry-run

Rerunning the same command after an interruption resumes from the last epoch.
No MLflow here: bundles are imported on the tracking machine with
``python -m neuroscan.tracking.import_run``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import keras
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from neuroscan import bundle
from neuroscan.config import Config, load_config
from neuroscan.data.datasets import load_split, make_dataset
from neuroscan.models import TRANSFER_MODELS, build_model
from neuroscan.models.common import compile_model, set_backbone_trainable
from neuroscan.reports.plots import learning_curves

MONITOR = "val_macro_f1"
EPOCH_END = "epoch_end_ms"  # wall-clock time each epoch ended


class MacroF1(keras.callbacks.Callback):
    """Adds ``val_macro_f1`` and ``learning_rate`` to the epoch logs.

    Must come first in the callback list so early stopping, checkpointing and LR
    reduction can monitor it.
    """

    def __init__(self, val_ds: Any, y_true: np.ndarray) -> None:
        super().__init__()
        self.val_ds = val_ds
        self.y_true = y_true

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if logs is None:
            return
        probs = self.model.predict(self.val_ds, verbose=0)
        y_pred = probs.argmax(axis=1)
        logs[MONITOR] = float(f1_score(self.y_true, y_pred, average="macro"))
        logs["learning_rate"] = float(
            keras.ops.convert_to_numpy(self.model.optimizer.learning_rate)
        )
        logs[EPOCH_END] = bundle.now_ms()


def cap_per_class(df: pd.DataFrame, max_images: int, seed: int) -> pd.DataFrame:
    """Stratified subsample for dry runs."""
    if not max_images or len(df) <= max_images:
        return df
    per_class = max(1, max_images // df["label"].nunique())
    return df.groupby("label", group_keys=False).sample(
        n=per_class, random_state=seed, replace=False
    )


def callbacks(
    cfg: Config, val_ds: Any, y_val: np.ndarray, run_dir: Path, stage: str, early_stop: bool
) -> list[keras.callbacks.Callback]:
    ckpt = run_dir / bundle.CHECKPOINT_DIR
    cbs: list[keras.callbacks.Callback] = [
        MacroF1(val_ds, y_val),
        keras.callbacks.ModelCheckpoint(
            str(ckpt / f"best_{stage}.keras"), monitor=MONITOR, mode="max", save_best_only=True
        ),
        # Resumes after a Colab disconnect when the same command is rerun.
        keras.callbacks.BackupAndRestore(str(ckpt / f"backup_{stage}")),
        keras.callbacks.CSVLogger(str(ckpt / f"log_{stage}.csv"), append=True),
    ]
    if early_stop:
        t = cfg.train
        cbs += [
            keras.callbacks.EarlyStopping(
                monitor=MONITOR,
                mode="max",
                patience=t.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor=MONITOR,
                mode="max",
                factor=t.lr_factor,
                patience=t.lr_patience,
                min_lr=t.min_lr,
                verbose=1,
            ),
        ]
    return cbs


def read_history(run_dir: Path, stage: str) -> dict[str, list[float]]:
    """Per-epoch logs from the CSV log, which survives resumes (Keras History doesn't)."""
    path = run_dir / bundle.CHECKPOINT_DIR / f"log_{stage}.csv"
    if not path.exists():
        return {}
    log = pd.read_csv(path).drop_duplicates("epoch", keep="last").sort_values("epoch")
    return {col: log[col].astype(float).tolist() for col in log.columns if col != "epoch"}


def best_checkpoint(run_dir: Path, stages: list[str]) -> tuple[Path, str, int, float]:
    """The checkpoint of the stage whose best epoch has the highest val macro F1."""
    best: tuple[Path, str, int, float] | None = None
    offset = 0
    for stage in stages:
        hist = read_history(run_dir, stage)
        scores = hist.get(MONITOR, [])
        if scores:
            i = int(np.argmax(scores))
            if best is None or scores[i] > best[3]:
                best = (
                    run_dir / bundle.CHECKPOINT_DIR / f"best_{stage}.keras",
                    stage,
                    offset + i,
                    scores[i],
                )
        offset += len(scores)
    if best is None:
        raise RuntimeError(f"No training log found in {run_dir}")
    return best


def train(cfg: Config, raw_dir: Path, split_csv: Path, out: Path, overwrite: bool = False) -> Path:
    run_dir = out / cfg.run_name
    if (run_dir / bundle.MODEL_FILE).exists():
        if not overwrite:
            raise FileExistsError(f"{run_dir} is a finished run; pass --overwrite to replace it")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / bundle.FIGURES_DIR).mkdir(exist_ok=True)
    (run_dir / bundle.CHECKPOINT_DIR).mkdir(exist_ok=True)

    meta_path = run_dir / bundle.META_FILE
    if meta_path.exists():  # resuming
        meta = bundle.read_json(meta_path)
        print(f"Resuming {run_dir} (started {meta['start_ms']})")
        commit = bundle.git_commit()
        if commit != meta["git_commit"]:
            # The trained weights come from the code that runs now; keep the first commit too.
            meta.setdefault("first_git_commit", meta["git_commit"])
            meta["git_commit"] = commit
            bundle.write_json(meta_path, meta)
    else:
        meta = {
            "run_name": cfg.run_name,
            "model": cfg.model.name,
            "git_commit": bundle.git_commit(),
            "split_file": split_csv.name,
            "split_hash": bundle.file_hash(split_csv),
            "start_ms": bundle.now_ms(),
            "gpu": bundle.gpu_name(),
            **bundle.environment(),
        }
        bundle.write_json(meta_path, meta)
    cfg.save(run_dir / bundle.CONFIG_FILE)

    keras.utils.set_random_seed(cfg.data.seed)
    split = load_split(split_csv)
    train_df = cap_per_class(split[split["split"] == "train"], cfg.train.max_images, cfg.data.seed)
    val_df = cap_per_class(split[split["split"] == "val"], cfg.train.max_images, cfg.data.seed)
    train_ds = make_dataset(train_df, raw_dir, cfg, training=True, cache=True)
    val_ds = make_dataset(val_df, raw_dir, cfg, training=False, cache=True)
    y_val = val_df["label_idx"].to_numpy()
    print(f"Train images: {len(train_df)}, validation images: {len(val_df)}")

    model = build_model(cfg)
    two_stage = cfg.model.name in TRANSFER_MODELS and cfg.train.head_epochs > 0

    if two_stage:
        head_final = run_dir / bundle.CHECKPOINT_DIR / "head_final.keras"
        if head_final.exists():  # resumed after stage 1 had finished
            model = keras.models.load_model(head_final)
        else:
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=cfg.train.head_epochs,
                callbacks=callbacks(cfg, val_ds, y_val, run_dir, "head", early_stop=False),
            )
            model.save(head_final)
        set_backbone_trainable(model, cfg.model.fine_tune_from)
        compile_model(model, cfg.train.learning_rate * cfg.train.fine_tune_lr_scale)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.train.epochs - cfg.train.head_epochs,
            callbacks=callbacks(cfg, val_ds, y_val, run_dir, "finetune", early_stop=True),
        )
    else:
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.train.epochs,
            callbacks=callbacks(cfg, val_ds, y_val, run_dir, "main", early_stop=True),
        )

    return finalize(run_dir, cfg, split_csv)


def end_time_ms(meta: dict[str, Any], epoch_ends: list[float], ckpt_dir: Path) -> int:
    """When the last epoch ended.

    Uses the logged epoch end times. Older logs don't have them; then the span between the
    first and last checkpoint file is added to the start time. File times are only used as
    a span because a zip downloaded from Google Drive shifts them all by the same offset.
    """
    if epoch_ends:
        return int(max(epoch_ends))
    times = [p.stat().st_mtime for p in ckpt_dir.rglob("*") if p.is_file()]
    return int(meta["start_ms"] + (max(times) - min(times)) * 1000)


def finalize(run_dir: Path, cfg: Config, split_csv: Path) -> Path:
    """Write history, best model, figure and final meta from the checkpoints and logs.

    Also used on its own (``--finalize``) when a session ended after the last epoch but
    before the bundle was written; the end time is that of the newest checkpoint file.
    """
    ckpt_dir = run_dir / bundle.CHECKPOINT_DIR
    meta_path = run_dir / bundle.META_FILE
    meta = bundle.read_json(meta_path)
    two_stage = cfg.model.name in TRANSFER_MODELS and cfg.train.head_epochs > 0
    stages = ["head", "finetune"] if two_stage else ["main"]

    history: dict[str, list[Any]] = {}
    epoch_ends: list[float] = []
    for stage in stages:
        part = read_history(run_dir, stage)
        epoch_ends += part.pop(EPOCH_END, [])
        n = len(part.get(MONITOR, []))
        for key, values in part.items():
            history.setdefault(key, []).extend(values)
        history.setdefault("stage", []).extend([stage] * n)
    bundle.write_json(run_dir / bundle.HISTORY_FILE, history)

    ckpt, best_stage, best_epoch, best_f1 = best_checkpoint(run_dir, stages)
    shutil.copy(ckpt, run_dir / bundle.MODEL_FILE)
    params = keras.models.load_model(run_dir / bundle.MODEL_FILE).count_params()

    boundary = cfg.train.head_epochs if two_stage else 0
    numeric = {k: v for k, v in history.items() if k != "stage"}
    (run_dir / bundle.FIGURES_DIR).mkdir(exist_ok=True)
    learning_curves(numeric, boundary).savefig(
        run_dir / bundle.FIGURES_DIR / "learning_curves.png", dpi=120
    )

    split = load_split(split_csv)
    meta.update(
        end_ms=end_time_ms(meta, epoch_ends, ckpt_dir),
        epochs_trained=len(history.get(MONITOR, [])),
        best_epoch=best_epoch,
        best_stage=best_stage,
        best_val_macro_f1=best_f1,
        num_train=len(
            cap_per_class(split[split["split"] == "train"], cfg.train.max_images, cfg.data.seed)
        ),
        num_val=len(
            cap_per_class(split[split["split"] == "val"], cfg.train.max_images, cfg.data.seed)
        ),
        params=params,
    )
    bundle.write_json(meta_path, meta)
    shutil.rmtree(ckpt_dir)
    print(
        f"Run bundle written to {run_dir} (best val macro F1 {best_f1:.4f}, epoch {best_epoch + 1})"
    )
    return run_dir


def dry_run_overrides(cfg_path: Path) -> dict[str, Any]:
    two_stage = (yaml.safe_load(cfg_path.read_text()).get("train") or {}).get("head_epochs", 0)
    return {
        "train.epochs": 2 if two_stage else 1,
        "train.head_epochs": 1 if two_stage else 0,
        "train.max_images": 32,
        "data.batch_size": 16,
    }


def parse_set(values: list[str]) -> dict[str, Any]:
    """``--set train.epochs=3`` -> ``{"train.epochs": 3}`` (values parsed as YAML)."""
    out = {}
    for item in values:
        key, _, raw = item.partition("=")
        out[key] = yaml.safe_load(raw)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--raw-dir", type=Path, help="defaults to data.raw_dir in the config")
    parser.add_argument("--split-csv", type=Path, help="defaults to data.split_csv in the config")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides"
    )
    parser.add_argument("--dry-run", action="store_true", help="1 epoch per stage on 32 images")
    parser.add_argument("--overwrite", action="store_true", help="replace a finished run")
    parser.add_argument(
        "--finalize",
        type=Path,
        metavar="RUN_DIR",
        help="only write the bundle of a run whose training finished (uses RUN_DIR/config.yaml)",
    )
    args = parser.parse_args()

    if args.finalize:
        cfg = load_config(args.finalize / bundle.CONFIG_FILE)
        finalize(args.finalize, cfg, args.split_csv or Path(cfg.data.split_csv))
        return

    if args.config is None:
        parser.error("--config is required")
    overrides = parse_set(args.set)
    if args.dry_run:
        overrides = {**dry_run_overrides(args.config), **overrides}
    cfg = load_config(args.config, overrides)
    if args.dry_run:
        cfg.run_name += "-dryrun"
    raw_dir = args.raw_dir or Path(cfg.data.raw_dir)
    split_csv = args.split_csv or Path(cfg.data.split_csv)
    train(cfg, raw_dir, split_csv, args.out, args.overwrite)


if __name__ == "__main__":
    main()
