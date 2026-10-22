"""Experiment configuration: YAML files mapped onto dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# Alphabetical, matching the Kaggle folder names and the model output order.
CLASS_NAMES: tuple[str, ...] = ("glioma", "meningioma", "notumor", "pituitary")
NO_TUMOR_INDEX = CLASS_NAMES.index("notumor")


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    split_csv: str = "splits/v1.csv"
    image_size: int = 224
    batch_size: int = 32
    seed: int = 42


@dataclass
class AugmentConfig:
    flip: bool = True
    rotation: float = 0.1
    zoom: float = 0.1
    brightness: float = 0.15


@dataclass
class ModelConfig:
    name: str = "baseline_cnn"
    dropout: float = 0.5
    dense_units: int = 512
    # Transfer models only: "imagenet" or None (random init, used in tests).
    weights: str | None = "imagenet"
    # Transfer models only: stage 2 unfreezes the backbone from the first layer whose name
    # starts with this prefix, e.g. "block6a" (EfficientNet) or "conv5" (ResNet). "" = all.
    fine_tune_from: str = ""


@dataclass
class TrainConfig:
    epochs: int = 60
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    # Transfer models: stage 1 trains the head only for this many epochs (0 = single stage).
    head_epochs: int = 0
    # Stage 2 learning rate = learning_rate * fine_tune_lr_scale.
    fine_tune_lr_scale: float = 0.1
    # Dry runs: cap the number of train and validation images each (0 = all).
    max_images: int = 0


@dataclass
class Config:
    run_name: str = "run"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _from_dict(cls: type, values: dict[str, Any]) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        default = getattr(cls(), name)
        if is_dataclass(default) and isinstance(value, dict):
            value = _from_dict(type(default), value)
        kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config. ``overrides`` uses dotted keys, e.g. ``{"train.epochs": 1}``."""
    values = yaml.safe_load(Path(path).read_text()) or {}
    for dotted, value in (overrides or {}).items():
        node = values
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    config: Config = _from_dict(Config, values)
    return config


def flatten(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """``{"a": {"b": 1}}`` -> ``{"a.b": 1}``, for logging params to MLflow."""
    flat: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat
