"""Run bundle: the folder a training run writes, later imported into MLflow.

Layout::

    config.yaml    exact config used
    meta.json      model, git commit, split hash, start/end time, GPU, versions, best epoch
    history.json   per-epoch metrics (Keras history plus val_macro_f1, learning_rate, stage)
    model.keras    best model (by validation macro F1)
    figures/       plots made during training
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess  # nosec B404 - fixed git command, no user input
import time
from pathlib import Path
from typing import Any

CONFIG_FILE = "config.yaml"
META_FILE = "meta.json"
HISTORY_FILE = "history.json"
MODEL_FILE = "model.keras"
FIGURES_DIR = "figures"
CHECKPOINT_DIR = "checkpoints"
# Written into source archives made for Colab, where there is no .git folder.
COMMIT_FILE = "GIT_COMMIT"


def now_ms() -> int:
    return int(time.time() * 1000)


def file_hash(path: str | Path, length: int = 12) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:length]


def _git(*args: str) -> str:
    git = shutil.which("git")
    if git is None:
        raise OSError("git not found")
    here = Path(__file__).resolve().parent
    out = subprocess.run(  # noqa: S603  # nosec B603 - fixed arguments, no user input
        [git, "-C", str(here), *args], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def git_commit() -> str:
    """Commit of the running code: from git if available, else from the archive's GIT_COMMIT."""
    try:
        commit = _git("rev-parse", "HEAD")
        dirty = _git("status", "--porcelain", "--untracked-files=no")
        return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.CalledProcessError):
        pass
    for parent in Path(__file__).resolve().parents:
        marker = parent / COMMIT_FILE
        if marker.is_file():
            return marker.read_text().strip()
    return "unknown"


def gpu_name() -> str:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return "cpu"
    details = tf.config.experimental.get_device_details(gpus[0])
    return str(details.get("device_name", "gpu")).replace(" ", "-")


def environment() -> dict[str, str]:
    import keras
    import tensorflow as tf

    from neuroscan import __version__

    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "neuroscan": __version__,
    }


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)  # atomic, so a Colab disconnect never leaves half a file


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())
