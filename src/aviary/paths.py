"""Repo/data/hermes path resolution. Data never lives inside the repo."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    root = Path(os.environ.get("AVIARY_DATA_DIR", "~/.local/share/aviary")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def hermes_dir() -> Path:
    raw = os.environ.get("AVIARY_HERMES_DIR")
    if not raw:
        raise RuntimeError(
            "AVIARY_HERMES_DIR is not set; point it at the pinned hermes-agent checkout"
        )
    return Path(raw).expanduser()


def runs_dir() -> Path:
    return REPO_ROOT / "runs"


def configs_dir() -> Path:
    return REPO_ROOT / "datagen" / "configs"
