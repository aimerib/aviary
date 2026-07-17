"""Repo/data/hermes path resolution. Data never lives inside the repo."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    root = Path(os.environ.get("AVIARY_DATA_DIR", "~/.local/share/aviary")).expanduser().resolve()
    # Data-in-git rule: run data must never live inside the working tree.
    if root == REPO_ROOT or REPO_ROOT in root.parents:
        raise RuntimeError(
            f"AVIARY_DATA_DIR ({root}) is inside the repo ({REPO_ROOT}); run data must live "
            "outside the working tree (data-in-git rule). Point it elsewhere."
        )
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
