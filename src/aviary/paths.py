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


def manifest_path(run_id: str) -> Path:
    return runs_dir() / f"{run_id}.manifest.yaml"


def external_dir(name: str = "") -> Path:
    """Cleaned third-party corpora, kept beside run stores but never inside one.

    A run store is the auditable output of OUR pipeline (raw -> gated -> rendered)
    and ships as a unit; third-party data has different provenance and licensing,
    so it lives in its own tree and is never swept into a run's ship."""
    root = data_dir() / "external"
    return root / name if name else root


def configs_dir() -> Path:
    return REPO_ROOT / "datagen" / "configs"
