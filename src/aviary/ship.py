"""Ship a run's data to a PRIVATE Hugging Face dataset repo (data-in-git rule).

Run data lives under `$AVIARY_DATA_DIR/<run_id>/` and never in git; the manifest
records where it went (provenance rule). This module is the only path that
uploads run data anywhere.

Two refusals are structural, not conveniences:

* **Ship-exempt lanes.** Lane D (personal streams) is radioactive: its run data
  ships NOWHERE. Rendered artifacts commingle lanes in one file, so a run that
  contains an exempt lane cannot be partially shipped — it refuses entirely.
* **Private only.** There is deliberately no flag to publish. The corpus carries
  fiction derivatives and fanwork; `private=True` is hard-coded so no call site
  can widen it by passing an argument.

A missing `HF_TOKEN` raises rather than degrading — a config failure that looks
like a successful no-op is the shape of bug that silently loses a corpus.
"""

from __future__ import annotations

import os
from pathlib import Path

from aviary.io.store import RunStore
from aviary.paths import manifest_path
from aviary.schema.manifest import RunManifest

# Uploaded verbatim: raw -> gated -> rendered is the full reproducibility chain,
# plus hermes/ (lane A's source-of-truth trajectories) and laneb/ intermediates.
_SKIP_DIRS: frozenset[str] = frozenset()

# Rendered artifacts a run must have before it is worth shipping.
_REQUIRED_RENDERED = ("train_with_thoughts", "train_no_thoughts")


class ShipError(RuntimeError):
    """Refused to ship. Never raised for a transport hiccup — only for a rule
    violation or missing precondition the caller must resolve deliberately."""


def default_repo_name(run_id: str) -> str:
    """Bare repo name; HF resolves it under the token's own namespace."""
    return f"aviary-{run_id}"


def plan_ship(run_id: str) -> tuple[Path, list[Path]]:
    """Validate a run is shippable and return (run_root, files_to_upload).

    Pure: no network, no mutation. The CLI's --dry-run stops here, and the tests
    exercise every refusal through this function.
    """
    manifest = RunManifest.load(manifest_path(run_id))

    exempt = manifest.artifacts.ship_exempt_lanes
    if exempt:
        raise ShipError(
            f"run {run_id!r} contains ship-exempt lane(s) {sorted(exempt)} — that data "
            "ships nowhere, and rendered artifacts commingle lanes so the run cannot be "
            "partially shipped. Render a target that excludes those lanes instead."
        )

    store = RunStore(run_id)
    if not store.root.is_dir():
        raise ShipError(f"no run store at {store.root} — nothing to ship")

    missing = [n for n in _REQUIRED_RENDERED if not store.rendered(n).is_file()]
    if missing:
        raise ShipError(
            f"run {run_id!r} has no rendered {missing} — run `just render {run_id}` first "
            "(shipping an ungated/unrendered run would publish records that never passed "
            "the split rule)"
        )

    files = sorted(
        p
        for p in store.root.rglob("*")
        if p.is_file() and not any(part in _SKIP_DIRS for part in p.relative_to(store.root).parts)
    )
    if not files:
        raise ShipError(f"run store {store.root} is empty")
    return store.root, files


def ship_run(run_id: str, repo_id: str | None = None, *, dry_run: bool = False) -> str:
    """Upload a run to a private HF dataset repo; record the repo in its manifest.

    Returns the repo id (dry run returns the intended id without contacting HF).
    """
    root, files = plan_ship(run_id)
    repo = repo_id or default_repo_name(run_id)
    if dry_run:
        return repo

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise ShipError("missing HF_TOKEN — export it before shipping (it is not read from .env)")

    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as e:  # pragma: no cover - depends on optional extra
        raise ShipError(
            "huggingface_hub is not installed — install the ship extra: `uv sync --extra ship`"
        ) from e

    api = HfApi(token=token)
    # private=True is hard-coded, never a parameter (see module docstring).
    info = api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    resolved = getattr(info, "repo_id", None) or repo

    api.upload_folder(
        folder_path=str(root),
        repo_id=resolved,
        repo_type="dataset",
        commit_message=f"aviary run {run_id}",
    )
    # The manifest is the in-git trace and lives outside the run store; ship it too
    # so the dataset carries its own provenance.
    api.upload_file(
        path_or_fileobj=str(manifest_path(run_id)),
        path_in_repo="manifest.yaml",
        repo_id=resolved,
        repo_type="dataset",
        commit_message=f"aviary manifest {run_id}",
    )

    # Provenance rule: the manifest records where the run shipped.
    manifest = RunManifest.load(manifest_path(run_id))
    manifest.artifacts.hf_dataset = resolved
    manifest.save(manifest_path(run_id))
    return resolved
