"""Upload to PRIVATE Hugging Face dataset repos (data-in-git rule).

The single path by which anything in this repo uploads data. Two kinds:

1. **Runs** — `$AVIARY_DATA_DIR/<run_id>/`, our own pipeline output; the manifest
   records where it went (provenance rule).
2. **Cleaned external corpora** — `$AVIARY_DATA_DIR/external/<name>/`, third-party
   data filtered for review. These go to a separately-named repo so a foreign
   corpus can never be mistaken for an aviary run, and their filter report ships
   WITH them: cleaned data whose provenance you cannot reconstruct is worse than
   no cleaned data, because it invites reuse on unexamined trust.

Refusals here are structural, not conveniences:

* **Ship-exempt lanes.** Lane D (personal streams) is radioactive: its run data
  ships NOWHERE. Rendered artifacts commingle lanes in one file, so a run that
  contains an exempt lane cannot be partially shipped — it refuses entirely.
* **Private only.** There is deliberately no flag to publish. These corpora carry
  fiction derivatives and fanwork; `private=True` is hard-coded so no call site
  can widen it by passing an argument.
* **No report, no ship.** An external corpus without `report.json` refuses.

A missing `HF_TOKEN` raises rather than degrading — a config failure that looks
like a successful no-op is the shape of bug that silently loses a corpus.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from aviary.io.store import RunStore
from aviary.paths import external_dir, manifest_path
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


def default_external_repo_name(name: str) -> str:
    """Distinct prefix so a foreign corpus is never mistaken for an aviary run."""
    return f"aviary-external-{name}"


def _api():
    """Authenticated HfApi, or a ShipError explaining what is missing."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise ShipError("missing HF_TOKEN — export it before shipping (it is not read from .env)")
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as e:  # pragma: no cover - depends on optional extra
        raise ShipError(
            "huggingface_hub is not installed — install the ship extra: `uv sync --extra ship`"
        ) from e
    return HfApi(token=token)


def _create_private_repo(api, repo: str) -> str:
    """private=True is hard-coded here, never a parameter (see module docstring)."""
    info = api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    return getattr(info, "repo_id", None) or repo


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

    api = _api()
    resolved = _create_private_repo(api, repo)

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


def plan_ship_external(name: str) -> tuple[Path, list[Path], dict]:
    """Validate a cleaned external corpus and return (root, files, report).

    Pure: no network, no mutation. Requiring `report.json` is the point — the
    filters that produced the data are the only way a future reader can judge
    whether it is safe to use, so the audit trail travels with the corpus.
    """
    root = external_dir(name)
    if not root.is_dir():
        raise ShipError(f"no cleaned corpus at {root} — run `just clean-{name}` first")

    data = root / "cleaned.jsonl"
    report_path = root / "report.json"
    if not data.is_file():
        raise ShipError(f"{root} has no cleaned.jsonl — nothing to ship")
    if not report_path.is_file():
        raise ShipError(
            f"{root} has no report.json — refusing to ship cleaned data without the "
            "filter report that produced it (an unauditable corpus invites reuse on "
            "unexamined trust)"
        )
    try:
        report = json.loads(report_path.read_text())
    except json.JSONDecodeError as e:
        raise ShipError(f"{report_path} is not valid JSON: {e}") from e

    files = sorted(p for p in root.rglob("*") if p.is_file())
    return root, files, report


def _dataset_card(name: str, report: dict) -> str:
    """Self-documenting card so the corpus explains itself months from now."""
    dropped = report.get("dropped", {})
    total = report.get("input_rows", 0)
    rows = "\n".join(
        f"| `{reason}` | {count:,} | {(count / total if total else 0):.2%} |"
        for reason, count in dropped.items()
    )
    flags = [k for k in ("strict_minor", "drop_claude") if report.get(k)]
    return f"""---
tags:
  - aviary
  - filtered-derivative
---

# {name} (cleaned)

PRIVATE filtered derivative of `{report.get("source", "unknown")}`, produced by
aviary's `clean-{name}` filter. **Not** aviary-generated corpus data — this is
third-party data prepared for possible interleaving, kept in a separate repo so
it is never confused with an aviary run.

## Filters applied

Input rows: **{total:,}** → survived: **{report.get("survived_filters", 0):,}**
({(report.get("survived_filters", 0) / total if total else 0):.2%}), written:
**{report.get("written", 0):,}**{f" (capped at {report['cap']:,})" if report.get("cap") else ""}.

| dropped by | rows | share |
| --- | ---: | ---: |
{rows}

Optional filters enabled: {", ".join(f"`{f}`" for f in flags) or "none"}.
Downsample seed: `{report.get("seed")}`.

`{{{{user}}}}` placeholders in surviving rows were substituted with stable
per-row names; rows with other unresolvable `{{{{var}}}}` were dropped.

## Caveat

The minor-content filters are regex heuristics with **false negatives**. They
reduce risk substantially; they do not eliminate it. Review before training.
"""


def ship_external(name: str, repo_id: str | None = None, *, dry_run: bool = False) -> str:
    """Upload a cleaned external corpus (+ its report and a card) to a private repo."""
    root, files, report = plan_ship_external(name)
    repo = repo_id or default_external_repo_name(name)
    if dry_run:
        return repo

    api = _api()
    resolved = _create_private_repo(api, repo)
    api.upload_folder(
        folder_path=str(root),
        repo_id=resolved,
        repo_type="dataset",
        commit_message=f"cleaned external corpus {name}",
    )
    api.upload_file(
        path_or_fileobj=_dataset_card(name, report).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=resolved,
        repo_type="dataset",
        commit_message=f"dataset card for {name}",
    )
    return resolved
