"""Lane A orchestration: emit inputs/configs, run hermes batch_runner, ingest.

Also the burn guard: `just burn` refuses to start without a recent pilot manifest
whose keep rates sit inside the expected bands and whose config hash matches.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from aviary.io.jsonl import read_raw_jsonl, write_jsonl
from aviary.io.store import RunStore
from aviary.lanes.a_agentic.hermes_config import (
    build_batch_command,
    emit_batch_inputs,
    verify_hermes_interface,
)
from aviary.lanes.a_agentic.ingest import IngestContext, IngestError, ingest_hermes_record
from aviary.lanes.a_agentic.taskbank import TaskInstance
from aviary.schema.manifest import RunManifest

if TYPE_CHECKING:
    from aviary.orchestrate import LaneBand

log = logging.getLogger(__name__)


class BurnGuardError(RuntimeError):
    pass


def guard_burn(
    runs_dir: Path,
    current_config_hash: str,
    bands: dict[str, LaneBand],
    burn_lanes: list[str],
    max_age_days: int,
    today_iso: str,
) -> RunManifest:
    """Authorize a burn against the latest finished pilot, or raise BurnGuardError.

    Every lane the burn will run is checked against its OWN band, using the pilot's
    per-lane keep rates. A burn lane the pilot never measured is a hard stop — that
    is the guard that keeps an unpiloted lane (especially a freshly-wired lane A)
    from riding into a full burn on a lane-B/C pilot's coattails."""
    pilots = []
    for path in sorted(runs_dir.glob("*.manifest.yaml")):
        if path.name.startswith("TEMPLATE"):
            continue
        m = RunManifest.load(path)
        if m.kind == "pilot" and m.finished:
            pilots.append(m)
    if not pilots:
        raise BurnGuardError("no finished pilot manifest found — run `just pilot` first")

    latest = max(pilots, key=lambda m: m.finished)
    age_days = _days_between(latest.finished[:10], today_iso[:10])
    if age_days > max_age_days:
        raise BurnGuardError(
            f"latest pilot ({latest.run_id}) is {age_days}d old (max {max_age_days})"
        )
    if latest.provenance.datagen_config_hash != current_config_hash:
        raise BurnGuardError(
            f"datagen config changed since pilot {latest.run_id} — a prompt/config edit "
            "is a new run; re-pilot first (cache-discipline rule)"
        )
    for lane in burn_lanes:
        band = bands.get(lane)
        if band is None:
            raise BurnGuardError(
                f"burn runs lane {lane!r} but no burn_bands entry defines its keep-rate "
                "band — add one before burning"
            )
        measured = latest.keep_rates.by_lane.get(lane)
        if measured is None:
            raise BurnGuardError(
                f"burn runs lane {lane!r} but pilot {latest.run_id} never measured it — "
                "re-pilot with that lane enabled before burning it"
            )
        for name, span in (("verify", band.verify), ("judge", band.judge)):
            rate = getattr(measured, name)
            if not span[0] <= rate <= span[1]:
                raise BurnGuardError(
                    f"lane {lane!r} {name} keep rate {rate:.2f} outside band {span} — fix "
                    "task difficulty or rubric anchors (or the band) before burning"
                )
    return latest


def _days_between(earlier: str, later: str) -> int:
    from datetime import date

    y1, m1, d1 = map(int, earlier.split("-"))
    y2, m2, d2 = map(int, later.split("-"))
    return (date(y2, m2, d2) - date(y1, m1, d1)).days


def run_hermes_batch(
    hermes_dir: Path,
    instances: list[TaskInstance],
    *,
    distribution: str,
    required_tools: list[str],
    wire_model: str,
    base_url: str,
    api_key: str,
    system_prompt: str,
    store: RunStore,
    num_workers: int,
    batch_size: int,
    max_turns: int,
    run_name: str,
    timeout_s: int | None = None,
) -> Path:
    """Emit inputs, invoke batch_runner (CLI flags, cwd=<checkout>), and collect
    its output into the run store. batch_runner writes to <checkout>/data/<run_name>/
    (hardcoded); we copy trajectories.jsonl out immediately so the run store stays
    the single source of truth."""
    verify_hermes_interface(hermes_dir, distribution=distribution, required_tools=required_tools)
    out_dir = store.hermes_out()
    inputs = out_dir / "inputs.jsonl"
    n = emit_batch_inputs(instances, inputs)
    cmd = build_batch_command(
        dataset_file=inputs,
        run_name=run_name,
        distribution=distribution,
        wire_model=wire_model,
        base_url=base_url,
        api_key=api_key,
        system_prompt=system_prompt,
        num_workers=num_workers,
        batch_size=batch_size,
        max_turns=max_turns,
    )
    # cmd carries the API key — log the shape, never the command.
    log.info(
        "hermes batch: %d rollout lines, distribution=%s -> %s",
        n,
        distribution,
        hermes_dir / "data" / run_name,
    )
    try:
        subprocess.run(cmd, cwd=hermes_dir, check=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        # A hung/looping rollout must not block the whole burn indefinitely.
        raise BurnGuardError(
            f"hermes batch exceeded {timeout_s}s wall-clock and was killed"
        ) from e
    produced = hermes_dir / "data" / run_name / "trajectories.jsonl"
    if not produced.exists():
        raise BurnGuardError(f"hermes batch finished but wrote no {produced}")
    collected = out_dir / "trajectories.jsonl"
    shutil.copy2(produced, collected)
    return collected


def ingest_trajectories(trajectories: Path, ctx: IngestContext, store: RunStore) -> int:
    records = []
    skipped = 0
    for raw in read_raw_jsonl(trajectories):
        try:
            records.append(ingest_hermes_record(raw, ctx))
        except IngestError as e:
            skipped += 1
            log.warning("skipped trajectory %s: %s", raw.get("prompt_index"), e)
    n = write_jsonl(store.raw("a"), records)
    log.info("lane A ingest: %d records (%d skipped)", n, skipped)
    return n
