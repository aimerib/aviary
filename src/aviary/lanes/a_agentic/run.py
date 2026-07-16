"""Lane A orchestration: emit inputs/configs, run hermes batch_runner, ingest.

Also the burn guard: `just burn` refuses to start without a recent pilot manifest
whose keep rates sit inside the expected bands and whose config hash matches.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from aviary.io.jsonl import read_raw_jsonl, write_jsonl
from aviary.io.store import RunStore
from aviary.lanes.a_agentic.hermes_config import (
    emit_batch_config,
    emit_batch_inputs,
    verify_config_keys,
)
from aviary.lanes.a_agentic.ingest import IngestContext, IngestError, ingest_hermes_record
from aviary.lanes.a_agentic.taskbank import TaskInstance
from aviary.schema.manifest import RunManifest

log = logging.getLogger(__name__)


class BurnGuardError(RuntimeError):
    pass


@dataclass
class BurnBands:
    verify: tuple[float, float] = (0.25, 0.65)
    judge: tuple[float, float] = (0.6, 0.95)
    max_age_days: int = 14


def guard_burn(
    runs_dir: Path, current_config_hash: str, bands: BurnBands, today_iso: str
) -> RunManifest:
    """Returns the qualifying pilot manifest or raises BurnGuardError."""
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
    if age_days > bands.max_age_days:
        raise BurnGuardError(
            f"latest pilot ({latest.run_id}) is {age_days}d old (max {bands.max_age_days})"
        )
    if latest.provenance.datagen_config_hash != current_config_hash:
        raise BurnGuardError(
            f"datagen config changed since pilot {latest.run_id} — a prompt/config edit "
            "is a new run; re-pilot first (cache-discipline rule)"
        )
    for name, band in (("verify", bands.verify), ("judge", bands.judge)):
        rate = getattr(latest.keep_rates, name)
        if not band[0] <= rate <= band[1]:
            raise BurnGuardError(
                f"pilot {name} keep rate {rate:.2f} outside band {band} — fix task "
                "difficulty or rubric anchors before burning"
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
    toolsets: list[str],
    wire_model: str,
    system_prompt: str,
    store: RunStore,
    num_workers: int,
    batch_size: int,
    run_name: str,
) -> Path:
    """Emit inputs+config, invoke batch_runner, return the trajectories.jsonl path."""
    verify_config_keys(hermes_dir)
    out_dir = store.hermes_out()
    inputs = out_dir / "inputs.jsonl"
    config = out_dir / "batch_config.yaml"
    n = emit_batch_inputs(instances, inputs)
    emit_batch_config(
        toolsets=toolsets,
        wire_model=wire_model,
        system_prompt=system_prompt,
        output_dir=out_dir / run_name,
        num_workers=num_workers,
        batch_size=batch_size,
        out_path=config,
    )
    log.info("hermes batch: %d rollout lines -> %s", n, out_dir / run_name)
    subprocess.run(
        [
            "python",
            "batch_runner.py",
            "--config",
            str(config),
            "--run_name",
            run_name,
            "--dataset_file",
            str(inputs),
        ],
        cwd=hermes_dir,
        check=True,
    )
    return out_dir / run_name / "trajectories.jsonl"


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
