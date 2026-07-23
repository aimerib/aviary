"""RunManifest v2 <-> runs/*.manifest.yaml. Manifests are the only in-git trace of a run."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

UNDATED_HINTS = ("latest", ":free")
# A pinned snapshot ends in a dated suffix: …-YYYYMMDD (optionally after / or _).
# e.g. deepseek-v4-flash-20260610, moonshotai/kimi-k3-20260522.
_DATED_SUFFIX = re.compile(r"[-_/]20\d{6}$")


def assert_dated_snapshot(v: str) -> str:
    """Provenance rule: teacher ids/wire models must be pinned dated snapshots,
    never `latest`/`:free`, and never undated or truncated. The date suffix is
    REQUIRED (positive check) — a blacklist alone lets a typo'd id like
    `…-2026061` slip into a paid burn and the manifest."""
    if any(h in v.lower() for h in UNDATED_HINTS):
        raise ValueError(f"model id {v!r} is not a pinned dated snapshot (undated hint)")
    if not _DATED_SUFFIX.search(v):
        raise ValueError(f"model id {v!r} lacks a dated snapshot suffix (expected …-YYYYMMDD)")
    return v


class TeacherEntry(BaseModel):
    id: str  # dated snapshot id, e.g. deepseek-v4-flash-20260610
    provider: str  # deepseek | zhipu | moonshot | ...
    route: Literal["direct", "openrouter"]

    @field_validator("id")
    @classmethod
    def _dated(cls, v: str) -> str:
        return assert_dated_snapshot(v)


class Teachers(BaseModel):
    roster: list[TeacherEntry]
    # role -> roster id; nested per lane, e.g. assignments["lane_b"]["dialogue"]
    assignments: dict[str, dict[str, str]] = Field(default_factory=dict)

    def resolve(self, lane_key: str, role: str) -> TeacherEntry:
        rid = self.assignments[lane_key][role]
        for t in self.roster:
            if t.id == rid:
                return t
        raise KeyError(f"assignment {lane_key}.{role} -> {rid!r} not in roster")


class Counts(BaseModel):
    instances: int = 0
    rollouts: int = 0
    verified: int = 0
    judged: int = 0
    rendered_train: int = 0
    rendered_eval: int = 0
    rendered_dpo_pairs: int = 0
    rendered_nsp: int = 0


class LaneKeepRate(BaseModel):
    verify: float = 0.0
    judge: float = 0.0


class KeepRates(BaseModel):
    # Aggregate across all lanes — a human-readable summary only. The burn guard
    # reads `by_lane`, because lanes have structurally different rates by design
    # (lane B judges low on the thought_quality floor; a blended figure hides it).
    verify: float = 0.0
    judge: float = 0.0
    by_lane: dict[str, LaneKeepRate] = Field(default_factory=dict)


class Spend(BaseModel):
    total: float = 0.0
    by_teacher: dict[str, float] = Field(default_factory=dict)
    by_lane: dict[str, float] = Field(default_factory=dict)


class Provenance(BaseModel):
    hermes_commit: str = ""
    task_bank_commit: str = ""
    datagen_config_hash: str = ""
    # Hash of gate/render inputs NOT under datagen/ (rubrics, scrub patterns,
    # verifiers, tasks). Frozen at generation, re-asserted at gate/render so a
    # rubric/verifier/task edit can't silently change keep/judge/dedupe outcomes.
    gate_inputs_hash: str = ""
    serializer_contract: str = "v3"
    # Build target this run generated for (datagen/configs/targets/<name>.yaml) and
    # its base model. Empty on pre-target manifests.
    target: str = ""
    base_model: str = ""


class Artifacts(BaseModel):
    hf_dataset: str = ""
    # Lanes whose run data never ships anywhere (lane D: personal corpus stays
    # under $AVIARY_DATA_DIR only). Recorded so the exemption is auditable.
    ship_exempt_lanes: list[str] = Field(default_factory=list)
    eval_families_held_out: list[str] = Field(default_factory=list)
    eval_param_seed: int | None = None


class RunManifest(BaseModel):
    manifest_version: Literal[2] = 2
    run_id: str
    kind: Literal["pilot", "burn"]
    # running until the generation body completes; failed if it raised. Distinguishes
    # a crash from an in-progress run (finished="" alone can't) in the only in-git trace.
    status: Literal["running", "complete", "failed"] = "running"
    lanes: list[str] = Field(default_factory=lambda: ["a", "b", "c"])
    started: str = ""
    finished: str = ""
    provenance: Provenance = Field(default_factory=Provenance)
    teachers: Teachers
    counts: Counts = Field(default_factory=Counts)
    keep_rates: KeepRates = Field(default_factory=KeepRates)
    spend_usd: Spend = Field(default_factory=Spend)
    prompt_set_hash: str = ""
    artifacts: Artifacts = Field(default_factory=Artifacts)
    notes: str = ""

    @classmethod
    def load(cls, path: Path) -> RunManifest:
        return cls.model_validate(yaml.safe_load(path.read_text()))

    def save(self, path: Path) -> None:
        # Atomic: a crash mid-write must not truncate the only in-git trace of a run.
        path.parent.mkdir(parents=True, exist_ok=True)
        data = yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(data)
        os.replace(tmp, path)
