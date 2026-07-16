"""RunManifest v2 <-> runs/*.manifest.yaml. Manifests are the only in-git trace of a run."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

UNDATED_HINTS = ("latest", ":free")


class TeacherEntry(BaseModel):
    id: str  # dated snapshot id, e.g. deepseek-v4-flash-20260610
    provider: str  # deepseek | zhipu | moonshot | ...
    route: Literal["direct", "openrouter"]

    @field_validator("id")
    @classmethod
    def _dated(cls, v: str) -> str:
        low = v.lower()
        if any(h in low for h in UNDATED_HINTS):
            raise ValueError(f"teacher id {v!r} is not a pinned dated snapshot")
        return v


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


class KeepRates(BaseModel):
    verify: float = 0.0
    judge: float = 0.0


class Spend(BaseModel):
    total: float = 0.0
    by_teacher: dict[str, float] = Field(default_factory=dict)
    by_lane: dict[str, float] = Field(default_factory=dict)


class Provenance(BaseModel):
    hermes_commit: str = ""
    task_bank_commit: str = ""
    datagen_config_hash: str = ""
    serializer_contract: str = "v2"


class Artifacts(BaseModel):
    hf_dataset: str = ""
    eval_families_held_out: list[str] = Field(default_factory=list)
    eval_param_seed: int | None = None


class RunManifest(BaseModel):
    manifest_version: Literal[2] = 2
    run_id: str
    kind: Literal["pilot", "burn"]
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
        path.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
        )
