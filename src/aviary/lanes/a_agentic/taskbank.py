"""Task bank: load/validate tasks/**/*.task.yaml, expand to concrete instances."""

from __future__ import annotations

import hashlib
import itertools
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from aviary.hashing import canonical_json

_PLACEHOLDER = re.compile(r"\{(\w+)\}")

# Placeholders the task bank does NOT expect in `params` — they are filled from the
# build target, not from the template's own parameter grid.
#
# `persona`: how the simulated user addresses the assistant. Task prompts used to
# hardcode "hey liv", Olivia's nickname, which put Olivia into every lane A record
# of a SORCHA run — the exact contamination the target system exists to prevent,
# and invisible to a grep for "Olivia".
TARGET_PLACEHOLDERS = frozenset({"persona"})


class TaskTemplate(BaseModel):
    id: str
    family: str
    holdout: bool = False
    description: str = ""
    params: dict[str, list] = Field(default_factory=dict)
    # "product": cartesian expansion. "zip": parallel lists, k-th of each — the
    # way to give every instance a UNIQUE file destination (hermes rollouts share
    # one workspace, so instances writing the same path can collide).
    param_mode: Literal["product", "zip"] = "product"
    prompt: str
    tools: list[str]
    n_rollouts: int = 4
    difficulty_target: tuple[float, float] = (0.3, 0.8)
    verifier: str
    notes: str = ""

    @model_validator(mode="after")
    def _check(self) -> TaskTemplate:
        placeholders = set(_PLACEHOLDER.findall(self.prompt)) - TARGET_PLACEHOLDERS
        missing = placeholders - set(self.params)
        if missing:
            raise ValueError(f"{self.id}: prompt placeholders without params: {sorted(missing)}")
        if not self.id.startswith(f"{self.family}."):
            raise ValueError(f"{self.id}: id must be <family>.<short_name>")
        if self.param_mode == "zip" and len({len(v) for v in self.params.values()}) > 1:
            raise ValueError(f"{self.id}: zip param_mode requires equal-length param lists")
        return self


class TaskInstance(BaseModel):
    template_id: str
    family: str
    holdout: bool
    params: dict[str, str]
    prompt: str
    n_rollouts: int
    verifier: str

    def instance_key(self) -> str:
        payload = canonical_json({"template": self.template_id, "params": self.params})
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def load_taskbank(tasks_root: Path) -> list[TaskTemplate]:
    templates: list[TaskTemplate] = []
    for path in sorted(tasks_root.rglob("*.task.yaml")):
        if path.name.startswith("TEMPLATE"):
            continue
        templates.append(TaskTemplate.model_validate(yaml.safe_load(path.read_text())))
    ids = [t.id for t in templates]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate template ids: {dupes}")
    return templates


def expand(template: TaskTemplate, persona: str = "") -> list[TaskInstance]:
    if not template.params:
        combos: list[dict[str, str]] = [{}]
    else:
        keys = sorted(template.params)
        if template.param_mode == "zip":
            combos = [
                dict(zip(keys, values, strict=True))
                for values in zip(*(template.params[k] for k in keys), strict=True)
            ]
        else:
            combos = [
                dict(zip(keys, values, strict=True))
                for values in itertools.product(*(template.params[k] for k in keys))
            ]
    return [
        TaskInstance(
            template_id=template.id,
            family=template.family,
            holdout=template.holdout,
            params={k: str(v) for k, v in combo.items()},
            prompt=template.prompt.format(**combo, persona=persona),
            n_rollouts=template.n_rollouts,
            verifier=template.verifier,
        )
        for combo in combos
    ]


def expand_all(templates: list[TaskTemplate], persona: str = "") -> list[TaskInstance]:
    """`persona` is how the user addresses the assistant in task prompts. It comes
    from the build target so one task bank serves every persona — a hardcoded name
    here lands in every lane A record of every target."""
    return [inst for t in templates for inst in expand(t, persona)]


def rollout_order(instances: list[TaskInstance]) -> list[TaskInstance]:
    """The prompt_index contract, in ONE place.

    One entry per rollout line: each instance appears n_rollouts times (rejection
    sampling), INTERLEAVED round-robin — hermes runs prompts concurrently in a
    shared workspace, so same-instance rollouts (same file destinations) must not
    run back-to-back. emit_batch_inputs writes prompts in exactly this order, and
    ingest maps a trajectory's prompt_index i back to rollout_order(instances)[i].
    Both sides MUST consume this primitive — if the two expansions ever drift,
    every lane-A record is silently mislabeled (wrong instance -> wrong
    family/holdout, which can leak holdout across the split)."""
    rounds = max((inst.n_rollouts for inst in instances), default=0)
    return [inst for r in range(rounds) for inst in instances if r < inst.n_rollouts]
