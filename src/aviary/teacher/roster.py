"""Teacher roster: datagen/configs/teachers.yaml -> routes and role assignments.

Every teacher `id` is a dated provenance pin (…-YYYYMMDD) we assign at verify time
— the manifest's provenance token. `wire_model` carries the provider's real name,
which for DeepSeek/GLM/Kimi is a rolling (undated) alias, so only `id` is
dated-validated. The judge policy is cross-vendor: a record is judged by a
different vendor than generated it.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from aviary.schema.manifest import assert_dated_snapshot


class TeacherRoute(BaseModel):
    id: str  # aviary provenance token: a dated pin (…-YYYYMMDD) we assign
    provider: str  # deepseek | zhipu | moonshot
    route: str  # direct | openrouter
    base_url: str
    # Name sent on the wire. DeepSeek/GLM/Kimi ship ROLLING names (deepseek-v4-flash,
    # glm-5.2, moonshotai/kimi-k3) with no dated snapshots, so this is intentionally
    # NOT dated-validated — provenance is carried by `id`, which is (see docstring).
    wire_model: str
    api_key_env: str
    api_key_env_fallback: str | None = None
    max_concurrency: int = 4
    # Extra request-body keys merged into the payload ONLY on JSON-extraction calls
    # (response_json). Home for reasoning-model controls that must be OFF for
    # structured output — e.g. DeepSeek {thinking: {type: disabled}}: in thinking
    # mode the model spends its whole max_tokens budget on reasoning and returns
    # empty content, so long-scene extraction silently fails. Left off generative /
    # roleplay calls, which keep their reasoning.
    json_extra_body: dict = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _dated(cls, v: str) -> str:
        return assert_dated_snapshot(v)


class Roster(BaseModel):
    teachers: list[TeacherRoute]
    # lane_key -> role -> teacher id, e.g. assignments["lane_b"]["dialogue"]
    assignments: dict[str, dict[str, str]]
    judge_policy: str = "cross_vendor"
    hermes_pin: str = ""

    def route_for(self, teacher_id: str) -> TeacherRoute:
        for t in self.teachers:
            if t.id == teacher_id:
                return t
        raise KeyError(f"unknown teacher id {teacher_id!r}")

    def assigned(self, lane_key: str, role: str) -> TeacherRoute:
        return self.route_for(self.assignments[lane_key][role])

    def judge_for(self, generator_id: str) -> TeacherRoute:
        """Cross-vendor judge: first judge-role teacher from a different vendor."""
        gen_provider = self.route_for(generator_id).provider
        candidates = [self.route_for(i) for i in self.assignments["judge"].values()]
        for c in candidates:
            if c.provider != gen_provider:
                return c
        raise ValueError(
            f"no cross-vendor judge available for provider {gen_provider!r}; "
            "add a judge assignment from another vendor"
        )

    @classmethod
    def load(cls, path: Path) -> Roster:
        return cls.model_validate(yaml.safe_load(path.read_text()))
