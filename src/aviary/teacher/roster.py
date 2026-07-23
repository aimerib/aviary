"""Teacher roster: datagen/configs/teachers.yaml -> routes and role assignments.

Every teacher `id` is a dated provenance pin (…-YYYYMMDD) we assign at verify time
— the manifest's provenance token. `wire_model` carries the provider's real name,
which for DeepSeek/GLM/Kimi is a rolling (undated) alias, so only `id` is
dated-validated. The judge policy is cross-vendor: a record is judged by a
different vendor than generated it.
"""

from __future__ import annotations

import hashlib
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

    def assigned_pool(self, lane_key: str, role: str) -> list[TeacherRoute]:
        """Every teacher serving `role` on this lane: the role itself plus any
        `<role>_*` variants, in declared order.

        One entry is the ordinary case. Several means the lane ROTATES, which is
        the roster's stated reason to exist — "no single idiolect dominates". Lane
        A splits easy/hard and lane B splits by extraction role, but lane C's
        character side was a single model, so every one of Sorcha's turns in the
        corpus came from one model's habits. That is a variety ceiling no prompt
        can raise.
        """
        roles = self.assignments.get(lane_key, {})
        ids = [t for name, t in roles.items() if name == role or name.startswith(f"{role}_")]
        if not ids:
            raise KeyError(f"no {role!r} assignment for lane {lane_key!r}")
        return [self.route_for(i) for i in ids]

    def rotate(self, pool: list[TeacherRoute], key: str, *, avoid: str = "") -> TeacherRoute:
        """Pick one of `pool` by hashing `key` — deterministic, so a rerun hits the
        same cache entries. `avoid` excludes a provider: lane C must never put the
        same vendor on both sides of a conversation, which would make the self-play
        a model talking to itself.
        """
        eligible = [r for r in pool if r.provider != avoid] or pool
        bucket = int(hashlib.sha256(f"rotate:{key}".encode()).hexdigest()[:8], 16)
        return eligible[bucket % len(eligible)]

    def judge_candidates(self, generator_id: str) -> list[TeacherRoute]:
        """Every judge-role teacher from a different vendor than the generator,
        in declared order. Empty is a configuration error, not a fallback."""
        gen_provider = self.route_for(generator_id).provider
        return [
            r
            for r in (self.route_for(i) for i in self.assignments["judge"].values())
            if r.provider != gen_provider
        ]

    def judge_for(self, generator_id: str, key: str = "") -> TeacherRoute:
        """Cross-vendor judge, load-balanced across every eligible vendor.

        First-match-in-order made the declaration order a priority list: with
        DeepSeek generating most records, GLM won every time, Kimi was structurally
        unreachable, and one vendor's rate limit became the whole pipeline's
        throughput ceiling (measured: 19 rate-limits per 130 calls). Spreading the
        load is the point.

        `key` picks the judge deterministically rather than randomly, for two
        reasons. Reproducibility — a rerun must route identically or the cached
        responses miss. And DPO integrity: callers pass a record's `sibling_group`,
        so every sibling of one instance lands on the SAME judge. DPO gates on
        `chosen.overall - rejected.overall`, and two vendors do not share a scoring
        scale, so siblings split across judges would manufacture and destroy pairs
        on vendor offset rather than on quality.

        An empty key keeps the old first-match behaviour, so callers that have no
        stable key are still deterministic.
        """
        candidates = self.judge_candidates(generator_id)
        if not candidates:
            raise ValueError(
                f"no cross-vendor judge available for provider "
                f"{self.route_for(generator_id).provider!r}; "
                "add a judge assignment from another vendor"
            )
        if not key:
            return candidates[0]
        bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % len(candidates)
        return candidates[bucket]

    @classmethod
    def load(cls, path: Path) -> Roster:
        return cls.model_validate(yaml.safe_load(path.read_text()))
