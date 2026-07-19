"""Build targets: which model this corpus trains, and whose voice it speaks.

A target (datagen/configs/targets/<name>.yaml) binds everything persona- or
model-specific that used to be hardcoded: the persona attached at lane A ingest
and generated-as in lanes A/C, the harmonizer's canonical voice and paraphrase
prompt, the judge's voice rubric, and which lanes a render may include. Swapping
targets swaps the model being built — flash-v2_2 (Olivia) and sorcha-v1 share
one pipeline and never share a record set's voice.

Selection: $AVIARY_TARGET, defaulting to flash-v2_2. The default must reproduce
pre-target behavior byte-for-byte; goldens and the frozen PromptSet hash are the
proof.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from aviary.paths import REPO_ROOT

DEFAULT_TARGET = "flash-v2_2"


class Target(BaseModel):
    name: str
    base_model: str  # recorded in the manifest; both current targets are Qwen3.5-35B-A3B
    persona: str  # datagen/persona/<persona>/ — system.md is the generation identity
    persona_speaker: str  # assistant `speaker` in lanes A/C; the harmonize voice gate
    judge_voice_axis: str  # the quality rubric's voice axis (documentation of intent)
    lanes: list[str]  # record sets a render for this target may include
    quality_rubric: str  # repo-relative; persona-owned (carries the voice axis)
    character_rubric: str = "gates/judge/character_rp.rubric.yaml"  # persona-transparent
    harmonize_prompt: str = ""  # repo-relative; persona-owned paraphrase prompt
    # Per-lane roles whose conversational spans harmonize may touch, further
    # voice-gated per record on persona_speaker. Absent lane -> no harmonization.
    harmonize: dict[str, list[str]] = Field(default_factory=dict)
    # Lane-specific quality rubrics beyond the defaults (e.g. lane D under sorcha).
    lane_rubrics: dict[str, str] = Field(default_factory=dict)

    @property
    def persona_dir(self) -> Path:
        return REPO_ROOT / "datagen" / "persona" / self.persona

    @property
    def persona_system_key(self) -> str:
        """PromptSet key for the persona system prompt. Constructed from the persona
        name so the flash target's key is byte-identical to the pre-target era
        ("olivia_system") — the frozen PromptSet hash is part of the proof."""
        return f"{self.persona}_system"

    @classmethod
    def load(cls, name: str) -> Target:
        path = REPO_ROOT / "datagen" / "configs" / "targets" / f"{name}.yaml"
        if not path.exists():
            known = sorted(p.stem for p in path.parent.glob("*.yaml"))
            raise FileNotFoundError(f"unknown target {name!r}; known targets: {known}")
        return cls.model_validate(yaml.safe_load(path.read_text()))

    @classmethod
    def resolve(cls) -> Target:
        return cls.load(os.environ.get("AVIARY_TARGET", DEFAULT_TARGET))
