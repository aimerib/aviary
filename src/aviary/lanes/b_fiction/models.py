"""Intermediate stage models for lane B (checkpointed as JSONL between stages)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CharacterProfile(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    personality: str = ""
    speech_style: str = ""
    relationships: dict[str, str] = Field(default_factory=dict)

    def matches(self, name: str) -> bool:
        needle = name.strip().lower()
        return needle == self.name.lower() or needle in (a.lower() for a in self.aliases)


class ProfileSet(BaseModel):
    work_id: str
    characters: list[CharacterProfile]

    def resolve(self, name: str) -> CharacterProfile | None:
        for c in self.characters:
            if c.matches(name):
                return c
        return None


class SceneSetup(BaseModel):
    scene_idx: int = 0
    chunk_idx: int = 0
    setting: str
    participants: list[str]
    context: str  # plot context, refined to not leak the dialogue itself


class SceneList(BaseModel):
    scenes: list[SceneSetup]


class SceneTurn(BaseModel):
    speaker: str
    content: str  # utterance + (action beats)
    thought: str | None = None  # inner monologue at that moment (ToM payload)


class ExtractedScene(BaseModel):
    work_id: str
    setup: SceneSetup
    turns: list[SceneTurn]


class DialogueResult(BaseModel):
    turns: list[SceneTurn]
