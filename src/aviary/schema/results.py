"""Result types produced by gates and the renderer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SampleKind = Literal["conversation", "next_speaker"]


class VerifierResult(BaseModel):
    passed: bool
    verifier_id: str
    details: dict[str, Any] = Field(default_factory=dict)


class RenderedSample(BaseModel):
    text: str
    train_spans: list[tuple[int, int]]
    meta: dict[str, str] = Field(default_factory=dict)


class RenderedPair(BaseModel):
    prompt_text: str
    chosen_text: str
    rejected_text: str
    meta: dict[str, str] = Field(default_factory=dict)
