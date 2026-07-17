"""Unified conversation record schema.

Every lane's adapter normalizes into ConversationRecord; everything downstream
(gates, renderer) consumes only this. Span protection is structural: the only
fields a paraphraser may ever see are Message.content and Message.thought on
user/assistant turns. `system`, `tools_schema_json`, tool calls, and tool-role
content are immutable by construction.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["system", "user", "assistant", "tool"]
Lane = Literal["a", "b", "c"]

SCHEMA_VERSION = 1


class ToolCall(BaseModel):
    """Immutable: name and arguments must survive every gate byte-for-byte."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict[str, Any]


class Message(BaseModel):
    role: Role
    speaker: str | None = None
    content: str = ""
    thought: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> Message:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool message requires tool_call_id")
        if self.role != "tool" and self.tool_call_id:
            raise ValueError("tool_call_id only valid on tool messages")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only assistant messages may carry tool_calls")
        if self.thought is not None and self.role != "assistant":
            raise ValueError("only assistant messages may carry a thought")
        return self


class SourceRef(BaseModel):
    kind: Literal["task_instance", "book_scene", "selfplay_seed"]
    detail: dict[str, Any] = Field(default_factory=dict)


class Provenance(BaseModel):
    record_id: str
    lane: Lane
    run_id: str
    family: str
    template_id: str | None = None
    holdout: bool = False
    teachers: dict[str, str] = Field(default_factory=dict)
    sibling_group: str | None = None
    hermes_commit: str | None = None
    prompt_set_hash: str = ""
    source: SourceRef


class JudgeScores(BaseModel):
    axes: dict[str, float]
    overall: float
    passed: bool
    judge_model: str


DropReason = Literal[
    "verify", "judge", "scrub", "dedupe", "span_violation", "structural", "degenerate", "error"
]


class GateState(BaseModel):
    verified: bool | None = None
    verifier_id: str | None = None
    verifier_details: dict[str, Any] = Field(default_factory=dict)
    judge: JudgeScores | None = None
    scrub_flags: list[str] = Field(default_factory=list)
    deduped_against: str | None = None
    harmonized: bool = False
    dropped: bool = False
    drop_reason: DropReason | None = None


class ConversationRecord(BaseModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    system: str
    tools_schema_json: str | None = None
    messages: list[Message]
    provenance: Provenance
    gate_state: GateState = Field(default_factory=GateState)

    @property
    def speakers(self) -> list[str]:
        """Distinct non-user speakers, in first-appearance order."""
        seen: dict[str, None] = {}
        for m in self.messages:
            if m.role == "assistant" and m.speaker:
                seen.setdefault(m.speaker, None)
        return list(seen)

    def sibling_key(self) -> str | None:
        return self.provenance.sibling_group


def make_record_id(lane: Lane, run_id: str, source: SourceRef, salt: str = "") -> str:
    payload = json.dumps(
        {"lane": lane, "run": run_id, "source": source.model_dump(), "salt": salt},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
