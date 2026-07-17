"""LLM rubric judge. Cross-vendor (judge vendor != generator vendor). Rejected
records are retained, not deleted — they are the DPO source."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from aviary.lanes.common import call_json
from aviary.schema.records import ConversationRecord, JudgeScores
from aviary.teacher.client import ChatRequest, TeacherClient
from aviary.teacher.prompts import PromptSet


class RubricAxis(BaseModel):
    weight: float = 1.0
    min: float | None = None
    description: str = ""


class Rubric(BaseModel):
    name: str
    axes: dict[str, RubricAxis]
    threshold: float

    @classmethod
    def load(cls, path: Path) -> Rubric:
        return cls.model_validate(yaml.safe_load(path.read_text()))


class _JudgeReply(BaseModel):
    scores: dict[str, float]
    rationale: str = ""


class JudgeTranscriptError(ValueError):
    pass


def judge_transcript(rec: ConversationRecord) -> str:
    """Human-readable transcript for the judge. NOT training format (choke-point rule)."""
    lines = [f"[system]\n{rec.system}\n"]
    for m in rec.messages:
        if m.role == "tool":
            lines.append(f"[tool result]\n{m.content[:2000]}\n")
            continue
        who = m.speaker or m.role
        body = m.content
        if m.tool_calls:
            calls = "; ".join(f"{tc.name}({tc.arguments})" for tc in m.tool_calls)
            body = f"{body}\n[calls tools: {calls}]" if body else f"[calls tools: {calls}]"
        if m.thought:
            body = f"(inner thought: {m.thought})\n{body}"
        lines.append(f"[{who}]\n{body}\n")
    return "\n".join(lines)


def judge_record(
    rec: ConversationRecord,
    rubric: Rubric,
    client: TeacherClient,
    judge_model: str,
    prompts: PromptSet,
) -> JudgeScores:
    axes_desc = "\n".join(
        f"- {name}: {axis.description} (1-5)" for name, axis in rubric.axes.items()
    )
    req = ChatRequest(
        model=judge_model,
        system=prompts["judge_prompt"].format(axes=axes_desc),
        messages=[{"role": "user", "content": judge_transcript(rec)}],
        temperature=0.0,
        # Judges are reasoning models (e.g. GLM) kept in thinking mode ON PURPOSE —
        # rubric scoring benefits from deliberation. The verdict JSON is small, but
        # reasoning tokens come out of this budget first, so on long transcripts 1024
        # was exhausted by reasoning and returned empty content (record dropped as a
        # judge error). 8192 gives reasoning + verdict room. (2026-07-17)
        max_tokens=8192,
    )
    reply = call_json(client, req, _JudgeReply, lane=rec.provenance.lane)

    missing = set(rubric.axes) - set(reply.scores)
    if missing:
        raise JudgeTranscriptError(f"judge omitted axes: {sorted(missing)}")
    scores = {k: max(1.0, min(5.0, float(v))) for k, v in reply.scores.items() if k in rubric.axes}

    total_weight = sum(a.weight for a in rubric.axes.values())
    overall = sum(scores[k] * rubric.axes[k].weight for k in rubric.axes) / total_weight
    passed = overall >= rubric.threshold and all(
        axis.min is None or scores[name] >= axis.min for name, axis in rubric.axes.items()
    )
    return JudgeScores(
        axes=scores, overall=round(overall, 3), passed=passed, judge_model=judge_model
    )
