"""Span-protected Olivia-voice paraphrase.

Structural protection by construction: only Message.content / Message.thought of
user/assistant turns are candidates, and per policy only assistant turns in lanes
a/c are actually rewritten (lane B character voices are the entropy source — off).
Sub-span protection via mask/restore; any restore failure drops the record
(Gemma-rule scar tissue: never bend the boundary).
"""

from __future__ import annotations

from dataclasses import dataclass

from aviary.gates.spans import extract_protected_spans, mask_spans, restore_spans
from aviary.schema.records import ConversationRecord
from aviary.teacher.client import ChatRequest, TeacherClient
from aviary.teacher.prompts import PromptSet

# lane -> which message roles get paraphrased; lane b intentionally absent
DEFAULT_POLICY: dict[str, tuple[str, ...]] = {"a": ("assistant",), "c": ("assistant",)}


@dataclass
class HarmonizeOutcome:
    record: ConversationRecord  # drop-marked (gate_state.dropped) on span violation
    dropped: bool = False
    rewritten_messages: int = 0


def _paraphrase(
    text: str, client: TeacherClient, prompts: PromptSet, model: str, lane: str
) -> str | None:
    spans = extract_protected_spans(text)
    masked, mapping = mask_spans(text, spans)
    req = ChatRequest(
        model=model,
        system=prompts["harmonize_prompt"],
        messages=[{"role": "user", "content": masked}],
        temperature=0.4,
        max_tokens=2048,
    )
    reply = client.complete(req, lane).text.strip()
    return restore_spans(reply, mapping)


def harmonize_record(
    rec: ConversationRecord,
    client: TeacherClient,
    prompts: PromptSet,
    model: str,
    policy: dict[str, tuple[str, ...]] | None = None,
) -> HarmonizeOutcome:
    roles = (policy or DEFAULT_POLICY).get(rec.provenance.lane, ())
    if not roles:
        return HarmonizeOutcome(record=rec)

    new_messages = []
    rewritten = 0
    for m in rec.messages:
        if m.role not in roles:
            new_messages.append(m)
            continue
        updates: dict[str, str] = {}
        for fieldname in ("content", "thought"):
            original = getattr(m, fieldname)
            if not original:
                continue
            restored = _paraphrase(original, client, prompts, model, rec.provenance.lane)
            if restored is None:
                dropped = rec.model_copy(
                    update={
                        "gate_state": rec.gate_state.model_copy(
                            update={"dropped": True, "drop_reason": "span_violation"}
                        )
                    }
                )
                return HarmonizeOutcome(record=dropped, dropped=True)
            if restored != original:
                updates[fieldname] = restored
        if updates:
            rewritten += 1
            new_messages.append(m.model_copy(update=updates))
        else:
            new_messages.append(m)

    out = rec.model_copy(
        update={
            "messages": new_messages,
            "gate_state": rec.gate_state.model_copy(update={"harmonized": True}),
        }
    )
    return HarmonizeOutcome(record=out, rewritten_messages=rewritten)
