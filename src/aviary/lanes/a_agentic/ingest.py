"""Adapter: hermes batch trajectories.jsonl -> ConversationRecord.

The raw batch record (source-of-truth rule) is ShareGPT-shaped:
{prompt_index, completed, partial, conversations: [{from, value}], toolsets_used,
tool_stats, ...} — and everything structured lives INSIDE the text values
(verified against the pinned checkout's real output, 2026-07-19):

- gpt turns:   optional leading <think>…</think>, prose, then zero or more
               <tool_call>\n{"name": …, "arguments": …}\n</tool_call> blocks.
               Calls carry NO ids.
- tool turns:  <tool_response>\n{"tool_call_id": …, "name": …, "content": …}
               \n</tool_response>. Responses carry the authoritative ids; calls
               get theirs by name-matched, order-preserving pairing.
- system turn: the harness prompt. hermes does NOT persist the ephemeral persona
               prompt (re-attached here from the run's PromptSet), but the
               harness turn DOES carry the authoritative <tools>[…]</tools>
               block — the exact schemas the model saw, which vary per record
               with runtime tool availability. Extracted verbatim.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from aviary.lanes.a_agentic.taskbank import TaskInstance, rollout_order
from aviary.schema.records import (
    ConversationRecord,
    Message,
    Provenance,
    SourceRef,
    ToolCall,
    make_record_id,
)

_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "tool": "tool",
    "observation": "tool",
    "function": "tool",
}

_THINK = re.compile(r"\A\s*<think>\s*(.*?)\s*</think>\s*", re.S)
_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_TOOL_RESPONSE = re.compile(r"<tool_response>\s*(\{.*?\})\s*</tool_response>", re.S)
_TOOLS_BLOCK = re.compile(r"<tools>\s*(\[.*?\])\s*</tools>", re.S)


class IngestError(ValueError):
    pass


@dataclass
class IngestContext:
    run_id: str
    instances: list[TaskInstance]  # in emit_batch_inputs order, one entry PER ROLLOUT LINE
    system_prompt: str
    persona_speaker: str  # the build target's voice; stamped on assistant turns
    teacher_id: str
    hermes_commit: str
    prompt_set_hash: str


@dataclass
class _PendingCall:
    turn: int  # index into parsed messages
    slot: int  # index into that turn's tool_calls
    name: str


def _parse_gpt_turn(value: str, turn_idx: int) -> tuple[str | None, str, list[dict]]:
    """-> (thought, prose content, raw call dicts)."""
    thought = None
    m = _THINK.match(value)
    if m:
        thought = m.group(1) or None
        value = value[m.end() :]
    raw_calls = []
    for cm in _TOOL_CALL.finditer(value):
        try:
            raw_calls.append(json.loads(cm.group(1)))
        except json.JSONDecodeError as e:
            raise IngestError(f"unparseable <tool_call> JSON at turn {turn_idx}: {e}") from e
    content = _TOOL_CALL.sub("", value).strip()
    return thought, content, raw_calls


def _parse_tool_turn(value: str, turn_idx: int) -> list[dict]:
    """One tool turn may carry SEVERAL <tool_response> blocks (parallel calls
    answered together) — verified against real batch output."""
    responses = []
    for m in _TOOL_RESPONSE.finditer(value):
        try:
            resp = json.loads(m.group(1))
        except json.JSONDecodeError as e:
            raise IngestError(f"unparseable <tool_response> JSON at turn {turn_idx}: {e}") from e
        if "tool_call_id" not in resp or "name" not in resp:
            raise IngestError(f"tool response at turn {turn_idx} missing tool_call_id/name")
        responses.append(resp)
    if not responses:
        raise IngestError(f"tool turn {turn_idx} without <tool_response> wrapper")
    return responses


def _result_text(content) -> str:
    """The result payload as the tool message's content. Dict payloads serialize to
    JSON (the verifiers' no-error convention parses this); strings pass through."""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def extract_tools_schema(raw: dict) -> str | None:
    """The verbatim <tools> JSON array from the recorded harness system turn — the
    exact schemas the model saw (they vary per record with runtime availability)."""
    for turn in raw.get("conversations", []):
        if _ROLE_MAP.get(turn.get("from", "")) is None and turn.get("from") != "system":
            continue
        if turn.get("from") == "system":
            m = _TOOLS_BLOCK.search(turn.get("value", "") or "")
            return m.group(1) if m else None
    return None


def ingest_hermes_record(raw: dict, ctx: IngestContext) -> ConversationRecord:
    if raw.get("partial") or raw.get("completed") is False:
        raise IngestError("incomplete trajectory (partial/failed rollout)")
    prompt_index = raw["prompt_index"]
    lines = rollout_order(ctx.instances)
    try:
        inst = lines[prompt_index]
    except IndexError as e:
        raise IngestError(f"prompt_index {prompt_index} outside emitted inputs") from e

    messages: list[Message] = []
    pending: list[_PendingCall] = []
    calls_by_turn: dict[int, list[ToolCall]] = {}
    for i, turn in enumerate(raw.get("conversations", [])):
        src_role = turn.get("from", "")
        if src_role == "system":
            continue  # harness prompt; persona prompt is re-attached below
        role = _ROLE_MAP.get(src_role)
        if role is None:
            raise IngestError(f"unknown role {src_role!r} at turn {i}")
        value = turn.get("value", "") or ""

        if role == "assistant":
            thought, content, raw_calls = _parse_gpt_turn(value, i)
            slot_base = len(messages)
            turn_calls: list[ToolCall] = []
            for j, rc in enumerate(raw_calls):
                name = rc.get("name")
                if not name:
                    raise IngestError(f"tool call without name at turn {i}")
                args = rc.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args) if args.strip() else {}
                # Id is assigned when the matching response arrives; placeholder
                # survives only for dangling calls (max_turns cutoffs).
                turn_calls.append(ToolCall(id=f"call_{i}_{j}", name=name, arguments=args))
                pending.append(_PendingCall(turn=slot_base, slot=j, name=name))
            calls_by_turn[slot_base] = turn_calls
            messages.append(
                Message(
                    role="assistant",
                    speaker=ctx.persona_speaker,
                    content=content,
                    thought=thought,
                    tool_calls=tuple(turn_calls),
                )
            )
        elif role == "tool":
            for resp in _parse_tool_turn(value, i):
                match_idx = next(
                    (k for k, pc in enumerate(pending) if pc.name == resp["name"]), None
                )
                if match_idx is None:
                    raise IngestError(
                        f"tool response for {resp['name']!r} at turn {i} with no open call"
                    )
                pc = pending.pop(match_idx)
                call_id = str(resp["tool_call_id"])
                calls_by_turn[pc.turn][pc.slot] = ToolCall(
                    id=call_id,
                    name=pc.name,
                    arguments=calls_by_turn[pc.turn][pc.slot].arguments,
                )
                # Rebuild the assistant message with the authoritative id.
                messages[pc.turn] = messages[pc.turn].model_copy(
                    update={"tool_calls": tuple(calls_by_turn[pc.turn])}
                )
                messages.append(
                    Message(
                        role="tool", tool_call_id=call_id, content=_result_text(resp["content"])
                    )
                )
        else:
            messages.append(Message(role="user", content=value))

    source = SourceRef(
        kind="task_instance",
        detail={
            "template_id": inst.template_id,
            "params": inst.params,
            "prompt_index": prompt_index,
            "toolsets_used": raw.get("toolsets_used", []),
            "tool_stats": raw.get("tool_stats", {}),
        },
    )
    return ConversationRecord(
        system=ctx.system_prompt,
        tools_schema_json=extract_tools_schema(raw),
        messages=messages,
        provenance=Provenance(
            record_id=make_record_id("a", ctx.run_id, source, salt=str(prompt_index)),
            lane="a",
            run_id=ctx.run_id,
            family=inst.family,
            template_id=inst.template_id,
            holdout=inst.holdout,
            teachers={"assistant": ctx.teacher_id},
            sibling_group=f"{inst.instance_key()}",
            hermes_commit=ctx.hermes_commit,
            prompt_set_hash=ctx.prompt_set_hash,
            source=source,
        ),
    )
