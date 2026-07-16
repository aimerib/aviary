"""Adapter: hermes batch trajectories.jsonl -> ConversationRecord.

The raw batch record (source-of-truth rule) is ShareGPT-shaped with metadata:
{prompt_index, conversations: [{from, value, tool_calls?}, ...], toolsets_used,
tool_stats}. hermes does NOT persist the ephemeral system prompt, so the frozen
Olivia system prompt is re-attached here from the run's PromptSet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from aviary.lanes.a_agentic.taskbank import TaskInstance
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


class IngestError(ValueError):
    pass


@dataclass
class IngestContext:
    run_id: str
    instances: list[TaskInstance]  # in emit_batch_inputs order, one entry PER ROLLOUT LINE
    system_prompt: str
    tools_schema_by_family: dict[str, str]  # family -> canonical <tools> JSON
    teacher_id: str
    hermes_commit: str
    prompt_set_hash: str


def flatten_rollout_lines(instances: list[TaskInstance]) -> list[TaskInstance]:
    """Mirror emit_batch_inputs: index i of the output = instance behind prompt_index i."""
    return [inst for inst in instances for _ in range(inst.n_rollouts)]


def _parse_tool_calls(raw_calls: list, turn_idx: int) -> tuple[ToolCall, ...]:
    calls = []
    for j, raw in enumerate(raw_calls):
        fn = raw.get("function", raw)
        name = fn.get("name")
        if not name:
            raise IngestError(f"tool call without name at turn {turn_idx}")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args) if args.strip() else {}
        calls.append(
            ToolCall(id=raw.get("id") or f"call_{turn_idx}_{j}", name=name, arguments=args)
        )
    return tuple(calls)


def ingest_hermes_record(raw: dict, ctx: IngestContext) -> ConversationRecord:
    prompt_index = raw["prompt_index"]
    lines = flatten_rollout_lines(ctx.instances)
    try:
        inst = lines[prompt_index]
    except IndexError as e:
        raise IngestError(f"prompt_index {prompt_index} outside emitted inputs") from e

    messages: list[Message] = []
    pending_call_ids: list[str] = []
    for i, turn in enumerate(raw.get("conversations", [])):
        role = _ROLE_MAP.get(turn.get("from", ""))
        if role is None:
            raise IngestError(f"unknown role {turn.get('from')!r} at turn {i}")
        value = turn.get("value", "") or ""
        if role == "assistant":
            calls = _parse_tool_calls(turn.get("tool_calls") or [], i)
            pending_call_ids.extend(tc.id for tc in calls)
            messages.append(
                Message(role="assistant", speaker="Olivia", content=value, tool_calls=calls)
            )
        elif role == "tool":
            call_id = pending_call_ids.pop(0) if pending_call_ids else f"call_orphan_{i}"
            messages.append(Message(role="tool", tool_call_id=call_id, content=value))
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
    tools_json = ctx.tools_schema_by_family.get(inst.family)
    if tools_json is None:
        raise IngestError(f"no toolset schema for family {inst.family!r} in datagen/toolsets/")
    return ConversationRecord(
        system=ctx.system_prompt,
        tools_schema_json=tools_json,
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
