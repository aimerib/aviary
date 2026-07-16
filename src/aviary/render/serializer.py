"""THE choke point. The only module in the repo allowed to emit training-format text.

Output is Swift Parrot ChatML (Qwen-style) with Hermes-format <tool_call> blocks.
The byte format is documented in render/CONTRACT.md and locked by render/goldens/.
Contract version: v2 (thoughts, group chat, NSP, DPO).
"""

from __future__ import annotations

import json
from enum import StrEnum

from aviary.schema.records import ConversationRecord, Message, ToolCall
from aviary.schema.results import RenderedSample

SERIALIZER_CONTRACT = "v2"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

TOOLS_PREAMBLE = (
    "\n\n# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n{tools}\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{{"name": <function-name>, "arguments": <args-json-object>}}\n'
    "</tool_call>"
)


class ThoughtMode(StrEnum):
    WITH = "with_thoughts"
    WITHOUT = "no_thoughts"


def canonical_tool_call_json(tc: ToolCall) -> str:
    """The single json.dumps policy for tool calls. Golden-locked."""
    return json.dumps(
        {"name": tc.name, "arguments": tc.arguments}, ensure_ascii=False, separators=(", ", ": ")
    )


def _system_text(rec: ConversationRecord) -> str:
    text = rec.system
    if rec.tools_schema_json is not None:
        text += TOOLS_PREAMBLE.format(tools=rec.tools_schema_json)
    return text


def _assistant_body(msg: Message, mode: ThoughtMode, speaker_prefix: bool) -> str:
    parts: list[str] = []
    if mode is ThoughtMode.WITH and msg.thought is not None:
        parts.append(f"<think>\n{msg.thought}\n</think>")
    body = msg.content
    if speaker_prefix and msg.speaker:
        body = f"{msg.speaker}: {body}" if body else f"{msg.speaker}:"
    if body:
        parts.append(body)
    for tc in msg.tool_calls:
        parts.append(f"<tool_call>\n{canonical_tool_call_json(tc)}\n</tool_call>")
    return "\n".join(parts)


def _turn(role: str, body: str) -> str:
    return f"{IM_START}{role}\n{body}{IM_END}\n"


def render_conversation(rec: ConversationRecord, mode: ThoughtMode) -> RenderedSample:
    """Render one record to training text. train_spans cover assistant turns
    (body through the closing <|im_end|>), the loss-bearing regions."""
    prefix_speakers = len(rec.speakers) > 1
    out: list[str] = [_turn("system", _system_text(rec))]
    spans: list[tuple[int, int]] = []
    pos = len(out[0])

    for msg in rec.messages:
        if msg.role == "system":
            raise ValueError("system prompt belongs in record.system, not messages")
        if msg.role == "assistant":
            body = _assistant_body(msg, mode, prefix_speakers)
            turn = _turn("assistant", body)
            start = pos + len(IM_START) + len("assistant\n")
            spans.append((start, pos + len(turn) - 1))  # include <|im_end|>, not the newline
        elif msg.role == "tool":
            turn = _turn("tool", f"<tool_response>\n{msg.content}\n</tool_response>")
        else:
            turn = _turn("user", msg.content)
        out.append(turn)
        pos += len(turn)

    text = "".join(out)
    return RenderedSample(
        text=text,
        train_spans=spans,
        meta={
            "record_id": rec.provenance.record_id,
            "lane": rec.provenance.lane,
            "family": rec.provenance.family,
            "thought_mode": mode.value,
            "sample_kind": "conversation",
            "contract": SERIALIZER_CONTRACT,
        },
    )


def render_next_speaker_samples(rec: ConversationRecord) -> list[RenderedSample]:
    """For multi-speaker records: at each assistant turn, predict who speaks next.

    Sample text = conversation prefix + '<|im_start|>assistant\\n<speaker>:' with the
    train span covering only the speaker label.
    """
    if len(rec.speakers) <= 1:
        return []
    samples: list[RenderedSample] = []
    base = _turn("system", _system_text(rec))
    prefix = base
    for i, msg in enumerate(rec.messages):
        if msg.role == "assistant" and msg.speaker and i > 0:
            head = f"{IM_START}assistant\n"
            label = f"{msg.speaker}:"
            samples.append(
                RenderedSample(
                    text=prefix + head + label,
                    train_spans=[(len(prefix) + len(head), len(prefix) + len(head) + len(label))],
                    meta={
                        "record_id": rec.provenance.record_id,
                        "lane": rec.provenance.lane,
                        "family": rec.provenance.family,
                        "thought_mode": ThoughtMode.WITHOUT.value,
                        "sample_kind": "next_speaker",
                        "turn_index": str(i),
                        "contract": SERIALIZER_CONTRACT,
                    },
                )
            )
        if msg.role == "assistant":
            prefix += _turn("assistant", _assistant_body(msg, ThoughtMode.WITHOUT, True))
        elif msg.role == "tool":
            prefix += _turn("tool", f"<tool_response>\n{msg.content}\n</tool_response>")
        else:
            prefix += _turn("user", msg.content)
    return samples
