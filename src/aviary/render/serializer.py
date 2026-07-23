"""THE choke point. The only module in the repo allowed to emit training-format text.

Output is Swift Parrot ChatML: the Qwen3.5 chat template, byte-for-byte, with XML
`<tool_call><function=…><parameter=…>` blocks. The byte format is documented in
render/CONTRACT.md and locked by render/goldens/.
Contract version: v3 (Qwen3.5-native tool calls, tool responses, think placement).

The authority for this format is the `chat_template` in Qwen3.5-35B-A3B's
tokenizer_config.json. `tests/test_template_parity.py` renders our fixtures through
that jinja and asserts byte-equality — that test, not this docstring, is the proof.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from enum import StrEnum

from aviary.schema.records import ConversationRecord, Message, ToolCall
from aviary.schema.results import RenderedSample

SERIALIZER_CONTRACT = "v3"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# Verbatim from the Qwen3.5 template. The tools block opens the system turn and the
# record's own system text is appended AFTER it (v2 had this backwards).
TOOLS_HEADER = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
TOOLS_FOOTER = (
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format "
    "with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n"
    "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
    "that can span\nmultiple lines\n</parameter>\n"
    "</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> "
    "block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE "
    "the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your "
    "current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>"
)


class ThoughtMode(StrEnum):
    WITH = "with_thoughts"
    WITHOUT = "no_thoughts"


def _tojson(value: object) -> str:
    """transformers' `tojson` filter: json.dumps(value, ensure_ascii=False).

    Note this is NOT Jinja's stock tojson, which HTML-escapes; transformers
    overrides it precisely so chat templates round-trip text unmangled.
    """
    return json.dumps(value, ensure_ascii=False)


def _arg_value(value: object) -> str:
    """The template's argument-value rule:

        args_value | tojson  if mapping or (sequence and not string)
        args_value | string  otherwise

    Jinja's `string` filter is Python `str()`, so bools render `True`/`False` and
    None renders `None` — matched here deliberately, not by accident.
    """
    if isinstance(value, dict | list | tuple):
        return _tojson(value)
    return str(value)


def render_tool_call(tc: ToolCall) -> str:
    """One `<tool_call>` block in Qwen3.5 XML. Golden-locked.

    Argument key order is preserved as recorded — the template iterates the mapping
    in insertion order and so do we.
    """
    parts = [f"<tool_call>\n<function={tc.name}>\n"]
    for name, value in tc.arguments.items():
        parts.append(f"<parameter={name}>\n{_arg_value(value)}\n</parameter>\n")
    parts.append("</function>\n</tool_call>")
    return "".join(parts)


def _wrap_tool(tool: dict) -> dict:
    """OpenAI function-tool envelope.

    The template dumps whatever the caller passes in `tools`. Every
    OpenAI-compatible server (vLLM, SGLang, llama.cpp) forwards the request's
    `tools` array verbatim, and that array is OpenAI-shaped. Our toolsets store the
    bare function object, so we wrap at render time to match what the model will
    actually see at inference.
    """
    return tool if tool.get("type") == "function" else {"type": "function", "function": tool}


def _system_turn(rec: ConversationRecord) -> str:
    system = (rec.system or "").strip()
    if rec.tools_schema_json is not None:
        tools = json.loads(rec.tools_schema_json)
        body = TOOLS_HEADER
        for tool in tools:
            body += "\n" + _tojson(_wrap_tool(tool))
        body += TOOLS_FOOTER
        if system:
            body += "\n\n" + system
        return _turn("system", body)
    return _turn("system", system) if system else ""


def think_positions(rec: ConversationRecord) -> set[int]:
    """Message indices whose assistant turn carries a `<think>` block.

    Mirrors the template's `last_query_index` rule: only assistant turns after the
    last real user query keep their reasoning; earlier ones render bare, because
    that is exactly what the model will see in its own context at inference.

    Records with no user turn at all are a training-only shape the stock template
    refuses to render ("No user query found in messages") — lane B group-chat
    scenes. There, every assistant turn keeps its thought: the interiority IS the
    lane's payload, and there is no inference-time context shape to match.
    """
    assistants = {i for i, m in enumerate(rec.messages) if m.role == "assistant"}
    users = [i for i, m in enumerate(rec.messages) if m.role == "user"]
    if not users:
        return assistants
    return {i for i in assistants if i > users[-1]}


def _assistant_body(msg: Message, mode: ThoughtMode, thinks: bool, speaker_prefix: bool) -> str:
    content = (msg.content or "").strip()
    if speaker_prefix and msg.speaker:
        content = f"{msg.speaker}: {content}" if content else f"{msg.speaker}:"

    out = ""
    if thinks:
        # WITHOUT mode is not "no block" — it is the template's empty-reasoning form,
        # identical to what `enable_thinking=false` primes at inference.
        thought = (msg.thought or "").strip() if mode is ThoughtMode.WITH else ""
        out += f"<think>\n{thought}\n</think>\n\n"
    out += content

    for i, tc in enumerate(msg.tool_calls):
        if i == 0:
            out += ("\n\n" if content else "") + render_tool_call(tc)
        else:
            out += "\n" + render_tool_call(tc)
    return out


def _turn(role: str, body: str) -> str:
    return f"{IM_START}{role}\n{body}{IM_END}\n"


def _tool_response_turn(msgs: list[Message]) -> str:
    """Consecutive tool results collapse into ONE `user` turn — the template groups
    them, and a per-result turn would not match anything the model ever sees."""
    body = "".join(
        f"\n<tool_response>\n{(m.content or '').strip()}\n</tool_response>" for m in msgs
    )
    return f"{IM_START}user{body}{IM_END}\n"


def _turns(rec: ConversationRecord, mode: ThoughtMode) -> Iterator[tuple[str, int | None]]:
    """(turn_text, assistant_message_index or None) for every turn after the system.

    Shared by conversation rendering and NSP prefix construction so the two can
    never drift.
    """
    prefix_speakers = len(rec.speakers) > 1
    thinks = think_positions(rec)
    msgs = rec.messages
    i = 0
    while i < len(msgs):
        msg = msgs[i]
        if msg.role == "system":
            raise ValueError("system prompt belongs in record.system, not messages")
        if msg.role == "tool":
            run = i
            while run < len(msgs) and msgs[run].role == "tool":
                run += 1
            yield _tool_response_turn(msgs[i:run]), None
            i = run
            continue
        if msg.role == "assistant":
            yield _turn("assistant", _assistant_body(msg, mode, i in thinks, prefix_speakers)), i
        else:
            yield _turn("user", (msg.content or "").strip()), None
        i += 1


def render_conversation(rec: ConversationRecord, mode: ThoughtMode) -> RenderedSample:
    """Render one record to training text. train_spans cover assistant turns
    (body through the closing <|im_end|>), the loss-bearing regions."""
    head = _system_turn(rec)
    out: list[str] = [head] if head else []
    spans: list[tuple[int, int]] = []
    pos = len(head)

    for turn, assistant_index in _turns(rec, mode):
        if assistant_index is not None:
            start = pos + len(IM_START) + len("assistant\n")
            spans.append((start, pos + len(turn) - 1))  # include <|im_end|>, not the newline
        out.append(turn)
        pos += len(turn)

    return RenderedSample(
        text="".join(out),
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

    Sample text = conversation prefix + the assistant turn opening (including the
    empty-think block wherever the main artifact would carry one) + '<speaker>:',
    with the train span covering only the speaker label.
    """
    if len(rec.speakers) <= 1:
        return []
    thinks = think_positions(rec)
    samples: list[RenderedSample] = []
    prefix = _system_turn(rec)

    for turn, assistant_index in _turns(rec, ThoughtMode.WITHOUT):
        msg = rec.messages[assistant_index] if assistant_index is not None else None
        if msg is not None and msg.speaker and assistant_index > 0:
            head = f"{IM_START}assistant\n"
            if assistant_index in thinks:
                head += "<think>\n\n</think>\n\n"
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
                        "turn_index": str(assistant_index),
                        "contract": SERIALIZER_CONTRACT,
                    },
                )
            )
        prefix += turn
    return samples
