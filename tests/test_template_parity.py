"""Contract v3's actual proof: our serializer output is byte-identical to what
Qwen3.5's own chat template produces for the same conversation.

`render/goldens/` locks what we emit; this locks that what we emit is *right*.
A golden can be regenerated from wrong code and still look self-consistent — the
only external authority is the `chat_template` shipped in the base model's
tokenizer_config.json, vendored at tests/fixtures/chat_template/qwen3_5.jinja.

Offline by construction: the template is a tracked fixture, jinja2 is a dev
dependency, and nothing here touches the network (the autouse socket ban proves it).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

jinja2 = pytest.importorskip("jinja2")
jinja2_sandbox = pytest.importorskip("jinja2.sandbox")

from aviary.render.serializer import (  # noqa: E402
    ThoughtMode,
    _wrap_tool,
    render_conversation,
    think_positions,
)
from aviary.schema.records import ConversationRecord  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REC_DIR = REPO / "tests" / "fixtures" / "records"
TEMPLATE = REPO / "tests" / "fixtures" / "chat_template" / "qwen3_5.jinja"


def _env() -> jinja2.Environment:
    """Reproduce how transformers compiles a chat template: whitespace control on,
    and `tojson` overridden so it does NOT HTML-escape."""

    def raise_exception(message: str):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(value, indent=None):
        return json.dumps(value, ensure_ascii=False, indent=indent)

    env = jinja2_sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


def as_template_messages(rec: ConversationRecord, mode: ThoughtMode) -> tuple[list[dict], list]:
    """The honest record -> chat-template input mapping.

    This is the part of the test that could lie, so it does the minimum: roles pass
    through, the speaker prefix (our own group-chat construct) is folded into
    content, and `reasoning_content` is supplied exactly where think_positions says
    a block belongs — empty string in WITHOUT mode, which is the template's own
    reasoning-off form.
    """
    prefix_speakers = len(rec.speakers) > 1
    thinks = think_positions(rec)
    out: list[dict] = []
    if (rec.system or "").strip() or rec.tools_schema_json is not None:
        out.append({"role": "system", "content": rec.system or ""})

    for i, msg in enumerate(rec.messages):
        if msg.role == "assistant":
            content = (msg.content or "").strip()
            if prefix_speakers and msg.speaker:
                content = f"{msg.speaker}: {content}" if content else f"{msg.speaker}:"
            entry: dict = {"role": "assistant", "content": content}
            if i in thinks:
                entry["reasoning_content"] = (
                    (msg.thought or "").strip() if mode is ThoughtMode.WITH else ""
                )
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {"name": tc.name, "arguments": dict(tc.arguments)} for tc in msg.tool_calls
                ]
            out.append(entry)
        else:
            out.append({"role": msg.role, "content": msg.content or ""})

    tools = (
        [_wrap_tool(t) for t in json.loads(rec.tools_schema_json)]
        if rec.tools_schema_json is not None
        else None
    )
    return out, tools


def render_with_stock_template(rec: ConversationRecord, mode: ThoughtMode) -> str:
    messages, tools = as_template_messages(rec, mode)
    template = _env().from_string(TEMPLATE.read_text())
    return template.render(messages=messages, tools=tools, add_generation_prompt=False)


def load(name: str) -> ConversationRecord:
    return ConversationRecord.model_validate_json((REC_DIR / f"{name}.record.json").read_text())


def has_user_turn(rec: ConversationRecord) -> bool:
    return any(m.role == "user" for m in rec.messages)


ALL_FIXTURES = sorted(p.name.removesuffix(".record.json") for p in REC_DIR.glob("*.record.json"))
CASES = [
    (name, mode)
    for name in ALL_FIXTURES
    for mode in (ThoughtMode.WITH, ThoughtMode.WITHOUT)
    if has_user_turn(load(name))
]


@pytest.mark.parametrize("name,mode", CASES, ids=lambda v: getattr(v, "value", v))
def test_serializer_matches_the_stock_qwen_template(name: str, mode: ThoughtMode):
    rec = load(name)
    ours = render_conversation(rec, mode).text
    theirs = render_with_stock_template(rec, mode)
    assert ours == theirs


def test_there_are_cases_with_tools_and_with_thoughts():
    # Guards against the parametrization silently degenerating to trivial records.
    assert any(load(n).tools_schema_json is not None for n, _ in CASES), "no tool-using case"
    assert any(
        any(m.thought for m in load(n).messages) for n, m_ in CASES if m_ is ThoughtMode.WITH
    ), "no thought-bearing case"


def test_group_chat_scenes_are_a_training_only_shape():
    """Lane B scenes have no user turn, so the stock template refuses them outright.
    That is why think_positions falls back to 'every assistant turn' there: there is
    no inference-time context shape to be faithful to."""
    rec = load("scene_thoughts")
    assert not has_user_turn(rec)
    with pytest.raises(jinja2.exceptions.TemplateError, match="No user query"):
        render_with_stock_template(rec, ThoughtMode.WITH)
    assert think_positions(rec) == {i for i, m in enumerate(rec.messages) if m.role == "assistant"}


def test_tool_results_render_as_one_grouped_user_turn():
    # The v2 bug this replaces: tool results as `tool`-role turns, which the model
    # would never see at inference.
    rec = load("recovered_error")
    text = render_conversation(rec, ThoughtMode.WITH).text
    assert "<|im_start|>tool" not in text
    assert "<|im_start|>user\n<tool_response>" in text


def test_tool_calls_are_xml_not_json():
    rec = load("single_tool")
    text = render_conversation(rec, ThoughtMode.WITH).text
    assert "<tool_call>\n<function=" in text
    assert "<parameter=" in text
    assert '<tool_call>\n{"name"' not in text
