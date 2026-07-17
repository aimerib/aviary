"""call_json repair-loop robustness — regression for the empty-reply crash.

A reasoning teacher (e.g. Kimi) can return empty content. The repair loop must not
echo that back as an empty assistant message (strict providers 400 on it), and
persistent unparseable replies must surface as a *skippable* ExtractionError, not a
run-killing TeacherError.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from aviary.lanes.common import ExtractionError, call_json
from aviary.teacher.client import ChatRequest
from aviary.teacher.fake import FakeTeacherClient


class Obj(BaseModel):
    ok: bool


def _req() -> ChatRequest:
    return ChatRequest(
        model="deepseek-v4-flash-20260717",
        system="sys",
        messages=[{"role": "user", "content": "extract"}],
    )


def _assert_no_empty_assistant(client: FakeTeacherClient) -> None:
    for r in client.requests:
        for m in r.messages:
            if m.get("role") == "assistant":
                assert m.get("content", "").strip(), "empty assistant message sent to provider"


def test_repair_recovers_after_empty_reply():
    # First reply empty, then valid: must recover without ever sending an empty
    # assistant message.
    replies = iter(["", '{"ok": true}'])

    def script(req: ChatRequest) -> str:
        return next(replies)

    client = FakeTeacherClient(script=script)
    out = call_json(client, _req(), Obj, lane="b")
    assert out.ok is True
    _assert_no_empty_assistant(client)


def test_persistent_empty_raises_skippable_extraction_error():
    client = FakeTeacherClient(script=lambda req: "")  # always empty
    with pytest.raises(ExtractionError):  # NOT TeacherError — pipeline drops the scene
        call_json(client, _req(), Obj, lane="b", max_repairs=2)
    _assert_no_empty_assistant(client)


def test_nonempty_invalid_reply_is_echoed_back():
    # A non-empty but wrong reply SHOULD be echoed so the model sees its mistake.
    replies = iter(["not json", '{"ok": false}'])
    client = FakeTeacherClient(script=lambda req: next(replies))
    out = call_json(client, _req(), Obj, lane="b")
    assert out.ok is False
    # the second request must contain the echoed (non-empty) assistant turn
    assert any(
        m.get("role") == "assistant" and m.get("content") == "not json"
        for m in client.requests[1].messages
    )
