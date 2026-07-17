"""Regression tests for verifiers/files/save_note.py: the outcome gate must check
the REQUESTED destination and non-empty content, not merely that some write ok'd.
The fixture-rule test only exercises the three canonical fixtures; these lock the
wrong-path and empty-content failure modes that curated fixtures can't express."""

from __future__ import annotations

from pathlib import Path

from aviary.gates.verify import run_verifier
from aviary.schema.records import ConversationRecord, Message, Provenance, SourceRef, ToolCall

REPO = Path(__file__).resolve().parents[1]
VERIFIER = REPO / "verifiers" / "files" / "save_note.py"


def _record(*, write_path: str, content: str, ok: bool, destination: str) -> ConversationRecord:
    tool_result = '{"ok": true, "bytes": 12}' if ok else '{"error": "EACCES"}'
    return ConversationRecord(
        system="s",
        messages=[
            Message(role="user", content=f"save it to {destination}"),
            Message(
                role="assistant",
                speaker="Olivia",
                content="On it.",
                tool_calls=(ToolCall(id="c1", name="write_file",
                                     arguments={"path": write_path, "content": content}),),
            ),
            Message(role="tool", tool_call_id="c1", content=tool_result),
            Message(role="assistant", speaker="Olivia", content="Saved."),
        ],
        provenance=Provenance(
            record_id="t",
            lane="a",
            run_id="t",
            family="files",
            template_id="files.save_note",
            source=SourceRef(
                kind="task_instance",
                detail={"params": {"kind": "haiku about rain", "destination": destination}},
            ),
        ),
    )


def test_write_to_correct_path_passes():
    rec = _record(write_path="lists/today.txt", content="milk\neggs",
                  ok=True, destination="lists/today.txt")
    assert run_verifier(VERIFIER, rec).passed


def test_write_to_wrong_path_fails():
    # The confirmed bug: content written successfully, but to the wrong place.
    rec = _record(write_path="wrong/place.txt", content="garbage",
                  ok=True, destination="lists/today.txt")
    result = run_verifier(VERIFIER, rec)
    assert not result.passed
    assert "wrote_requested_path" in result.details


def test_empty_content_to_right_path_fails():
    rec = _record(write_path="lists/today.txt", content="   ",
                  ok=True, destination="lists/today.txt")
    result = run_verifier(VERIFIER, rec)
    assert not result.passed
    assert "content_nonempty" in result.details


def test_missing_destination_fails_closed():
    rec = _record(write_path="lists/today.txt", content="milk",
                  ok=True, destination="lists/today.txt")
    rec.provenance.source.detail["params"].pop("destination")
    assert not run_verifier(VERIFIER, rec).passed
