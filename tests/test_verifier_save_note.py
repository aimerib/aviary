"""Regression tests for verifiers/files/save_note.py: the outcome gate must check
the REQUESTED destination and non-empty content, not merely that some write ok'd —
and (v2) that the index file was written and actually names the saved file.
The fixture-rule test only exercises the three canonical fixtures; these lock the
wrong-path / empty-content / index failure modes that curated fixtures can't express."""

from __future__ import annotations

from pathlib import Path

from aviary.gates.verify import run_verifier
from aviary.schema.records import ConversationRecord, Message, Provenance, SourceRef, ToolCall

REPO = Path(__file__).resolve().parents[1]
VERIFIER = REPO / "verifiers" / "files" / "save_note.py"


def _record(
    *,
    write_path: str,
    content: str,
    ok: bool,
    destination: str,
    index: str | None = "lists/today.index.txt",
    index_content: str | None = None,
    index_ok: bool = True,
) -> ConversationRecord:
    tool_result = '{"ok": true, "bytes": 12}' if ok else '{"error": "EACCES"}'
    params: dict = {"kind": "haiku about rain", "destination": destination}
    messages = [
        Message(role="user", content=f"save it to {destination}"),
        Message(
            role="assistant",
            speaker="Olivia",
            content="On it.",
            tool_calls=(
                ToolCall(
                    id="c1", name="write_file", arguments={"path": write_path, "content": content}
                ),
            ),
        ),
        Message(role="tool", tool_call_id="c1", content=tool_result),
    ]
    if index is not None:
        params["index"] = index
        if index_content is None:
            index_content = f"saved {Path(destination).name} -> {destination}"
        idx_result = '{"bytes_written": 9}' if index_ok else '{"error": "EACCES"}'
        messages += [
            Message(
                role="assistant",
                speaker="Olivia",
                content="",
                tool_calls=(
                    ToolCall(
                        id="c2",
                        name="write_file",
                        arguments={"path": index, "content": index_content},
                    ),
                ),
            ),
            Message(role="tool", tool_call_id="c2", content=idx_result),
        ]
    messages.append(Message(role="assistant", speaker="Olivia", content="Saved."))
    return ConversationRecord(
        system="s",
        messages=messages,
        provenance=Provenance(
            record_id="t",
            lane="a",
            run_id="t",
            family="files",
            template_id="files.save_note",
            source=SourceRef(kind="task_instance", detail={"params": params}),
        ),
    )


def test_write_to_correct_path_passes():
    rec = _record(
        write_path="lists/today.txt", content="milk\neggs", ok=True, destination="lists/today.txt"
    )
    assert run_verifier(VERIFIER, rec).passed


def test_write_to_wrong_path_fails():
    # The confirmed bug: content written successfully, but to the wrong place.
    rec = _record(
        write_path="wrong/place.txt", content="garbage", ok=True, destination="lists/today.txt"
    )
    result = run_verifier(VERIFIER, rec)
    assert not result.passed
    assert "wrote_requested_path" in result.details


def test_empty_content_to_right_path_fails():
    rec = _record(
        write_path="lists/today.txt", content="   ", ok=True, destination="lists/today.txt"
    )
    result = run_verifier(VERIFIER, rec)
    assert not result.passed
    assert "content_nonempty" in result.details


def test_missing_destination_fails_closed():
    rec = _record(
        write_path="lists/today.txt", content="milk", ok=True, destination="lists/today.txt"
    )
    rec.provenance.source.detail["params"].pop("destination")
    assert not run_verifier(VERIFIER, rec).passed


def test_skipped_index_fails():
    # v2 coordination gate: the note alone is not the outcome the user asked for.
    rec = _record(
        write_path="lists/today.txt",
        content="milk",
        ok=True,
        destination="lists/today.txt",
        index=None,
    )
    rec.provenance.source.detail["params"]["index"] = "lists/today.index.txt"
    result = run_verifier(VERIFIER, rec)
    assert not result.passed
    assert "index_written" in result.details


def test_index_not_naming_file_fails():
    rec = _record(
        write_path="lists/today.txt",
        content="milk",
        ok=True,
        destination="lists/today.txt",
        index_content="saved a thing somewhere",
    )
    result = run_verifier(VERIFIER, rec)
    assert not result.passed
    assert "index_names_file" in result.details
