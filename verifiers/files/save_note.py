"""Verifier for files.save_note: outcome gate on the FINAL file state.

Passes iff BOTH files the user asked for end in successful writes — the note at
the `destination` param with non-empty content, and the `index` param whose
content names the saved file — and the final assistant turn confirms it. Path
taken is irrelevant: a failed write followed by a successful retry passes
(recovered-failure trajectories are prime data).

Scope: this gates that something substantive reached the RIGHT paths and that
the index actually references the note. Whether the content is genuinely a
haiku / a five-item list / etc. is a semantic judgement and belongs to the
judge stage, not a pure verifier.
"""

import json
from pathlib import PurePosixPath

from aviary.schema.records import ConversationRecord
from aviary.schema.results import VerifierResult

VERIFIER_ID = "files/save_note"


def _is_destination(path: str, destination: str) -> bool:
    """Outcome-equivalent path identity: teachers legitimately write the requested
    relative destination as an absolute path anchored at their workspace cwd
    (observed: hermes rollouts). Match on whole trailing components so
    `x/notes/a.txt` matches destination `notes/a.txt` but `x/evil-notes/a.txt`
    does not."""
    d = PurePosixPath(destination).parts
    p = PurePosixPath(path).parts
    return bool(d) and len(p) >= len(d) and p[-len(d) :] == d


def _final_write(rec: ConversationRecord, target: str | None) -> tuple[bool, str]:
    """Final state of the requested file: (last write succeeded, content it carried)
    over every write_file call whose path resolves to `target`."""
    ok, content_out = False, ""
    calls: dict[str, tuple[str, str]] = {}  # call id -> (path, content)
    for m in rec.messages:
        if m.role == "assistant":
            for tc in m.tool_calls:
                if tc.name == "write_file" and "path" in tc.arguments:
                    calls[tc.id] = (
                        str(tc.arguments["path"]),
                        str(tc.arguments.get("content", "")),
                    )
        elif m.role == "tool" and m.tool_call_id in calls:
            path, content = calls[m.tool_call_id]
            if not (target and _is_destination(path, target)):
                continue
            # Hermes result convention: tools return JSON; success = a parsed dict
            # WITHOUT an "error" key (write success is {"bytes_written": N, ...}).
            # Unparseable results fail closed.
            try:
                result = json.loads(m.content)
            except json.JSONDecodeError:
                result = None
            ok = isinstance(result, dict) and not result.get("error")
            content_out = content
    return ok, content_out


def verify(rec: ConversationRecord) -> VerifierResult:
    params = rec.provenance.source.detail.get("params", {})
    destination = params.get("destination")
    index = params.get("index")

    # Fail closed if provenance doesn't name the requested destination or index:
    # without them we cannot assert the outcome, and accepting any successful
    # write is exactly the hole this verifier is meant to close.
    dest_ok, dest_content = _final_write(rec, destination)
    index_ok, index_content = _final_write(rec, index)

    wrote_target = bool(destination) and dest_ok
    content_nonempty = bool(destination) and bool(dest_content.strip())
    # The index must not merely exist — it has to actually name the saved file
    # (the coordination the v2 template exists to test).
    index_written = bool(index) and index_ok
    index_names_file = bool(destination) and PurePosixPath(destination).name in index_content

    last = rec.messages[-1]
    confirmed = last.role == "assistant" and not last.tool_calls and bool(last.content.strip())

    checks = {
        "wrote_requested_path": wrote_target,
        "content_nonempty": content_nonempty,
        "index_written": index_written,
        "index_names_file": index_names_file,
        "confirmed_to_user": confirmed,
    }
    return VerifierResult(
        passed=all(checks.values()),
        verifier_id=VERIFIER_ID,
        details={k: v for k, v in checks.items() if not v},
    )
