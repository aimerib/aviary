"""Verifier for files.save_note: outcome gate on the FINAL file state.

Passes iff the file the user asked for — the task's `destination` param — ends in
a successful write with non-empty content, and the final assistant turn confirms
it. Path taken is irrelevant: a failed write followed by a successful retry passes
(recovered-failure trajectories are prime data).

Scope: this gates that SOMETHING substantive reached the RIGHT path. Whether the
content is genuinely a haiku / a five-item list / etc. is a semantic judgement and
belongs to the judge stage, not a pure verifier — so we check the destination and
non-emptiness here, not correctness.
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


def verify(rec: ConversationRecord) -> VerifierResult:
    params = rec.provenance.source.detail.get("params", {})
    destination = params.get("destination")

    # Final state of the requested file = the LAST write to any path that resolves
    # to the destination (an earlier failed write followed by a successful retry
    # passes — recovered-failure trajectories are prime data).
    target_ok = False
    target_content = ""
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
            if not (destination and _is_destination(path, destination)):
                continue
            # Hermes result convention: tools return JSON; success = a parsed dict
            # WITHOUT an "error" key (write success is {"bytes_written": N, ...}).
            # Unparseable results fail closed.
            try:
                result = json.loads(m.content)
            except json.JSONDecodeError:
                result = None
            target_ok = isinstance(result, dict) and not result.get("error")
            target_content = content

    # Fail closed if provenance doesn't name the requested destination: without it
    # we cannot assert the outcome, and accepting any successful write is exactly
    # the hole this verifier is meant to close.
    wrote_target = bool(destination) and target_ok
    content_nonempty = bool(destination) and bool(target_content.strip())

    last = rec.messages[-1]
    confirmed = last.role == "assistant" and not last.tool_calls and bool(last.content.strip())

    checks = {
        "wrote_requested_path": wrote_target,
        "content_nonempty": content_nonempty,
        "confirmed_to_user": confirmed,
    }
    return VerifierResult(
        passed=all(checks.values()),
        verifier_id=VERIFIER_ID,
        details={k: v for k, v in checks.items() if not v},
    )
