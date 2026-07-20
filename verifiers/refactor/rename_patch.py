"""Verifier for refactor.rename_patch: outcome gate on the FINAL file content.

Passes iff the requested destination's final content contains zero occurrences
of old_term and at least one occurrence of new_term, and the final assistant
turn confirms. Path is irrelevant (outcome-gate rule): a full rewrite that
lands on the correct final state passes exactly like a `patch` call chain.

Final content is reconstructed by replaying, IN ORDER, every write_file call
(full overwrite) and every patch call in mode='replace' (old_string ->
new_string, honoring replace_all) against the tracked path. patch calls in
mode='patch' (V4A multi-file diff) are not parsed and leave tracked content
unchanged — a conservative miss (may fail a record that actually succeeded),
never a false pass.
"""

import json
from pathlib import PurePosixPath

from aviary.schema.records import ConversationRecord
from aviary.schema.results import VerifierResult

VERIFIER_ID = "refactor/rename_patch"


def _is_destination(path: str, destination: str) -> bool:
    """Outcome-equivalent path identity (see files/save_note)."""
    d = PurePosixPath(destination).parts
    p = PurePosixPath(path).parts
    return bool(d) and len(p) >= len(d) and p[-len(d) :] == d


def _ok(tool_content: str) -> bool:
    try:
        result = json.loads(tool_content)
    except json.JSONDecodeError:
        return False
    return isinstance(result, dict) and not result.get("error")


def verify(rec: ConversationRecord) -> VerifierResult:
    params = rec.provenance.source.detail.get("params", {})
    destination = params.get("destination")
    old_term = params.get("old_term")
    new_term = params.get("new_term")

    tracked: str | None = None
    calls: dict[str, dict] = {}  # call id -> {"kind": ..., ...}
    for m in rec.messages:
        if m.role == "assistant":
            for tc in m.tool_calls:
                if not (destination and "path" in tc.arguments):
                    continue
                path = str(tc.arguments["path"])
                if not _is_destination(path, destination):
                    continue
                if tc.name == "write_file":
                    calls[tc.id] = {
                        "kind": "write",
                        "content": str(tc.arguments.get("content", "")),
                    }
                elif tc.name == "patch" and tc.arguments.get("mode", "replace") == "replace":
                    calls[tc.id] = {
                        "kind": "replace",
                        "old": str(tc.arguments.get("old_string", "")),
                        "new": str(tc.arguments.get("new_string", "")),
                        "all": bool(tc.arguments.get("replace_all", False)),
                    }
        elif m.role == "tool" and m.tool_call_id in calls:
            call = calls[m.tool_call_id]
            if not _ok(m.content):
                continue
            if call["kind"] == "write":
                tracked = call["content"]
            elif call["kind"] == "replace" and tracked is not None:
                if call["all"]:
                    tracked = tracked.replace(call["old"], call["new"])
                else:
                    tracked = tracked.replace(call["old"], call["new"], 1)

    wrote_target = tracked is not None
    old_gone = wrote_target and bool(old_term) and old_term not in tracked
    new_present = wrote_target and bool(new_term) and new_term in tracked

    last = rec.messages[-1]
    confirmed = last.role == "assistant" and not last.tool_calls and bool(last.content.strip())

    checks = {
        "wrote_requested_path": wrote_target,
        "old_term_gone": old_gone,
        "new_term_present": new_present,
        "confirmed_to_user": confirmed,
    }
    return VerifierResult(
        passed=all(checks.values()),
        verifier_id=VERIFIER_ID,
        details={k: v for k, v in checks.items() if not v},
    )
