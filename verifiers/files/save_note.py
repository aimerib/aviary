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

from aviary.schema.records import ConversationRecord
from aviary.schema.results import VerifierResult

VERIFIER_ID = "files/save_note"


def verify(rec: ConversationRecord) -> VerifierResult:
    params = rec.provenance.source.detail.get("params", {})
    destination = params.get("destination")

    # Final state per path = last write_file result + the content it carried.
    final_write_ok: dict[str, bool] = {}
    final_write_content: dict[str, str] = {}
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
            # Hermes result convention: tools return JSON; success = a parsed dict
            # WITHOUT an "error" key (write success is {"bytes_written": N, ...}).
            # Unparseable results fail closed.
            try:
                result = json.loads(m.content)
            except json.JSONDecodeError:
                result = None
            final_write_ok[path] = isinstance(result, dict) and not result.get("error")
            final_write_content[path] = content

    # Fail closed if provenance doesn't name the requested destination: without it
    # we cannot assert the outcome, and accepting any successful write is exactly
    # the hole this verifier is meant to close.
    wrote_target = bool(destination) and final_write_ok.get(destination, False)
    content_nonempty = bool(destination) and bool(final_write_content.get(destination, "").strip())

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
