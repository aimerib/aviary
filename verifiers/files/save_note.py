"""Verifier for files.save_note: outcome gate on the FINAL file state.

Passes iff the last write to the requested path succeeded and the final assistant
turn confirms it. Path taken is irrelevant: a failed write followed by a successful
retry passes (recovered-failure trajectories are prime data).
"""

import json

from aviary.schema.records import ConversationRecord
from aviary.schema.results import VerifierResult

VERIFIER_ID = "files/save_note"


def verify(rec: ConversationRecord) -> VerifierResult:
    # final state per path = last tool result for the last write_file call to it
    final_write_ok: dict[str, bool] = {}
    calls: dict[str, str] = {}  # call id -> path
    for m in rec.messages:
        if m.role == "assistant":
            for tc in m.tool_calls:
                if tc.name == "write_file" and "path" in tc.arguments:
                    calls[tc.id] = str(tc.arguments["path"])
        elif m.role == "tool" and m.tool_call_id in calls:
            try:
                result = json.loads(m.content)
            except json.JSONDecodeError:
                result = {}
            final_write_ok[calls[m.tool_call_id]] = bool(result.get("ok")) and not result.get(
                "error"
            )

    target_paths = [p for p in final_write_ok if not p.endswith("/.keep")]
    wrote_target = bool(target_paths) and all(final_write_ok[p] for p in target_paths)
    last = rec.messages[-1]
    confirmed = last.role == "assistant" and not last.tool_calls and bool(last.content.strip())

    checks = {"final_write_succeeded": wrote_target, "confirmed_to_user": confirmed}
    return VerifierResult(
        passed=all(checks.values()),
        verifier_id=VERIFIER_ID,
        details={k: v for k, v in checks.items() if not v},
    )
