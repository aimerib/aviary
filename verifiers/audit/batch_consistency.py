"""Verifier for audit.batch_consistency: outcome gate over a FULL batch of N
deterministically-named files, not just one or two named destinations.

Expected paths are {prefix_dir}/note-{i}.txt for i in 1..count (derived from
params, never re-guessed). Passes iff EVERY one ends in a successful final
write whose first line equals required_line exactly, and the final assistant
turn confirms. One missed or malformed file among N fails the whole record —
path irrelevant, a first-draft miss caught and fixed on a later write passes.
"""

import json
from pathlib import PurePosixPath

from aviary.schema.records import ConversationRecord
from aviary.schema.results import VerifierResult

VERIFIER_ID = "audit/batch_consistency"


def _is_destination(path: str, destination: str) -> bool:
    """Outcome-equivalent path identity (see files/save_note)."""
    d = PurePosixPath(destination).parts
    p = PurePosixPath(path).parts
    return bool(d) and len(p) >= len(d) and p[-len(d) :] == d


def _final_write(rec: ConversationRecord, target: str | None) -> tuple[bool, str]:
    ok, content_out = False, ""
    calls: dict[str, tuple[str, str]] = {}
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
            try:
                result = json.loads(m.content)
            except json.JSONDecodeError:
                result = None
            ok = isinstance(result, dict) and not result.get("error")
            content_out = content
    return ok, content_out


def verify(rec: ConversationRecord) -> VerifierResult:
    params = rec.provenance.source.detail.get("params", {})
    prefix_dir = params.get("prefix_dir")
    required_line = params.get("required_line")
    try:
        count = int(params["count"])
    except (KeyError, TypeError, ValueError):
        count = None

    checks: dict[str, bool] = {}
    if count is None or not prefix_dir or required_line is None:
        checks["params_present"] = False
    else:
        for i in range(1, count + 1):
            target = f"{prefix_dir}/note-{i}.txt"
            ok, content = _final_write(rec, target)
            first_line = content.split("\n", 1)[0] if content else ""
            checks[f"note_{i}_written"] = ok
            checks[f"note_{i}_correct_first_line"] = ok and first_line == required_line

    last = rec.messages[-1]
    checks["confirmed_to_user"] = (
        last.role == "assistant" and not last.tool_calls and bool(last.content.strip())
    )

    return VerifierResult(
        passed=all(checks.values()),
        verifier_id=VERIFIER_ID,
        details={k: v for k, v in checks.items() if not v},
    )
