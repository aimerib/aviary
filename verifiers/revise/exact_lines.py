"""Verifier for revise.exact_lines: outcome gate on the FINAL file shape.

Passes iff the requested destination ends in a successful write whose content is
exactly n_lines non-empty lines with no numbering or bullet markers, and the
final assistant turn confirms. Path is irrelevant (outcome-gate rule): a sloppy
first draft followed by a read-back and a corrective rewrite is prime data —
only the final shape is gated.
"""

import json
import re
from pathlib import PurePosixPath

from aviary.schema.records import ConversationRecord
from aviary.schema.results import VerifierResult

VERIFIER_ID = "revise/exact_lines"

# "no numbering, no bullets": leading 1. / 1) / - / * / • (bare markers).
_MARKER = re.compile(r"^\s*(?:\d+\s*[.)]|[-*•])\s")


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
    destination = params.get("destination")
    try:
        n_lines = int(params["n_lines"])
    except (KeyError, TypeError, ValueError):
        n_lines = None  # fail closed below

    ok, content = _final_write(rec, destination)
    wrote_target = bool(destination) and ok

    # A single trailing newline is a file convention, not a blank line.
    lines = content.rstrip("\n").split("\n") if content else []
    exact_count = n_lines is not None and len(lines) == n_lines
    no_blanks = bool(lines) and all(ln.strip() for ln in lines)
    plain_lines = bool(lines) and not any(_MARKER.match(ln) for ln in lines)

    last = rec.messages[-1]
    confirmed = last.role == "assistant" and not last.tool_calls and bool(last.content.strip())

    checks = {
        "wrote_requested_path": wrote_target,
        "exact_line_count": exact_count,
        "no_blank_lines": no_blanks,
        "no_list_markers": plain_lines,
        "confirmed_to_user": confirmed,
    }
    return VerifierResult(
        passed=all(checks.values()),
        verifier_id=VERIFIER_ID,
        details={k: v for k, v in checks.items() if not v},
    )
