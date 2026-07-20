"""Verifier for count.tagged_count: two-part outcome gate against a
PRECOMPUTED ground truth (true_count fixed at task-authoring time from
entries_block, never derived from the teacher's own output).

Passes iff destination's final content reproduces entries_block verbatim
(line-normalized: trailing whitespace and a trailing blank line are ignored,
everything else must match), count_destination's final content states
true_count as a number, and the final assistant turn confirms. Path
irrelevant — miscounting then catching it on a re-read still passes.
"""

import json
import re
from pathlib import PurePosixPath

from aviary.schema.records import ConversationRecord
from aviary.schema.results import VerifierResult

VERIFIER_ID = "count/tagged_count"

_NUMBER = re.compile(r"\d+")


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


def _norm_lines(text: str) -> list[str]:
    lines = [ln.rstrip() for ln in text.split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def verify(rec: ConversationRecord) -> VerifierResult:
    params = rec.provenance.source.detail.get("params", {})
    destination = params.get("destination")
    count_destination = params.get("count_destination")
    entries_block = params.get("entries_block", "")
    true_count = params.get("true_count")

    log_ok, log_content = _final_write(rec, destination)
    count_ok, count_content = _final_write(rec, count_destination)

    wrote_log = bool(destination) and log_ok
    log_verbatim = wrote_log and _norm_lines(log_content) == _norm_lines(entries_block)

    wrote_count = bool(count_destination) and count_ok
    correct_count = False
    if wrote_count and true_count is not None:
        correct_count = any(int(tok) == int(true_count) for tok in _NUMBER.findall(count_content))

    last = rec.messages[-1]
    confirmed = last.role == "assistant" and not last.tool_calls and bool(last.content.strip())

    checks = {
        "wrote_log_verbatim": log_verbatim,
        "wrote_correct_count": correct_count,
        "confirmed_to_user": confirmed,
    }
    return VerifierResult(
        passed=all(checks.values()),
        verifier_id=VERIFIER_ID,
        details={k: v for k, v in checks.items() if not v},
    )
