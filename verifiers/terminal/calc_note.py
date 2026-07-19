"""Verifier for terminal.calc_note: outcome gate on the FINAL saved number.

Recomputes the amortized monthly payment from the task params (standard formula,
M = P*r/(1-(1+r)^-n), r = rate_pct/1200, n = years*12) and passes iff the
requested destination ends in a successful write whose content contains that
number to the cent, and the final assistant turn confirms. Path is irrelevant
(outcome-gate rule): whether the teacher used the terminal, python, or divine
inspiration, only the number's correctness is gated — the task's difficulty is
that mental math reliably gets it wrong.
"""

import json
import re
from pathlib import PurePosixPath

from aviary.schema.records import ConversationRecord
from aviary.schema.results import VerifierResult

VERIFIER_ID = "terminal/calc_note"

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _is_destination(path: str, destination: str) -> bool:
    """Outcome-equivalent path identity (see files/save_note): whole trailing
    components, so workspace-absolute writes of the requested path match."""
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


def _expected_payment(params: dict) -> float | None:
    try:
        principal = float(params["principal"])
        rate = float(params["rate_pct"]) / 100.0 / 12.0
        n = int(params["years"]) * 12
    except (KeyError, TypeError, ValueError):
        return None
    if rate <= 0 or n <= 0:
        return None
    return principal * rate / (1.0 - (1.0 + rate) ** -n)


def verify(rec: ConversationRecord) -> VerifierResult:
    params = rec.provenance.source.detail.get("params", {})
    destination = params.get("destination")
    expected = _expected_payment(params)

    ok, content = _final_write(rec, destination)
    wrote_target = bool(destination) and ok

    # Correct to the cent: any number in the file within $0.011 of the recomputed
    # payment. Comma thousands separators are normalized; the loan inputs
    # (principal, rate, years) are orders of magnitude away from a monthly
    # payment, so they can't false-positive.
    correct = False
    if expected is not None:
        for tok in _NUMBER.findall(content):
            try:
                val = float(tok.replace(",", ""))
            except ValueError:
                continue
            if abs(val - round(expected, 2)) <= 0.011:
                correct = True
                break

    last = rec.messages[-1]
    confirmed = last.role == "assistant" and not last.tool_calls and bool(last.content.strip())

    checks = {
        "wrote_requested_path": wrote_target,
        "correct_amount": correct,
        "confirmed_to_user": confirmed,
    }
    return VerifierResult(
        passed=all(checks.values()),
        verifier_id=VERIFIER_ID,
        details={k: v for k, v in checks.items() if not v},
    )
