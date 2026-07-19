"""Verifier for webmath.fact_ratio: outcome gate on the FINAL grounded artifact.

Live-web quantities can't be recomputed by a pure verifier, so the gate is
structural: the requested destination ends in a successful write whose content
cites >=2 distinct grounded sources (URLs that actually appeared in a tool
result this session), is INTERNALLY CONSISTENT (some stated number equals the
division of two other stated numbers within 2% — raw quantities and ratio must
actually agree), and stays inside the word budget (<=200 non-URL words).
Whether the numbers are faithful to the cited sources is semantic judgement and
belongs to the judge stage. Path is irrelevant: dead links and re-searches
along the way are fine.
"""

import json
import re
from pathlib import PurePosixPath

from aviary.schema.records import ConversationRecord
from aviary.schema.results import VerifierResult

VERIFIER_ID = "webmath/fact_ratio"

_URL = re.compile(r"https?://[^\s\"'<>\)\]\}]+")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _urls(text: str) -> set[str]:
    return {u.rstrip(".,;:!?").rstrip("/") for u in _URL.findall(text)}


def _is_destination(path: str, destination: str) -> bool:
    """Outcome-equivalent path identity (see files/save_note)."""
    d = PurePosixPath(destination).parts
    p = PurePosixPath(path).parts
    return bool(d) and len(p) >= len(d) and p[-len(d) :] == d


def _ratio_consistent(text: str) -> bool:
    """Some stated number equals the division of two other stated numbers within
    2% — i.e. the file carries the raw quantities AND a ratio that actually
    follows from them. Napkin math and mismatched rounding fail; honest terminal
    division passes. Units cancel, so '123.1 million' vs '123,100,000' both work
    as long as numerator and denominator use the same convention."""
    vals: list[float] = []
    for tok in _NUMBER.findall(text):
        try:
            v = float(tok.replace(",", ""))
        except ValueError:
            continue
        if v > 0:
            vals.append(v)
    for i, x in enumerate(vals):
        for j, a in enumerate(vals):
            if j == i:
                continue
            for k, b in enumerate(vals):
                if k in (i, j) or a <= b:
                    continue
                r = a / b
                if r > 1.0 and abs(x - r) <= 0.02 * r:
                    return True
    return False


def verify(rec: ConversationRecord) -> VerifierResult:
    params = rec.provenance.source.detail.get("params", {})
    destination = params.get("destination")

    target_ok = False
    target_content = ""
    calls: dict[str, tuple[str, str]] = {}
    grounded: set[str] = set()  # every URL any tool result surfaced this session
    for m in rec.messages:
        if m.role == "assistant":
            for tc in m.tool_calls:
                if tc.name == "write_file" and "path" in tc.arguments:
                    calls[tc.id] = (
                        str(tc.arguments["path"]),
                        str(tc.arguments.get("content", "")),
                    )
        elif m.role == "tool":
            grounded |= _urls(m.content)
            if m.tool_call_id in calls:
                path, content = calls[m.tool_call_id]
                if not (destination and _is_destination(path, destination)):
                    continue
                try:
                    result = json.loads(m.content)
                except json.JSONDecodeError:
                    result = None
                target_ok = isinstance(result, dict) and not result.get("error")
                target_content = content

    wrote_target = bool(destination) and target_ok
    cited = _urls(target_content) if destination else set()
    grounded_citations = cited & grounded
    words = [w for w in target_content.split() if not _URL.match(w)]

    last = rec.messages[-1]
    confirmed = last.role == "assistant" and not last.tool_calls and bool(last.content.strip())

    checks = {
        "wrote_requested_path": wrote_target,
        "cited_two_grounded_sources": len(grounded_citations) >= 2,
        "ratio_internally_consistent": _ratio_consistent(target_content),
        "within_word_budget": bool(destination) and 0 < len(words) <= 200,
        "confirmed_to_user": confirmed,
    }
    return VerifierResult(
        passed=all(checks.values()),
        verifier_id=VERIFIER_ID,
        details={k: v for k, v in checks.items() if not v},
    )
