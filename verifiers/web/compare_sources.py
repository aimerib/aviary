"""Verifier for web.compare_sources: outcome gate on the FINAL saved artifact.

Passes iff the requested destination ends in a successful write whose content
cites >=2 distinct grounded sources — URLs that actually appeared in a tool
result this session — and the final assistant turn confirms without pending
tool calls. Path taken is irrelevant: dead links, failed fetches, and re-searches
along the way are fine (recovered-failure trajectories are prime data).

Scope: this gates that the artifact reached the RIGHT path and is grounded in
sources the session really retrieved. Whether the comparison is faithful to
those sources is semantic judgement and belongs to the judge stage.
"""

import json
import re
from pathlib import PurePosixPath

from aviary.schema.records import ConversationRecord
from aviary.schema.results import VerifierResult

VERIFIER_ID = "web/compare_sources"

# Deliberately shape-agnostic: tool results carry URLs verbatim (toolset contract),
# so we regex them out rather than couple to any one result-JSON schema.
_URL = re.compile(r"https?://[^\s\"'<>\)\]\}]+")


def _urls(text: str) -> set[str]:
    return {u.rstrip(".,;:!?").rstrip("/") for u in _URL.findall(text)}


def _is_destination(path: str, destination: str) -> bool:
    """Outcome-equivalent path identity: teachers legitimately write the requested
    relative destination as an absolute path anchored at their workspace cwd
    (observed: hermes rollouts). Match on whole trailing components so
    `x/research/a.md` matches destination `research/a.md` but `x/no-research/a.md`
    does not."""
    d = PurePosixPath(destination).parts
    p = PurePosixPath(path).parts
    return bool(d) and len(p) >= len(d) and p[-len(d) :] == d


def verify(rec: ConversationRecord) -> VerifierResult:
    params = rec.provenance.source.detail.get("params", {})
    destination = params.get("destination")

    # Final state of the requested file = the LAST write to any path that resolves
    # to the destination (a failed write followed by a successful retry passes —
    # recovered-failure trajectories are prime data).
    target_ok = False
    target_content = ""
    calls: dict[str, tuple[str, str]] = {}  # call id -> (path, content)
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
                # Hermes result convention: tools return JSON; success = a parsed
                # dict WITHOUT an "error" key (write success is {"bytes_written": N}).
                # Unparseable results fail closed.
                try:
                    result = json.loads(m.content)
                except json.JSONDecodeError:
                    result = None
                target_ok = isinstance(result, dict) and not result.get("error")
                target_content = content

    # Fail closed without a destination param (same hole save_note closes).
    wrote_target = bool(destination) and target_ok
    cited = _urls(target_content) if destination else set()
    # >=2 distinct cited URLs that the session actually retrieved. Extra ungrounded
    # links don't fail here (judge territory); too few grounded ones do.
    grounded_citations = cited & grounded

    last = rec.messages[-1]
    confirmed = last.role == "assistant" and not last.tool_calls and bool(last.content.strip())

    checks = {
        "wrote_requested_path": wrote_target,
        "cited_two_grounded_sources": len(grounded_citations) >= 2,
        "confirmed_to_user": confirmed,
    }
    return VerifierResult(
        passed=all(checks.values()),
        verifier_id=VERIFIER_ID,
        details={k: v for k, v in checks.items() if not v},
    )
