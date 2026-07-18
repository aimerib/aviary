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

from aviary.schema.records import ConversationRecord
from aviary.schema.results import VerifierResult

VERIFIER_ID = "web/compare_sources"

# Deliberately shape-agnostic: tool results carry URLs verbatim (toolset contract),
# so we regex them out rather than couple to any one result-JSON schema.
_URL = re.compile(r"https?://[^\s\"'<>\)\]\}]+")


def _urls(text: str) -> set[str]:
    return {u.rstrip(".,;:!?").rstrip("/") for u in _URL.findall(text)}


def verify(rec: ConversationRecord) -> VerifierResult:
    params = rec.provenance.source.detail.get("params", {})
    destination = params.get("destination")

    # Final state per path = last write_file result + the content it carried.
    final_write_ok: dict[str, bool] = {}
    final_write_content: dict[str, str] = {}
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
                try:
                    result = json.loads(m.content)
                except json.JSONDecodeError:
                    result = {}
                final_write_ok[path] = bool(result.get("ok")) and not result.get("error")
                final_write_content[path] = content

    # Fail closed without a destination param (same hole save_note closes).
    wrote_target = bool(destination) and final_write_ok.get(destination, False)
    cited = _urls(final_write_content.get(destination, "")) if destination else set()
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
