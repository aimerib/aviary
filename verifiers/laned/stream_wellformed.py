"""Structural verifier for lane D personal-stream records: outcome gate on the
normalized record's FINAL shape, not on how the adapter got there.

Checks are structural only (this is a pure function; privacy/pseudonymization is
the scrub stage's job): a real two-way conversation, every message timestamped in
order, family consistent with the record's first timestamp and source, and no
tool machinery (personal streams carry none).
"""

from datetime import datetime

from aviary.schema.records import ConversationRecord
from aviary.schema.results import VerifierResult

VERIFIER_ID = "laned/stream_wellformed"


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def verify(rec: ConversationRecord) -> VerifierResult:
    msgs = rec.messages
    ts = [_parse_ts(m.ts) for m in msgs]

    two_sided = len({m.role for m in msgs}) >= 2 and len(msgs) >= 2
    non_empty = all(m.content.strip() for m in msgs)
    speakers_present = all(m.speaker for m in msgs)
    all_timestamped = all(t is not None for t in ts)
    ordered = all_timestamped and all(a <= b for a, b in zip(ts, ts[1:], strict=False))
    no_tools = all(m.role in ("user", "assistant") and not m.tool_calls for m in msgs)

    family_consistent = False
    if all_timestamped and msgs:
        source = str(rec.provenance.source.detail.get("source", ""))
        family_consistent = rec.provenance.family == f"{source}/{ts[0]:%Y-%m}"

    checks = {
        "two_sided_conversation": two_sided,
        "content_non_empty": non_empty,
        "speakers_present": speakers_present,
        "all_messages_timestamped": all_timestamped,
        "timestamps_ordered": ordered,
        "no_tool_machinery": no_tools,
        "family_matches_first_month": family_consistent,
    }
    return VerifierResult(
        passed=all(checks.values()),
        verifier_id=VERIFIER_ID,
        details={k: v for k, v in checks.items() if not v},
    )
