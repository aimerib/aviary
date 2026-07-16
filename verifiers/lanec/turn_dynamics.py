"""Lane C structural verifier: the conversation has real turn-taking dynamics.

Checks the final transcript alternates user/assistant, is long enough to teach
multi-turn behavior, and is not a degenerate loop (repeated near-identical turns).
"""

from aviary.schema.records import ConversationRecord
from aviary.schema.results import VerifierResult

VERIFIER_ID = "lanec/turn_dynamics"

MIN_EXCHANGES = 3
MAX_REPEAT_RATIO = 0.5


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def verify(rec: ConversationRecord) -> VerifierResult:
    convo = [m for m in rec.messages if m.role in ("user", "assistant")]
    exchanges = sum(1 for m in convo if m.role == "user")
    alternates = all(a.role != b.role for a, b in zip(convo, convo[1:], strict=False))
    assistant_turns = [_normalized(m.content) for m in convo if m.role == "assistant"]
    unique_ratio = len(set(assistant_turns)) / len(assistant_turns) if assistant_turns else 0.0

    checks = {
        "enough_exchanges": exchanges >= MIN_EXCHANGES,
        "alternating": alternates,
        "not_degenerate": unique_ratio > MAX_REPEAT_RATIO,
    }
    return VerifierResult(
        passed=all(checks.values()),
        verifier_id=VERIFIER_ID,
        details={k: v for k, v in checks.items() if not v},
    )
