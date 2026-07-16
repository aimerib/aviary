"""Lane B structural verifier: a scene record is well-formed roleplay data.

Outcome-only: judges the final record shape, never how extraction got there.
"""

from aviary.schema.records import ConversationRecord
from aviary.schema.results import VerifierResult

MIN_TURNS = 6
MIN_THOUGHT_COVERAGE = 0.5

VERIFIER_ID = "laneb/scene_wellformed"


def verify(rec: ConversationRecord) -> VerifierResult:
    turns = [m for m in rec.messages if m.role == "assistant"]
    speakers = {m.speaker for m in turns if m.speaker}
    with_thought = sum(1 for m in turns if m.thought)
    checks = {
        "enough_turns": len(turns) >= MIN_TURNS,
        "multi_speaker": len(speakers) >= 2,
        "speakers_labeled": all(m.speaker for m in turns),
        "thought_coverage": bool(turns) and with_thought / len(turns) >= MIN_THOUGHT_COVERAGE,
        "no_bracket_thoughts": not any("[" in m.content and "]" in m.content for m in turns),
        "system_has_scene": "## Scene" in rec.system,
    }
    return VerifierResult(
        passed=all(checks.values()),
        verifier_id=VERIFIER_ID,
        details={k: v for k, v in checks.items() if not v},
    )
