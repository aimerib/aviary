"""Lane C structural verifier: the character never writes the user's turns.

Impersonation = an assistant turn that puts words in the user's mouth: lines
opening with a user-style speaker tag, or second-person dialogue attribution
('you say', 'you reply'). Outcome-only over the final transcript.
"""

import re

from aviary.schema.records import ConversationRecord
from aviary.schema.results import VerifierResult

VERIFIER_ID = "lanec/no_impersonation"

USER_TAG = re.compile(r"^\s*(?:\{\{user\}\}|user)\s*:", re.I | re.M)
YOU_SAY = re.compile(r'\byou\s+(?:say|reply|respond|answer)\b[,:]?\s*["“]', re.I)


def verify(rec: ConversationRecord) -> VerifierResult:
    offenders = []
    for i, m in enumerate(rec.messages):
        if m.role != "assistant":
            continue
        if USER_TAG.search(m.content) or YOU_SAY.search(m.content):
            offenders.append(i)
    return VerifierResult(
        passed=not offenders,
        verifier_id=VERIFIER_ID,
        details={"offending_turns": offenders} if offenders else {},
    )
