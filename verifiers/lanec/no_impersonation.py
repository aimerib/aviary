"""Lane C structural verifier: the character never writes the user's turns.

Impersonation = an assistant turn that puts words in the user's mouth: a line
opening with a user-style speaker tag (`{{user}}:`, `User:`, `You:`), or PRESENT-
TENSE second-person dialogue attribution that scripts a user line ('you say,
"..."'). Outcome-only over the final transcript.

Note the deliberate narrowness of YOU_SAY: it requires present tense AND a trailing
quote, so Olivia recalling something the user really said ('You said "the deadline
was fake"') is NOT impersonation and passes. Broadening to past tense or dropping
the quote would reject those legitimate keepers (see recovered_pass fixture).
"""

import re

from aviary.schema.records import ConversationRecord
from aviary.schema.results import VerifierResult

VERIFIER_ID = "lanec/no_impersonation"

USER_TAG = re.compile(r"^\s*(?:\{\{user\}\}|user|you)\s*:", re.I | re.M)
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
