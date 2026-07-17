"""Regression tests for verifiers/lanec/no_impersonation.py. Locks the `You:`
speaker-tag gap and the deliberate boundary: past-tense recollection of the user's
real speech is NOT impersonation."""

from __future__ import annotations

from pathlib import Path

from aviary.gates.verify import run_verifier
from aviary.schema.records import ConversationRecord, Message, Provenance, SourceRef

REPO = Path(__file__).resolve().parents[1]
VERIFIER = REPO / "verifiers" / "lanec" / "no_impersonation.py"


def _rec(reply: str) -> ConversationRecord:
    return ConversationRecord(
        system="s",
        messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", speaker="Olivia", content=reply),
        ],
        provenance=Provenance(
            record_id="t",
            lane="c",
            run_id="t",
            family="banter",
            source=SourceRef(kind="selfplay_seed", detail={}),
        ),
    )


def test_you_colon_tag_is_impersonation():
    rec = _rec("Sure.\nYou: thanks olivia you're the best\nSee, easy.")
    assert not run_verifier(VERIFIER, rec).passed


def test_present_tense_scripted_attribution_is_impersonation():
    rec = _rec('And then you say, "fine, you win" — predictable.')
    assert not run_verifier(VERIFIER, rec).passed


def test_past_tense_recollection_is_allowed():
    # Olivia quoting the user's real prior speech is not impersonation.
    rec = _rec('Bold. You said "the deadline was fake" to its inventor. Respect.')
    assert run_verifier(VERIFIER, rec).passed


def test_plain_second_person_is_allowed():
    rec = _rec("You always do this and you know it. Go get coffee.")
    assert run_verifier(VERIFIER, rec).passed
