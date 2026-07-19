"""Per-lane scrub policy + pseudonymizer (Sorcha prep T4). All names synthetic."""

from __future__ import annotations

from aviary.gates.pseudonym import Pseudonymizer, PseudonymRules
from aviary.gates.scrub import LaneScrubPolicy, load_patterns, scan_record
from aviary.schema.records import ConversationRecord, Message, Provenance, SourceRef

REPO_PATTERNS = None  # loaded per-test from the tracked scrub configs


def _rec(lane: str, text: str, speaker: str | None = None) -> ConversationRecord:
    return ConversationRecord(
        system="",
        messages=[
            Message(role="user", content=text, speaker=speaker),
            Message(role="assistant", content="mm."),
        ],
        provenance=Provenance(
            record_id="r1",
            lane=lane,  # type: ignore[arg-type]
            run_id="t",
            family="chat/2026-03",
            source=SourceRef(kind="personal_stream", detail={}),
        ),
    )


def _rules() -> PseudonymRules:
    return PseudonymRules(
        preserve=["Sam"],
        name_patterns=["Rio", "Marisol"],
    )


def test_pseudonymizer_stable_and_preserving(tmp_path):
    p = Pseudonymizer(_rules(), tmp_path / "pseudonyms.json")
    a = p.apply(_rec("d", "Sam told Rio about the trip. Rio laughed."))
    b = p.apply(_rec("d", "Then Marisol called Rio."))
    assert a.messages[0].content == "Sam told Person-1 about the trip. Person-1 laughed."
    assert b.messages[0].content == "Then Person-2 called Person-1."  # stable across records

    # Stability survives process restart via the persisted mapping.
    p2 = Pseudonymizer(_rules(), tmp_path / "pseudonyms.json")
    c = p2.apply(_rec("d", "Rio again"))
    assert c.messages[0].content == "Person-1 again"


def test_pseudonymizer_covers_speaker_and_thought(tmp_path):
    p = Pseudonymizer(_rules(), tmp_path / "m.json")
    rec = _rec("d", "hey", speaker="Rio")
    out = p.apply(rec)
    assert out.messages[0].speaker == "Person-1"
    preserved = p.apply(_rec("d", "hi", speaker="Sam"))
    assert preserved.messages[0].speaker == "Sam"  # owner identifiers survive


def test_pseudonymizer_untouched_record_is_same_object(tmp_path):
    p = Pseudonymizer(_rules(), tmp_path / "m.json")
    rec = _rec("d", "nothing to change here")
    assert p.apply(rec) is rec  # no silent copies: transform flag keys off identity


def test_lane_policy_exempts_labels_per_lane(tmp_path):
    from pathlib import Path

    REPO = Path(__file__).resolve().parents[1]
    patterns = load_patterns(REPO / "gates" / "scrub" / "pii_patterns.yaml")
    email_labels = {p.label for p in patterns if p.pattern.search("mail me at sam@example.com")}
    assert email_labels, "expected an email-shaped PII pattern in the tracked config"

    rec = _rec("d", "reach me at sam@example.com")
    hits_default = scan_record(rec, patterns)
    assert any(h.startswith("drop:") for h in hits_default)  # default posture drops

    policy = LaneScrubPolicy(exempt_labels=frozenset(email_labels))
    exempt = [p for p in patterns if p.label not in policy.exempt_labels]
    assert not any(h.startswith("drop:") for h in scan_record(rec, exempt))


def test_run_gates_applies_lane_policy_and_transform(tmp_path):
    from aviary.gates.judge import Rubric
    from aviary.gates.pipeline import run_gates
    from aviary.io.jsonl import read_jsonl, write_jsonl
    from aviary.io.store import RunStore
    from aviary.teacher.fake import FakeTeacherClient
    from aviary.teacher.prompts import PromptSet
    from aviary.teacher.roster import Roster

    roster = Roster(
        teachers=[
            {
                "id": "glm-5-20260430",
                "provider": "zhipu",
                "route": "direct",
                "base_url": "x",
                "wire_model": "glm",
                "api_key_env": "K",
            },
            {
                "id": "deepseek-v4-flash-20260610",
                "provider": "deepseek",
                "route": "direct",
                "base_url": "x",
                "wire_model": "ds",
                "api_key_env": "K",
            },
        ],
        assignments={
            "judge": {"primary": "glm-5-20260430", "secondary": "deepseek-v4-flash-20260610"},
            "harmonizer": {"primary": "deepseek-v4-flash-20260610"},
        },
    )
    prompts = PromptSet({"judge_prompt": "judge it", "harmonize_prompt": "harmonize it"})

    rec = _rec("d", "Rio texted sam@example.com at midnight")
    rec = rec.model_copy(
        update={
            "provenance": rec.provenance.model_copy(
                update={"teachers": {"assistant": "deepseek-v4-flash-20260610"}}
            )
        }
    )
    store = RunStore("t", root=tmp_path)
    write_jsonl(store.raw("d"), [rec])

    # A minimal always-pass verifier stands in for lane D's structural verifiers
    # (they land with T6); this test pins the SCRUB policy wiring.
    ok_verifier = tmp_path / "ok.py"
    ok_verifier.write_text(
        "from aviary.schema.results import VerifierResult\n"
        "def verify(rec):\n"
        "    return VerifierResult(passed=True, verifier_id='t/ok', details={})\n"
    )

    from pathlib import Path

    REPO = Path(__file__).resolve().parents[1]
    patterns = load_patterns(REPO / "gates" / "scrub" / "pii_patterns.yaml")
    email_labels = {p.label for p in patterns if p.pattern.search("x sam@example.com y")}
    pseudo = Pseudonymizer(_rules(), store.root / "scrub" / "pseudonyms.json")
    laned_rubric = Rubric.load(REPO / "datagen" / "persona" / "sorcha" / "laned.rubric.yaml")
    stats = run_gates(
        store,
        lambda r: [ok_verifier],
        {"d": laned_rubric},
        patterns,
        roster,
        FakeTeacherClient(script=lambda r: '{"scores": {"coherence": 5, "naturalness": 4}}'),
        prompts,
        persona_speaker="Sorcha",
        scrub_policies={
            "d": LaneScrubPolicy(exempt_labels=frozenset(email_labels), transform=pseudo.apply)
        },
    )
    assert stats.kept == 1  # email exempted for lane d -> no scrub drop
    kept = list(read_jsonl(store.gated_kept(), ConversationRecord))
    assert "Person-1 texted sam@example.com" in kept[0].messages[0].content  # pseudonymized
    assert "flag:transformed" in kept[0].gate_state.scrub_flags
    assert (store.root / "scrub" / "pseudonyms.json").exists()  # mapping under the run dir
