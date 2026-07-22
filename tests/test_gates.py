from __future__ import annotations

import json
from pathlib import Path

from aviary.gates.dedupe import find_duplicates
from aviary.gates.harmonize import harmonize_record
from aviary.gates.judge import Rubric, judge_record
from aviary.gates.pipeline import default_resolver, run_gates
from aviary.gates.scrub import load_patterns, scan_record
from aviary.io.jsonl import read_jsonl
from aviary.io.store import RunStore
from aviary.schema.records import ConversationRecord, Message, Provenance, SourceRef
from aviary.teacher.client import ChatRequest
from aviary.teacher.fake import FakeTeacherClient
from aviary.teacher.prompts import PromptSet
from aviary.teacher.roster import Roster

REPO = Path(__file__).resolve().parents[1]

ROSTER = Roster(
    teachers=[
        {
            "id": "deepseek-v4-flash-20260610",
            "provider": "deepseek",
            "route": "direct",
            "base_url": "https://api.deepseek.com/v1",
            "wire_model": "deepseek-chat-20260610",
            "api_key_env": "DEEPSEEK_API_KEY",
        },
        {
            "id": "glm-5-20260430",
            "provider": "zhipu",
            "route": "direct",
            "base_url": "https://api.z.ai/api/paas/v4",
            "wire_model": "glm-5-20260430",
            "api_key_env": "GLM_API_KEY",
        },
    ],
    assignments={
        "judge": {"primary": "glm-5-20260430", "secondary": "deepseek-v4-flash-20260610"},
        "harmonizer": {"primary": "deepseek-v4-flash-20260610"},
    },
)


def make_prompts() -> PromptSet:
    return PromptSet.load(
        {
            "judge_prompt": REPO / "gates" / "judge" / "judge_prompt.md",
            "harmonize_prompt": REPO / "datagen" / "persona" / "olivia" / "paraphrase_prompt.md",
        }
    )


def banter(rid: str, reply: str, teachers=None) -> ConversationRecord:
    return ConversationRecord(
        system="You are Olivia.",
        messages=[
            Message(role="user", content="rough day, distract me"),
            Message(role="assistant", speaker="Olivia", content=reply),
            Message(role="user", content="ha, fair. more"),
            Message(role="assistant", speaker="Olivia", content=f"{reply} But consider: snacks."),
            Message(role="user", content="ok that helps"),
            Message(role="assistant", speaker="Olivia", content="Snacks always do. Go get one."),
        ],
        provenance=Provenance(
            record_id=rid,
            lane="c",
            run_id="testrun",
            family="banter",
            teachers=teachers or {"character": "deepseek-v4-flash-20260610"},
            source=SourceRef(kind="selfplay_seed", detail={"seed_id": rid}),
        ),
    )


def test_scrub_patterns():
    patterns = load_patterns(
        REPO / "gates" / "scrub" / "denylist.yaml", REPO / "gates" / "scrub" / "pii_patterns.yaml"
    )
    bad = banter("r1", "As an AI, I cannot have opinions about your day.")
    assert any(h.startswith("drop:ai_selfid") for h in scan_record(bad, patterns))
    leaky = banter("r2", "Well, DeepSeek would say otherwise.")
    assert any("teacher_deepseek" in h for h in scan_record(leaky, patterns))
    for glm in ("glm-4", "glm-4.6", "glm45", "GLM-5.2"):
        rec = banter(f"r2-{glm}", f"As {glm}, I would say otherwise.")
        assert any("teacher_glm" in h for h in scan_record(rec, patterns)), glm
    email = banter("r3", "Just email bob@example.com about it.")
    assert any("pii_email" in h for h in scan_record(email, patterns))
    clean = banter("r4", "Your day sounds like a spreadsheet crime scene.")
    assert scan_record(clean, patterns) == []


def test_dedupe_exact_and_near():
    # near-dup detection targets long records (rollout siblings that barely diverge);
    # short records are covered by the exact hash.
    base = (
        "Here is the plan for your week, which you will ignore, and that is fine. "
        "Monday you answer the emails you have been performing elaborate grief rituals over. "
        "Tuesday you block two hours and touch the project you keep calling almost done. "
        "Wednesday is for the meeting where you say the thing instead of nodding. "
        "Thursday you take the walk, yes the long one, no headphones. "
        "Friday you write down what actually happened versus this plan and we laugh about it. "
    ) * 2
    a = banter("a", base + "Deal?")
    b = banter("b", base + "Deal?")
    c = banter("c", base + "Bargain?")
    d = banter("d", "Entirely different advice: hydrate, gremlin.")
    dupes = find_duplicates([a, b, c, d], threshold=0.8)
    assert dupes.get("b") == "a"
    assert dupes.get("c") == "a"
    assert "d" not in dupes


def test_judge_scoring_and_threshold():
    rubric = Rubric.load(REPO / "datagen" / "persona" / "olivia" / "quality.rubric.yaml")

    def script(req: ChatRequest) -> str:
        return json.dumps(
            {
                "scores": {
                    "olivia_voice": 5,
                    "coherence": 4,
                    "helpfulness": 4,
                    "naturalness": 4,
                },
                "rationale": "distinct voice",
            }
        )

    scores = judge_record(
        banter("j1", "x"),
        rubric,
        FakeTeacherClient(script=script),
        "glm-5-20260430",
        make_prompts(),
    )
    assert scores.passed and scores.overall == 4.4

    def low_voice(req: ChatRequest) -> str:
        return json.dumps(
            {"scores": {"olivia_voice": 2, "coherence": 5, "helpfulness": 5, "naturalness": 5}}
        )

    scores2 = judge_record(
        banter("j2", "x"),
        rubric,
        FakeTeacherClient(script=low_voice),
        "glm-5-20260430",
        make_prompts(),
    )
    assert not scores2.passed  # axis min gate despite decent weighted mean


def test_harmonize_preserves_protected_and_drops_on_violation():
    rec = banter(
        "h1",
        "Saved 42 items to /tmp/list.txt for you, which took longer than it had any right to. Go check it before you blame me.",
    )

    def echo(req: ChatRequest) -> str:
        return req.messages[0]["content"].replace("Saved", "Stashed")

    outcome = harmonize_record(
        rec, FakeTeacherClient(script=echo), make_prompts(), "deepseek-v4-flash-20260610", "Olivia"
    )
    assert not outcome.dropped
    texts = [m.content for m in outcome.record.messages if m.role == "assistant"]
    assert any("/tmp/list.txt" in t and "42" in t for t in texts)
    assert any("Stashed" in t for t in texts)

    def eats_placeholders(req: ChatRequest) -> str:
        return "I rewrote everything and lost your precious placeholders."

    outcome2 = harmonize_record(
        rec,
        FakeTeacherClient(script=eats_placeholders),
        make_prompts(),
        "deepseek-v4-flash-20260610",
        "Olivia",
    )
    assert outcome2.dropped
    assert outcome2.record.gate_state.drop_reason == "span_violation"


def test_harmonize_retries_restore_failure_then_recovers():
    rec = banter(
        "h1r",
        "Saved 42 items to /tmp/list.txt for you, which took longer than it had any right to. Go check it before you blame me.",
    )

    def flaky(req: ChatRequest) -> str:
        # First sample (temp 0.4) mangles the placeholders; the bumped-temperature
        # retry echoes them back intact. Restore failure must re-sample, not drop.
        if req.temperature == 0.4:
            return "I rewrote everything and lost your precious placeholders."
        return req.messages[0]["content"].replace("Saved", "Stashed")

    client = FakeTeacherClient(script=flaky)
    outcome = harmonize_record(rec, client, make_prompts(), "deepseek-v4-flash-20260610", "Olivia")
    assert not outcome.dropped
    texts = [m.content for m in outcome.record.messages if m.role == "assistant"]
    assert any("/tmp/list.txt" in t and "42" in t and "Stashed" in t for t in texts)
    # Deterministic ladder: retry at 0.5 only for the two turns whose placeholders
    # got eaten; the short third turn is skip-guarded (below _MIN_PROSE_CHARS) and
    # never reaches the paraphraser at all.
    assert [r.temperature for r in client.requests] == [0.4, 0.5, 0.4, 0.5]


def test_harmonize_skips_placeholder_dense_turns():
    # A turn that mostly quotes protected content (code fence + path) has no voice
    # to harmonize — it must be kept verbatim without a paraphrase call, because
    # placeholder-dense inputs are where restore failures come from.
    quoted = "Done. Here is `lists/x.txt`:\n\n```\nalpha\nbeta\ngamma\n```\n"
    rec = banter("h1s", quoted)

    # Any turn that does reach the paraphraser gets identity-echoed and recorded.
    calls = []

    def echo(req: ChatRequest) -> str:
        calls.append(req)
        return req.messages[0]["content"]

    outcome = harmonize_record(
        rec, FakeTeacherClient(script=echo), make_prompts(), "deepseek-v4-flash-20260610", "Olivia"
    )
    assert not outcome.dropped
    texts = [m.content for m in outcome.record.messages if m.role == "assistant"]
    assert any(t == quoted for t in texts)  # kept byte-verbatim, never sent out
    assert all(quoted not in c.messages[0]["content"] for c in calls)


def test_lane_b_records_skip_harmonize():
    rec = banter("h2", "text").model_copy(
        update={"provenance": banter("h2", "text").provenance.model_copy(update={"lane": "b"})}
    )
    client = FakeTeacherClient(script=lambda r: "SHOULD NOT BE CALLED")
    outcome = harmonize_record(rec, client, make_prompts(), "deepseek-v4-flash-20260610", "Olivia")
    assert not outcome.dropped
    assert client.requests == []


def test_rubric_selection_is_voice_aware():
    # Lane-C Olivia chats -> quality rubric; character-RP records -> character rubric.
    # Judging a character on olivia_voice is meaningless, so voice must pick the rubric.
    from aviary.gates.judge import Rubric
    from aviary.gates.pipeline import _rubric_for

    quality = Rubric(name="quality", axes={}, threshold=3.5)
    character = Rubric(name="character_rp", axes={}, threshold=3.5)
    rubrics = {"c": quality, "c_character": character}

    olivia = banter("r1", "hello there")  # lane c, speaker Olivia
    rp = olivia.model_copy(
        update={
            "messages": [
                m.model_copy(update={"speaker": "Aliya"}) if m.role == "assistant" else m
                for m in olivia.messages
            ]
        }
    )
    assert _rubric_for(olivia, rubrics, "Olivia").name == "quality"
    assert _rubric_for(rp, rubrics, "Olivia").name == "character_rp"


def test_lane_c_character_rp_is_protected_from_olivia_voice():
    # A lane-C record where the character side plays a fiction character (speaker !=
    # Olivia) must NOT be harmonized — paraphrasing it into Olivia's voice would leak
    # the assistant persona into roleplay. Same protection as lane B, keyed on voice.
    base = banter("h3", "The Cyclones took everything from me, and still I sail.")
    rp = base.model_copy(
        update={
            "messages": [
                m.model_copy(update={"speaker": "Aliya"}) if m.role == "assistant" else m
                for m in base.messages
            ]
        }
    )
    client = FakeTeacherClient(script=lambda r: "SHOULD NOT BE CALLED")
    outcome = harmonize_record(rp, client, make_prompts(), "deepseek-v4-flash-20260610", "Olivia")
    assert not outcome.dropped
    assert client.requests == []  # RP character voice never sent to the paraphraser

    # ...but an Olivia-voiced lane-C record in the same lane still IS harmonized.
    called = FakeTeacherClient(script=lambda r: r.messages[0]["content"])
    out2 = harmonize_record(base, called, make_prompts(), "deepseek-v4-flash-20260610", "Olivia")
    assert not out2.dropped
    assert called.requests  # Olivia's voice is harmonizable


def _good_judge(req: ChatRequest) -> str:
    if "data-quality judge" in req.system:
        return json.dumps(
            {"scores": {"olivia_voice": 5, "coherence": 5, "helpfulness": 4, "naturalness": 4}}
        )
    return req.messages[0]["content"]  # harmonizer: identity


def test_gate_pipeline_end_to_end(tmp_path):
    from aviary.io.jsonl import write_jsonl

    store = RunStore("testrun", root=tmp_path)
    good = banter("g1", "Your calendar is a crime and I have receipts.")
    dup = banter("g2", "Your calendar is a crime and I have receipts.")
    selfid = banter("g3", "As an AI, I have no calendar opinions.")
    degenerate = banter("g4", "ok").model_copy(
        update={
            "messages": [
                Message(role="user", content="hi"),
                Message(role="assistant", speaker="Olivia", content="same reply"),
                Message(role="user", content="hi again"),
                Message(role="assistant", speaker="Olivia", content="same reply"),
            ]
        }
    )
    write_jsonl(store.raw("c"), [good, dup, selfid, degenerate])

    patterns = load_patterns(
        REPO / "gates" / "scrub" / "denylist.yaml", REPO / "gates" / "scrub" / "pii_patterns.yaml"
    )
    rubrics = {"c": Rubric.load(REPO / "datagen" / "persona" / "olivia" / "quality.rubric.yaml")}
    stats = run_gates(
        store,
        default_resolver({}),
        rubrics,
        patterns,
        ROSTER,
        FakeTeacherClient(script=_good_judge),
        make_prompts(),
        persona_speaker="Olivia",
        judge_workers=4,  # parallel judging must keep outcomes/order deterministic
        harmonize_workers=4,  # ...and so must parallel harmonize
    )
    assert stats.total == 4
    assert stats.kept == 1
    kept = list(read_jsonl(store.gated_kept(), ConversationRecord))
    assert kept[0].provenance.record_id == "g1"
    assert kept[0].gate_state.harmonized

    rejected = list(read_jsonl(store.gated_rejected(), ConversationRecord))
    reasons = {r.provenance.record_id: r.gate_state.drop_reason for r in rejected}
    assert reasons == {"g2": "dedupe", "g3": "scrub", "g4": "verify"}


def test_run_gates_harmonize_parallel_preserves_order_and_processes_all(tmp_path):
    # Harmonize fans out over records concurrently (harmonize_workers > 1). The
    # outcome list must stay aligned to input order and every record must be
    # processed exactly once — the regression risk when the serial loop became a
    # TeacherPool. Distinct replies so none dedupe; all keepable so all harmonize.
    from aviary.io.jsonl import write_jsonl

    store = RunStore("harmonrun", root=tmp_path)
    replies = [
        "Your calendar is a crime and I have receipts.",
        "The kettle has been judging me since Tuesday, frankly.",
        "I reorganized the whole bookshelf by mood and chaos won handily.",
        "Pigeons are just city seagulls with substantially worse public relations.",
        "I named the bug in my code Gerald and now I cannot delete him.",
    ]
    records = [banter(f"h{i}", r) for i, r in enumerate(replies)]
    write_jsonl(store.raw("c"), records)

    patterns = load_patterns(
        REPO / "gates" / "scrub" / "denylist.yaml", REPO / "gates" / "scrub" / "pii_patterns.yaml"
    )
    rubrics = {"c": Rubric.load(REPO / "datagen" / "persona" / "olivia" / "quality.rubric.yaml")}
    stats = run_gates(
        store,
        default_resolver({}),
        rubrics,
        patterns,
        ROSTER,
        FakeTeacherClient(script=_good_judge),
        make_prompts(),
        persona_speaker="Olivia",
        judge_workers=4,
        harmonize_workers=8,  # more workers than records: all run concurrently
    )
    assert stats.total == len(records)
    assert stats.kept == len(records)
    kept = list(read_jsonl(store.gated_kept(), ConversationRecord))
    # Order preserved despite concurrent execution, and every record harmonized once.
    assert [r.provenance.record_id for r in kept] == [f"h{i}" for i in range(len(records))]
    assert all(r.gate_state.harmonized for r in kept)


def test_run_gates_isolates_stage_errors(tmp_path):
    # A verifier that blows up on one record must drop only that record; the run
    # completes and writes outputs (paid pipeline: no all-or-nothing aborts).
    from aviary.io.jsonl import write_jsonl

    store = RunStore("errrun", root=tmp_path)
    good = banter("ok", "Your calendar is a crime and I have receipts.")
    boom = banter("boom", "Your inbox is also a crime, differently.")
    write_jsonl(store.raw("c"), [good, boom])

    lanec = sorted((REPO / "verifiers" / "lanec").glob("*.py"))

    def resolver(rec):
        if rec.provenance.record_id == "boom":
            raise RuntimeError("verifier resolution blew up")
        return lanec

    patterns = load_patterns(
        REPO / "gates" / "scrub" / "denylist.yaml", REPO / "gates" / "scrub" / "pii_patterns.yaml"
    )
    rubrics = {"c": Rubric.load(REPO / "datagen" / "persona" / "olivia" / "quality.rubric.yaml")}
    stats = run_gates(
        store,
        resolver,
        rubrics,
        patterns,
        ROSTER,
        FakeTeacherClient(script=_good_judge),
        make_prompts(),
        persona_speaker="Olivia",
    )
    assert stats.total == 2
    reasons = {
        r.provenance.record_id: r.gate_state.drop_reason
        for r in read_jsonl(store.gated_rejected(), ConversationRecord)
    }
    assert reasons["boom"] == "error"
    kept_ids = {r.provenance.record_id for r in read_jsonl(store.gated_kept(), ConversationRecord)}
    assert "ok" in kept_ids
