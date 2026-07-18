from __future__ import annotations

import random
from pathlib import Path

from aviary.gates.verify import run_verifier
from aviary.lanes.c_selfplay.driver import END_SENTINEL, LaneCConfig, run_selfplay
from aviary.lanes.c_selfplay.seeds import Seed, load_inline_seeds
from aviary.lanes.c_selfplay.usersim import UserSimPersona, inject_typos, load_personas
from aviary.teacher.client import ChatRequest
from aviary.teacher.fake import FakeTeacherClient
from aviary.teacher.prompts import PromptSet

REPO = Path(__file__).resolve().parents[1]

SEED = Seed(
    seed_id="banter-001",
    family="banter",
    character_name="Olivia",
    card="You are Olivia.\n\n## Scenario\nLate night chat.",
    scenario="Late night chat.",
    user_goal="vent about the day",
)

MODELS = {"user_sim": "deepseek-v4-flash-20260610", "character": "glm-5-20260430"}


def prompts() -> PromptSet:
    return PromptSet.load({"lanec_user_sim": REPO / "datagen" / "prompts" / "lanec_user_sim.md"})


def persona(**overrides) -> UserSimPersona:
    base = {"id": "t", "description": "test", "target_len_words": (2, 10)}
    return UserSimPersona.model_validate(base | overrides)


def test_typo_injector_is_deterministic():
    text = "honestly the whole meeting could have been an email about nothing"
    out1 = inject_typos(text, 0.5, random.Random(99))
    out2 = inject_typos(text, 0.5, random.Random(99))
    out3 = inject_typos(text, 0.5, random.Random(100))
    assert out1 == out2
    assert out1 != text
    assert out3 != out1
    assert inject_typos(text, 0.0, random.Random(1)) == text


def test_typo_injector_spares_protected_tokens():
    text = "check https://example.com/x and /tmp/file.txt maybe"
    out = inject_typos(text, 1.0, random.Random(5))
    assert "https://example.com/x" in out
    assert "/tmp/file.txt" in out


def test_selfplay_user_ends_chat():
    turns = iter(["hey", "rough day", END_SENTINEL])

    def script(req: ChatRequest) -> str:
        if "HUMAN USER" in req.system:
            return next(turns)
        return f"A different sympathetic reply number {len(req.messages)}, tell me more about it."

    rec = run_selfplay(
        SEED,
        persona(),
        LaneCConfig(min_turns=8, max_turns=8),
        FakeTeacherClient(script=script),
        MODELS,
        prompts(),
        run_id="t",
        prompt_set_hash="h",
        rng=random.Random(1),
    )
    assert rec.provenance.source.detail["stop_reason"] == "user_ended"
    assert len([m for m in rec.messages if m.role == "user"]) == 2
    assert rec.provenance.teachers == MODELS
    assert rec.provenance.lane == "c"


def test_selfplay_degenerate_loop_stops():
    def script(req: ChatRequest) -> str:
        if "HUMAN USER" in req.system:
            return "and then what happened next tell me"
        return "The same seven words repeat here always, the same seven words repeat here always."

    rec = run_selfplay(
        SEED,
        persona(),
        LaneCConfig(min_turns=10, max_turns=10),
        FakeTeacherClient(script=script),
        MODELS,
        prompts(),
        run_id="t",
        prompt_set_hash="h",
        rng=random.Random(2),
    )
    assert rec.provenance.source.detail["stop_reason"] == "degenerate"
    # ...and the structural verifier rejects it downstream
    result = run_verifier(REPO / "verifiers" / "lanec" / "turn_dynamics.py", rec)
    assert not result.passed


def test_selfplay_empty_turn_ends_cleanly():
    # A reasoning-model empty reply must never enter the transcript — it would be sent
    # back as empty-content history and 400 some providers. The conversation ends and
    # the now-dangling user turn is dropped so the record closes on a full exchange.
    def script(req: ChatRequest) -> str:
        if "HUMAN USER" in req.system:
            return "so tell me about the crash"
        return "   " if len(req.messages) >= 3 else "I remember the storm, the ship breaking apart."

    rec = run_selfplay(
        SEED,
        persona(),
        LaneCConfig(min_turns=8, max_turns=8),
        FakeTeacherClient(script=script),
        MODELS,
        prompts(),
        run_id="t",
        prompt_set_hash="h",
        rng=random.Random(3),
    )
    assert rec.provenance.source.detail["stop_reason"] == "empty_turn"
    assert all(m.content.strip() for m in rec.messages)  # no empty turn leaked in
    users = [m for m in rec.messages if m.role == "user"]
    assts = [m for m in rec.messages if m.role == "assistant"]
    assert len(users) == len(assts)  # dangling user turn dropped; ends on an exchange


def test_selfplay_max_turns_and_alternation():
    def script(req: ChatRequest) -> str:
        if "HUMAN USER" in req.system:
            return f"user message number {len(req.messages)} with fresh content"
        return f"a distinct reply about topic {len(req.messages)} with new words each time"

    cfg = LaneCConfig(min_turns=4, max_turns=4)
    rec = run_selfplay(
        SEED,
        persona(),
        cfg,
        FakeTeacherClient(script=script),
        MODELS,
        prompts(),
        run_id="t",
        prompt_set_hash="h",
        rng=random.Random(3),
    )
    assert rec.provenance.source.detail["stop_reason"] == "max_turns"
    roles = [m.role for m in rec.messages]
    assert roles == ["user", "assistant"] * 4
    result = run_verifier(REPO / "verifiers" / "lanec" / "turn_dynamics.py", rec)
    assert result.passed


def test_inline_seeds_and_personas_load():
    seeds_yaml = REPO / "datagen" / "configs" / "lane_c.yaml"
    if seeds_yaml.exists():
        seeds = load_inline_seeds(seeds_yaml, "You are Olivia.")
        assert all(s.card.startswith("You are Olivia.") for s in seeds)
    personas = load_personas(REPO / "datagen" / "persona" / "user_sims")
    assert {"lazy_texter", "engaged_rper", "task_asker"} <= set(personas)
    assert personas["lazy_texter"].typo_rate > 0
