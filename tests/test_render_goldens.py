"""Golden-file rule: render/goldens/ locks serializer output byte-for-byte.

If these tests fail, the default assumption is that the CODE is wrong. Goldens
change only with an explicit, human-approved serializer contract bump.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aviary.render.dpo import pair_siblings
from aviary.render.serializer import ThoughtMode, render_conversation, render_next_speaker_samples
from aviary.schema.records import ConversationRecord

REPO = Path(__file__).resolve().parents[1]
REC_DIR = REPO / "tests" / "fixtures" / "records"
GOLD_DIR = REPO / "render" / "goldens"


def load_record(name: str) -> ConversationRecord:
    return ConversationRecord.model_validate_json((REC_DIR / f"{name}.record.json").read_text())


def load_golden(name: str) -> dict:
    return json.loads((GOLD_DIR / f"{name}.golden.json").read_text())


CONVERSATION_CASES = [
    ("plain_chat", ThoughtMode.WITH),
    ("plain_chat", ThoughtMode.WITHOUT),
    ("single_tool", ThoughtMode.WITH),
    ("recovered_error", ThoughtMode.WITH),
    ("recovered_error", ThoughtMode.WITHOUT),
    ("scene_thoughts", ThoughtMode.WITH),
    ("scene_thoughts", ThoughtMode.WITHOUT),
]


@pytest.mark.parametrize("name,mode", CONVERSATION_CASES)
def test_conversation_matches_golden(name: str, mode: ThoughtMode):
    rendered = render_conversation(load_record(name), mode)
    golden = load_golden(f"{name}.{mode.value}")
    assert rendered.text == golden["text"]
    assert [list(s) for s in rendered.train_spans] == golden["train_spans"]
    assert rendered.meta == golden["meta"]


def test_nsp_matches_goldens():
    samples = render_next_speaker_samples(load_record("scene_thoughts"))
    goldens = sorted(GOLD_DIR.glob("scene_thoughts.nsp_*.golden.json"))
    assert len(samples) == len(goldens) == 3
    for i, sample in enumerate(samples):
        golden = load_golden(f"scene_thoughts.nsp_{i}")
        assert sample.text == golden["text"]
        assert [list(s) for s in sample.train_spans] == golden["train_spans"]


def test_nsp_only_for_multi_speaker():
    assert render_next_speaker_samples(load_record("plain_chat")) == []


def test_dpo_pair_matches_golden():
    pairs = pair_siblings(
        [load_record("dpo_chosen")], [load_record("dpo_rejected")], min_margin=1.0
    )
    assert len(pairs) == 1
    golden = load_golden("dpo_pair")
    assert pairs[0].prompt_text == golden["prompt_text"]
    assert pairs[0].chosen_text == golden["chosen_text"]
    assert pairs[0].rejected_text == golden["rejected_text"]


def test_dpo_margin_gate():
    assert (
        pair_siblings([load_record("dpo_chosen")], [load_record("dpo_rejected")], min_margin=5.0)
        == []
    )


def test_every_golden_has_a_fixture_and_vice_versa():
    fixture_names = {p.name.removesuffix(".record.json") for p in REC_DIR.glob("*.record.json")}
    for golden in GOLD_DIR.glob("*.golden.json"):
        base = golden.name.removesuffix(".golden.json").split(".")[0]
        if base == "dpo_pair":
            continue
        assert base in fixture_names, f"golden {golden.name} has no fixture record"
    assert fixture_names, "no fixture records found"
