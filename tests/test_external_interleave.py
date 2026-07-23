"""Adapting external RP rows into records and rendering them through THE
serializer. The contract being defended: interleaved text must be byte-identical
in template to our own corpus, so these tests compare against
`render_conversation` output rather than asserting on hand-written strings.
"""

from __future__ import annotations

import json

import pytest

from aviary.external.interleave import prepare, split_think, to_record
from aviary.render.serializer import ThoughtMode, render_conversation

THINK = "SCENE: tavern\nCHARACTERS: Mara\nPLAN: greet"


def row(*turns: tuple[str, str], rid: str = "r1") -> dict:
    return {"id": rid, "conversations": [{"from": f, "value": v} for f, v in turns]}


def full_row(rid: str = "r1") -> dict:
    return row(
        ("system", "You are Mara, a tired innkeeper."),
        ("human", "I walk in."),
        ("gpt", f"<think>\n{THINK}\n</think>\nMara looks up."),
        rid=rid,
    )


# --- think extraction --------------------------------------------------------


def test_split_think_lifts_leading_block():
    thought, rest = split_think(f"<think>\n{THINK}\n</think>\nMara looks up.")
    assert thought == THINK
    assert rest == "Mara looks up."


def test_split_think_ignores_non_leading_block():
    body = "Mara looks up.\n<think>\nlate\n</think>"
    assert split_think(body) == (None, body)


def test_split_think_passthrough_without_block():
    assert split_think("just prose") == (None, "just prose")


# --- adaptation --------------------------------------------------------------


def test_system_turn_becomes_record_system_not_a_message():
    # The serializer raises if a system role appears inside messages.
    rec = to_record(full_row(), name="x", run_id="external-x")
    assert rec is not None
    assert rec.system == "You are Mara, a tired innkeeper."
    assert all(m.role != "system" for m in rec.messages)


def test_roles_are_mapped():
    rec = to_record(full_row(), name="x", run_id="external-x")
    assert [m.role for m in rec.messages] == ["user", "assistant"]


def test_thought_is_preserved_on_the_record():
    # Preserved so --with-thoughts stays possible; simply not rendered by default.
    rec = to_record(full_row(), name="x", run_id="external-x")
    assistant = [m for m in rec.messages if m.role == "assistant"][0]
    assert assistant.thought == THINK
    assert "<think>" not in assistant.content


def test_provenance_marks_external_origin():
    rec = to_record(full_row(rid="abc"), name="rp", run_id="external-rp")
    assert rec.provenance.family == "external:rp"
    assert rec.provenance.record_id == "abc"
    assert rec.provenance.source.detail["external"] == "rp"


def test_row_without_system_prompt_is_rejected():
    # The character card would otherwise be embedded as the first assistant turn.
    assert to_record(row(("human", "hi"), ("gpt", "hello")), name="x", run_id="r") is None
    assert to_record(row(("system", "   "), ("gpt", "hi")), name="x", run_id="r") is None


def test_row_without_assistant_turn_is_rejected():
    assert to_record(row(("system", "s"), ("human", "hi")), name="x", run_id="r") is None


# --- rendering goes through the choke point ---------------------------------


def test_rendered_text_matches_the_serializer_exactly(tmp_path, monkeypatch):
    monkeypatch.setenv("AVIARY_DATA_DIR", str(tmp_path / "data"))
    root = tmp_path / "data" / "external" / "rp"
    root.mkdir(parents=True)
    (root / "cleaned.jsonl").write_text(json.dumps(full_row()) + "\n")

    report = prepare("rp")
    assert report["rendered"] == 1

    written = json.loads((root / "rendered" / "train_no_thoughts.jsonl").read_text())
    expected = render_conversation(
        to_record(full_row(), name="rp", run_id="external-rp"), ThoughtMode.WITHOUT
    )
    assert written["text"] == expected.text


def test_default_strips_think_and_flag_restores_it(tmp_path, monkeypatch):
    monkeypatch.setenv("AVIARY_DATA_DIR", str(tmp_path / "data"))
    root = tmp_path / "data" / "external" / "rp"
    root.mkdir(parents=True)
    (root / "cleaned.jsonl").write_text(json.dumps(full_row()) + "\n")

    prepare("rp")
    stripped = (root / "rendered" / "train_no_thoughts.jsonl").read_text()
    assert "<think>" not in stripped
    assert "Mara looks up." in stripped

    prepare("rp", with_thoughts=True)
    kept = (root / "rendered" / "train_with_thoughts.jsonl").read_text()
    assert "<think>" in kept and "SCENE: tavern" in kept


def test_skips_are_reported_not_silent(tmp_path, monkeypatch):
    monkeypatch.setenv("AVIARY_DATA_DIR", str(tmp_path / "data"))
    root = tmp_path / "data" / "external" / "rp"
    root.mkdir(parents=True)
    with (root / "cleaned.jsonl").open("w") as f:
        f.write(json.dumps(full_row("a")) + "\n")
        f.write(json.dumps(row(("human", "hi"), ("gpt", "no system"), rid="b")) + "\n")

    report = prepare("rp")
    assert report["source_rows"] == 2
    assert report["rendered"] == 1
    assert sum(report["skipped"].values()) == 1


def test_mix_report_quantifies_dilution(tmp_path, monkeypatch):
    # The whole point of --against: make the interleave ratio a number, not a guess.
    monkeypatch.setenv("AVIARY_DATA_DIR", str(tmp_path / "data"))
    root = tmp_path / "data" / "external" / "rp"
    root.mkdir(parents=True)
    (root / "cleaned.jsonl").write_text(json.dumps(full_row()) + "\n")

    run_rendered = tmp_path / "data" / "myrun" / "rendered"
    run_rendered.mkdir(parents=True)
    (run_rendered / "train_no_thoughts.jsonl").write_text(
        json.dumps({"text": "x" * 4000}) + "\n"
    )

    report = prepare("rp", against_run="myrun")
    mix = report["mix_against"]
    assert mix["aviary_records"] == 1
    assert mix["aviary_approx_tokens"] == 1000
    assert pytest.approx(mix["aviary_token_share"] + mix["external_token_share"]) == 1.0


def test_missing_cleaned_corpus_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("AVIARY_DATA_DIR", str(tmp_path / "data"))
    with pytest.raises(FileNotFoundError, match="cleaned.jsonl"):
        prepare("nope")
