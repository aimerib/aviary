"""Lane D adapter tests. ALL data here is synthetic, authored in this file —
the radioactive-data rule forbids real personal content anywhere in the repo."""

from __future__ import annotations

import json

import pytest

from aviary.lanes.d_personal.adapter import (
    PARSERS,
    LaneDConfig,
    SourceConfig,
    parse_all,
)

OWNER = "Sam"  # synthetic persons only
FRIEND = "Rio"


def _line(conv, ts, sender, text):
    return json.dumps(
        {"conversation_id": conv, "ts": ts, "sender": sender, "text": text}
    )


def _export(tmp_path, lines):
    p = tmp_path / "export.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return p


def _cfg(path, **overrides) -> LaneDConfig:
    base = {
        "sources": [
            {"id": "chat", "parser": "generic_jsonl", "path": str(path), "assistant_sender": OWNER}
        ],
    }
    return LaneDConfig.model_validate(base | overrides)


def test_generic_jsonl_normalizes_roles_ts_and_family(tmp_path):
    path = _export(
        tmp_path,
        [
            _line("c1", "2026-03-14T21:07:03Z", FRIEND, "you up? that movie was unhinged"),
            _line("c1", "2026-03-14T21:09:41Z", OWNER, "extremely. the boat scene??"),
            _line("c1", "2026-03-14T21:10:02Z", FRIEND, "I yelled out loud"),
        ],
    )
    recs = list(parse_all(_cfg(path), run_id="t"))
    assert len(recs) == 1
    rec = recs[0]
    assert rec.provenance.lane == "d"
    assert rec.provenance.family == "chat/2026-03"
    assert rec.provenance.source.kind == "personal_stream"
    assert [m.role for m in rec.messages] == ["user", "assistant", "user"]
    assert rec.messages[1].speaker == OWNER
    assert rec.messages[0].ts == "2026-03-14T21:07:03Z"  # timestamps preserved
    assert rec.system == ""  # persona is a build-target concern


def test_session_gap_splits_records_and_short_segments_drop(tmp_path):
    path = _export(
        tmp_path,
        [
            # evening session
            _line("c1", "2026-03-14T21:00:00Z", FRIEND, "dinner went sideways"),
            _line("c1", "2026-03-14T21:02:00Z", OWNER, "sideways how"),
            # 3-day silence -> new record
            _line("c1", "2026-03-17T09:30:00Z", FRIEND, "ok update on the dinner thing"),
            _line("c1", "2026-03-17T09:31:12Z", OWNER, "finally. go"),
            # lone message after another long gap -> below min, dropped
            _line("c1", "2026-03-29T23:59:00Z", FRIEND, "asleep?"),
        ],
    )
    recs = list(parse_all(_cfg(path), run_id="t"))
    assert len(recs) == 2
    assert [len(r.messages) for r in recs] == [2, 2]
    segs = [r.provenance.source.detail["segment"] for r in recs]
    assert segs == [0, 1]
    ids = {r.provenance.record_id for r in recs}
    assert len(ids) == 2  # segment index feeds record identity


def test_holdout_is_time_blocked_by_family(tmp_path):
    path = _export(
        tmp_path,
        [
            _line("c1", "2026-03-01T10:00:00Z", FRIEND, "march thread"),
            _line("c1", "2026-03-01T10:01:00Z", OWNER, "yep"),
            _line("c2", "2026-05-02T10:00:00Z", FRIEND, "may thread"),
            _line("c2", "2026-05-02T10:01:00Z", OWNER, "yep"),
        ],
    )
    recs = list(parse_all(_cfg(path, holdout_families=["chat/2026-05"]), run_id="t"))
    by_family = {r.provenance.family: r.provenance.holdout for r in recs}
    assert by_family == {"chat/2026-03": False, "chat/2026-05": True}


def test_max_messages_caps_a_record(tmp_path):
    lines = [
        _line("c1", f"2026-04-01T10:{i:02d}:00Z", OWNER if i % 2 else FRIEND, f"msg {i}")
        for i in range(10)
    ]
    path = _export(tmp_path, lines)
    recs = list(parse_all(_cfg(path, max_messages_per_record=4), run_id="t"))
    assert [len(r.messages) for r in recs] == [4, 4, 2]


def test_malformed_line_and_unknown_parser_fail_loudly(tmp_path):
    path = _export(tmp_path, [json.dumps({"conversation_id": "c1", "ts": "2026-01-01T00:00:00Z"})])
    with pytest.raises(ValueError, match="missing fields"):
        list(parse_all(_cfg(path), run_id="t"))

    bad = _cfg(path)
    bad.sources[0].parser = "carrier_pigeon"
    with pytest.raises(KeyError, match="carrier_pigeon"):
        list(parse_all(bad, run_id="t"))


def test_stub_parsers_are_registered_but_refuse():
    assert {"imessage", "discord", "journal"} <= set(PARSERS)
    stub_cfg = SourceConfig(
        id="x", parser="imessage", path="/nonexistent", assistant_sender=OWNER
    )
    with pytest.raises(NotImplementedError, match="imessage"):
        list(PARSERS["imessage"].parse(stub_cfg, LaneDConfig(), "t"))
