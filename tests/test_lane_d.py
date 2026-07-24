"""Lane D adapter tests. ALL data here is synthetic, authored in this file —
the radioactive-data rule forbids real personal content anywhere in the repo."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from aviary.lanes.d_personal.adapter import (
    PARSERS,
    LaneDConfig,
    SourceConfig,
    _demojibake,
    parse_all,
)

OWNER = "Sam"  # synthetic persons only
FRIEND = "Rio"


def _line(conv, ts, sender, text):
    return json.dumps({"conversation_id": conv, "ts": ts, "sender": sender, "text": text})


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
    stub_cfg = SourceConfig(id="x", parser="discord", path="/nonexistent", assistant_sender=OWNER)
    with pytest.raises(NotImplementedError, match="discord"):
        list(PARSERS["discord"].parse(stub_cfg, LaneDConfig(), "t"))


# --- role anchoring ----------------------------------------------------------


def test_exactly_one_role_anchor_is_required():
    for kwargs in ({}, {"assistant_sender": OWNER, "owner_sender": FRIEND}):
        with pytest.raises(ValueError, match="exactly one"):
            SourceConfig(id="x", parser="imessage", path="/nonexistent", **kwargs)


def test_owner_sender_puts_the_owner_in_the_user_seat():
    # The companion mapping: the model is never trained to produce the owner's
    # turns. SOUL.md's "never impersonates the user" starts here, not at the judge.
    src = SourceConfig(id="x", parser="imessage", path="/x", owner_sender=OWNER)
    assert src.role_for(OWNER) == "user"
    assert src.role_for(FRIEND) == "assistant"
    assert src.role_for("Anyone Else") == "assistant"

    legacy = SourceConfig(id="x", parser="generic_jsonl", path="/x", assistant_sender=OWNER)
    assert legacy.role_for(OWNER) == "assistant"
    assert legacy.role_for(FRIEND) == "user"


# --- imessage-exporter format ------------------------------------------------

APPLE_2026_03_14 = 795_222_423_000_000_000  # ns since 2001-01-01; ~2026-03-14


def _im(sender, offset_min, parts, *, kind="message"):
    return json.dumps(
        {
            "guid": f"g{offset_min}",
            "timestamp": APPLE_2026_03_14 + offset_min * 60 * 1_000_000_000,
            "sender": sender,
            "is_from_me": sender == OWNER,
            "service": "iMessage",
            "type": kind,
            "parts": parts,
        }
    )


def _text(t):
    return [{"type": "text", "text": t}]


def _im_export(tmp_path, lines, thread="Rio"):
    d = tmp_path / "imessage"
    d.mkdir(exist_ok=True)
    (d / f"{thread}.jsonl").write_text("\n".join(lines) + "\n")
    return d


def _im_cfg(path, **overrides):
    base = {
        "sources": [{"id": "im", "parser": "imessage", "path": str(path), "owner_sender": OWNER}]
    }
    return LaneDConfig.model_validate(base | overrides)


def test_imessage_parses_a_thread_into_a_record(tmp_path):
    d = _im_export(
        tmp_path,
        [
            _im(FRIEND, 0, _text("the trail was pure mud")),
            _im(OWNER, 2, _text("and yet you sound delighted")),
            _im(FRIEND, 3, _text("I lost a shoe")),
        ],
    )
    recs = list(parse_all(_im_cfg(d), "t"))
    assert len(recs) == 1
    rec = recs[0]
    assert [m.role for m in rec.messages] == ["assistant", "user", "assistant"]
    assert [m.speaker for m in rec.messages] == [FRIEND, OWNER, FRIEND]
    assert rec.provenance.lane == "d"
    assert rec.provenance.family == "im/2026-03"  # time-blocked holdout unit
    assert rec.provenance.source.detail["conversation_id"] == "Rio"


def test_apple_epoch_timestamps_become_iso(tmp_path):
    d = _im_export(tmp_path, [_im(FRIEND, 0, _text("hi")), _im(OWNER, 1, _text("hello"))])
    rec = next(iter(parse_all(_im_cfg(d), "t")))
    assert rec.messages[0].ts.startswith("2026-03-14T")


def test_announcements_and_textless_messages_are_skipped(tmp_path):
    d = _im_export(
        tmp_path,
        [
            _im(FRIEND, 0, _text("real message")),
            _im(FRIEND, 1, [], kind="announcement"),
            # A bare attachment is not a turn, and must NOT become a "[image]" token.
            _im(
                FRIEND,
                2,
                [{"type": "segments", "segments": [{"mime_type": "image/png", "path": "/x"}]}],
            ),
            _im(OWNER, 3, _text("reply")),
        ],
    )
    rec = next(iter(parse_all(_im_cfg(d), "t")))
    assert [m.content for m in rec.messages] == ["real message", "reply"]
    assert "image" not in rec.model_dump_json()


def test_edited_messages_use_the_final_revision(tmp_path):
    d = _im_export(
        tmp_path,
        [
            _im(
                FRIEND,
                0,
                [
                    {
                        "type": "edited",
                        "history": [
                            {"text": "So sadly I need just red", "timestamp": 1},
                            {"text": "No sadly I need just red", "timestamp": 2},
                        ],
                    }
                ],
            ),
            _im(OWNER, 1, _text("got it")),
        ],
    )
    rec = next(iter(parse_all(_im_cfg(d), "t")))
    assert rec.messages[0].content == "No sadly I need just red"


def test_shared_link_secrets_are_stripped_by_default(tmp_path):
    # Real exports carry live capability URLs (a 1Password share link's fragment IS
    # the credential). Scrub looks for PII, not for secret-bearing URLs.
    secret = "https://share.1password.com/s?a=b#SECRETFRAGMENT"
    d = _im_export(
        tmp_path,
        [
            _im(FRIEND, 0, [{"type": "url", "url": secret, "title": "shared item"}]),
            _im(OWNER, 1, _text("thanks")),
        ],
    )
    rec = next(iter(parse_all(_im_cfg(d), "t")))
    assert "SECRETFRAGMENT" not in rec.model_dump_json()
    assert "a=b" not in rec.model_dump_json()
    assert rec.messages[0].content == "https://share.1password.com/s"

    kept = next(iter(parse_all(_im_cfg(d, keep_url_query=True), "t")))
    assert kept.messages[0].content == secret


def test_object_replacement_char_is_stripped(tmp_path):
    d = _im_export(
        tmp_path,
        [_im(FRIEND, 0, _text("look ￼ at this")), _im(OWNER, 1, _text("ok"))],
    )
    rec = next(iter(parse_all(_im_cfg(d), "t")))
    assert "￼" not in rec.messages[0].content


def test_one_sided_runs_are_dropped(tmp_path):
    # The known export artifact: a window where outgoing messages are blank. A
    # monologue is not a conversation and must not be trained on as one.
    d = _im_export(tmp_path, [_im(FRIEND, 0, _text("a")), _im(FRIEND, 1, _text("b"))])
    assert list(parse_all(_im_cfg(d), "t")) == []
    assert len(list(parse_all(_im_cfg(d, require_both_roles=False), "t"))) == 1


def test_thread_selection_and_min_size(tmp_path):
    d = tmp_path / "imessage"
    d.mkdir()
    for name in ("Rio", "Noise"):
        (d / f"{name}.jsonl").write_text(
            "\n".join([_im(FRIEND, 0, _text("hi")), _im(OWNER, 1, _text("hey"))]) + "\n"
        )

    def threads(**src_overrides):
        cfg = LaneDConfig.model_validate(
            {
                "sources": [
                    {
                        "id": "im",
                        "parser": "imessage",
                        "path": str(d),
                        "owner_sender": OWNER,
                        **src_overrides,
                    }
                ]
            }
        )
        return {r.provenance.source.detail["conversation_id"] for r in parse_all(cfg, "t")}

    assert threads() == {"Rio", "Noise"}
    assert threads(include_threads=["Rio"]) == {"Rio"}
    assert threads(exclude_threads=["Noise"]) == {"Rio"}
    assert threads(min_thread_messages=3) == set()


def test_session_gap_splits_imessage_threads(tmp_path):
    d = _im_export(
        tmp_path,
        [
            _im(FRIEND, 0, _text("morning")),
            _im(OWNER, 1, _text("morning")),
            _im(FRIEND, 60 * 24, _text("next day")),
            _im(OWNER, 60 * 24 + 1, _text("hi again")),
        ],
    )
    recs = list(parse_all(_im_cfg(d, session_gap_minutes=240), "t"))
    assert len(recs) == 2
    assert {r.provenance.source.detail["segment"] for r in recs} == {0, 1}


# --- thread dominance cap + substance floor ----------------------------------


def _threaded_export(tmp_path, sizes: dict[str, int]):
    """One thread per name, `n` two-message sessions each (a day apart)."""
    d = tmp_path / "imessage"
    d.mkdir(exist_ok=True)
    for name, n in sizes.items():
        lines = []
        for i in range(n):
            base = i * 60 * 24 * 3  # 3 days apart -> always a new session
            lines.append(_im(FRIEND, base, _text(f"message {i} from the other side")))
            lines.append(_im(OWNER, base + 1, _text(f"reply {i} from the owner here")))
        (d / f"{name}.jsonl").write_text("\n".join(lines) + "\n")
    return d


def test_thread_cap_limits_any_single_voice(tmp_path):
    # 67% of the real export is one relationship. Uncapped, that thread IS lane D's
    # assistant voice — and harmonize is {} for Sorcha, so nothing downstream fixes it.
    d = _threaded_export(tmp_path, {"Dominant": 80, "Small": 10, "Tiny": 10})

    uncapped = list(parse_all(_im_cfg(d), "t"))
    assert len(uncapped) == 100
    share = sum(1 for r in uncapped if r.provenance.source.detail["conversation_id"] == "Dominant")
    assert share / len(uncapped) == 0.8

    capped = list(
        parse_all(
            LaneDConfig.model_validate(
                {
                    "sources": [
                        {
                            "id": "im",
                            "parser": "imessage",
                            "path": str(d),
                            "owner_sender": OWNER,
                            "max_thread_share": 0.35,
                        }
                    ]
                }
            ),
            "t",
        )
    )
    counts: dict[str, int] = {}
    for r in capped:
        counts[r.provenance.source.detail["conversation_id"]] = (
            counts.get(r.provenance.source.detail["conversation_id"], 0) + 1
        )
    assert max(counts.values()) / len(capped) <= 0.35 + 1e-9
    assert counts["Small"] == 10 and counts["Tiny"] == 10  # small threads untouched


def test_capped_thread_is_strided_not_truncated(tmp_path):
    # A capped thread must stay representative of its whole span; keeping the first
    # N would collapse six years of a relationship into its first months.
    d = _threaded_export(tmp_path, {"Dominant": 40, "Small": 5})
    recs = list(
        parse_all(
            LaneDConfig.model_validate(
                {
                    "sources": [
                        {
                            "id": "im",
                            "parser": "imessage",
                            "path": str(d),
                            "owner_sender": OWNER,
                            "max_thread_share": 0.5,
                        }
                    ]
                }
            ),
            "t",
        )
    )
    segments = sorted(
        r.provenance.source.detail["segment"]
        for r in recs
        if r.provenance.source.detail["conversation_id"] == "Dominant"
    )
    assert len(segments) < 40
    assert segments[0] == 0
    assert segments[-1] > 30  # reaches the far end of the thread, not just the head


def test_substance_floor_drops_acknowledgement_tokens(tmp_path):
    d = _im_export(
        tmp_path,
        [
            _im(FRIEND, 0, _text("k")),
            _im(OWNER, 1, _text("yup")),
            _im(
                FRIEND,
                60 * 24,
                _text(
                    "so the whole thing fell apart at the last minute and I have been sitting "
                    "with it all afternoon trying to work out whether I actually saw it coming"
                ),
            ),
            _im(
                OWNER,
                60 * 24 + 1,
                _text(
                    "that is genuinely awful, tell me what happened from the start and do not "
                    "skip the boring parts, the boring parts are usually where it went wrong"
                ),
            ),
        ],
    )
    assert len(list(parse_all(_im_cfg(d), "t"))) == 2  # both sessions, unfiltered
    kept = list(parse_all(_im_cfg(d, min_record_chars=200, min_mean_message_chars=25), "t"))
    assert len(kept) == 1
    assert "fell apart" in kept[0].messages[0].content


# --- Meta "Download Your Information" (Instagram/Messenger) -------------------

MAX = "Max"  # a second synthetic friend, for owner-across-threads detection


def _mojibake(s: str) -> str:
    """Exactly how Meta mangles it: real UTF-8 bytes reinterpreted as latin-1. The
    parser's _demojibake must invert this on every text field."""
    return s.encode("utf-8").decode("latin-1")


def _ig_ms(offset_min: int) -> int:
    base = datetime(2026, 3, 14, 21, 0, tzinfo=UTC)
    return int((base + timedelta(minutes=offset_min)).timestamp() * 1000)


def _ig_thread(root, thread, participants, messages, *, files=1):
    """messages: (sender, offset_min, content). Written mojibaked and NEWEST-FIRST
    (Meta's order), optionally sharded across message_1..message_N.json."""
    td = root / thread
    td.mkdir(parents=True, exist_ok=True)
    rows = [
        {"sender_name": _mojibake(s), "timestamp_ms": _ig_ms(off), "content": _mojibake(c)}
        for s, off, c in messages
    ]
    rows.reverse()  # newest-first
    shards = [rows[i::files] for i in range(files)]  # each shard keeps Meta's ordering
    for i, shard in enumerate(shards, 1):
        payload = {
            "participants": [{"name": _mojibake(p)} for p in participants],
            "messages": shard,
            "title": _mojibake(participants[0]),
            "thread_path": f"inbox/{thread}",
        }
        (td / f"message_{i}.json").write_text(json.dumps(payload))
    return td


def _ig_cfg(path, **overrides):
    base = {"sources": [{"id": "ig", "parser": "instagram", "path": str(path)}]}
    return LaneDConfig.model_validate(base | overrides)


def _ig_export(tmp_path):
    """Two threads sharing owner Sam — enough for structural owner detection."""
    root = tmp_path / "instagram"
    _ig_thread(
        root,
        "rio_123",
        [FRIEND, OWNER],
        [
            (FRIEND, 0, "the trail was pure mud"),
            (OWNER, 2, "and yet you sound delighted"),
            (FRIEND, 3, "I lost a shoe out there"),
        ],
    )
    _ig_thread(
        root,
        "max_456",
        [MAX, OWNER],
        [(MAX, 0, "cafe plan still on?"), (OWNER, 1, "always")],
    )
    return root


def test_demojibake_restores_meta_double_encoding():
    assert _demojibake(_mojibake("café ☕ résumé — 😀")) == "café ☕ résumé — 😀"
    # Idempotent-safe on already-correct text (codepoint > 255 can't latin-1 encode).
    assert _demojibake("plain ascii") == "plain ascii"
    assert _demojibake("already 😀 unicode") == "already 😀 unicode"


def test_instagram_detects_owner_and_seats_them_as_user(tmp_path):
    recs = list(parse_all(_ig_cfg(_ig_export(tmp_path)), "t"))
    rio = next(r for r in recs if r.provenance.source.detail["conversation_id"] == "rio_123")
    # Sam is in both threads -> the owner -> the USER seat (companion mapping).
    assert [m.role for m in rio.messages] == ["assistant", "user", "assistant"]
    assert [m.speaker for m in rio.messages] == [FRIEND, OWNER, FRIEND]
    assert rio.provenance.lane == "d"
    assert rio.provenance.family == "ig/2026-03"


def test_instagram_reverse_order_is_sorted_and_shards_merge(tmp_path):
    root = tmp_path / "instagram"
    _ig_thread(root, "other_1", [MAX, OWNER], [(MAX, 9, "z last")])  # keep owner detectable
    _ig_thread(
        root,
        "rio_123",
        [FRIEND, OWNER],
        [(FRIEND, 0, "first"), (OWNER, 1, "second"), (FRIEND, 2, "third"), (OWNER, 3, "fourth")],
        files=2,  # sharded across message_1 + message_2, still newest-first within each
    )
    rio = next(
        r
        for r in parse_all(_ig_cfg(root), "t")
        if r.provenance.source.detail["conversation_id"] == "rio_123"
    )
    assert [m.content for m in rio.messages] == ["first", "second", "third", "fourth"]


def test_instagram_drops_auto_strings_and_fixes_unicode(tmp_path):
    root = tmp_path / "instagram"
    _ig_thread(root, "other_1", [MAX, OWNER], [(MAX, 9, "keep owner detectable"), (OWNER, 10, "k")])
    _ig_thread(
        root,
        "rio_123",
        [FRIEND, OWNER],
        [
            (FRIEND, 0, "did you see the café pic ☕"),
            (FRIEND, 1, f"{FRIEND} sent an attachment."),  # Meta auto-string -> drop
            (OWNER, 2, "Reacted ❤ to your message"),  # reaction-as-content -> drop
            (OWNER, 3, "gorgeous"),
        ],
    )
    rio = next(
        r
        for r in parse_all(_ig_cfg(root), "t")
        if r.provenance.source.detail["conversation_id"] == "rio_123"
    )
    assert [m.content for m in rio.messages] == ["did you see the café pic ☕", "gorgeous"]
    assert "sent an attachment" not in rio.model_dump_json()
    assert "\\u00e2" not in rio.model_dump_json()  # no lingering mojibake bytes


def test_instagram_max_participants_skips_group_threads(tmp_path):
    root = tmp_path / "instagram"
    _ig_thread(root, "rio_123", [FRIEND, OWNER], [(FRIEND, 0, "one on one"), (OWNER, 1, "yes")])
    _ig_thread(
        root, "max_456", [MAX, OWNER], [(MAX, 0, "second DM"), (OWNER, 1, "keeps owner clear")]
    )
    _ig_thread(
        root,
        "group_789",
        [FRIEND, MAX, OWNER, "Jo"],
        [(FRIEND, 0, "group chat noise"), (MAX, 1, "so much noise"), (OWNER, 2, "lol")],
    )
    threads = {
        r.provenance.source.detail["conversation_id"]
        for r in parse_all(
            _ig_cfg(
                root,
                sources=[
                    {"id": "ig", "parser": "instagram", "path": str(root), "max_participants": 2}
                ],
            ),
            "t",
        )
    }
    assert threads == {"rio_123", "max_456"}  # the 4-person group is skipped


def test_instagram_refuses_to_guess_an_ambiguous_owner(tmp_path):
    # One lone DM: owner and friend each appear once. Guessing would risk seating the
    # owner as assistant — training the model to produce his turns. Fail loud instead.
    root = tmp_path / "instagram"
    _ig_thread(root, "rio_123", [FRIEND, OWNER], [(FRIEND, 0, "hi"), (OWNER, 1, "hey")])
    with pytest.raises(ValueError, match="owner is ambiguous"):
        list(parse_all(_ig_cfg(root), "t"))


def test_instagram_rejects_a_config_role_anchor():
    with pytest.raises(ValueError, match="resolves the owner"):
        SourceConfig(id="ig", parser="instagram", path="/x", owner_sender=OWNER)


# --- Teams 1-on-1 export -----------------------------------------------------


def _teams_file(root, name, owner, messages):
    """messages: (sender, ts_iso, text)."""
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "me": {"name": owner, "email": "o@x"},
        "participant": {"name": messages[0][0], "email": "p@x"},
        "messages": [{"from": s, "sent": ts, "text": t} for s, ts, t in messages],
    }
    (root / f"{name}.json").write_text(json.dumps(payload))
    return root


def _teams_cfg(path, **overrides):
    base = {"sources": [{"id": "teams", "parser": "teams", "path": str(path)}]}
    return LaneDConfig.model_validate(base | overrides)


def test_teams_uses_me_name_as_owner_in_the_user_seat(tmp_path):
    root = _teams_file(
        tmp_path / "teams",
        "casey",
        OWNER,
        [
            ("Casey", "2026-03-14T21:00:00Z", "can you review the deploy runbook?"),
            (OWNER, "2026-03-14T21:02:00Z", "yep, give me ten minutes and I will read it through"),
            ("Casey", "2026-03-14T21:05:00Z", "no rush, thank you"),
        ],
    )
    recs = list(parse_all(_teams_cfg(root), "t"))
    assert len(recs) == 1
    rec = recs[0]
    assert [m.role for m in rec.messages] == ["assistant", "user", "assistant"]
    assert [m.speaker for m in rec.messages] == ["Casey", OWNER, "Casey"]
    assert rec.provenance.family == "teams/2026-03"


def test_teams_files_are_separate_threads_and_blank_text_is_skipped(tmp_path):
    root = tmp_path / "teams"
    _teams_file(
        root,
        "casey",
        OWNER,
        [
            ("Casey", "2026-04-01T10:00:00Z", "first"),
            ("Casey", "2026-04-01T10:01:00Z", ""),  # blank (an attachment card) -> skipped
            (OWNER, "2026-04-01T10:02:00Z", "second"),
        ],
    )
    _teams_file(
        root,
        "dana",
        OWNER,
        [("Dana", "2026-04-02T10:00:00Z", "hi"), (OWNER, "2026-04-02T10:01:00Z", "hello")],
    )
    recs = list(parse_all(_teams_cfg(root), "t"))
    threads = {r.provenance.source.detail["conversation_id"] for r in recs}
    assert threads == {"casey", "dana"}
    casey = next(r for r in recs if r.provenance.source.detail["conversation_id"] == "casey")
    assert [m.content for m in casey.messages] == ["first", "second"]


def test_teams_rejects_a_config_role_anchor():
    with pytest.raises(ValueError, match="resolves the owner"):
        SourceConfig(id="teams", parser="teams", path="/x", owner_sender=OWNER)


# --- Reddit comment exchanges ------------------------------------------------


def _reddit_file(root, account, owner, comments):
    """comments: (sent, parent_author, parent_title, parent_text, reply_text)."""
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "account": account,
        "me": {"name": owner, "reddit_username": account},
        "posts": [],
        "comments": [
            {
                "sent": s,
                "subreddit": "books",
                "score": 3,
                "permalink": f"/r/books/{account}/{i}",
                "in_reply_to": {"author": pa, "title": pt, "text": px},
                "text": rt,
            }
            for i, (s, pa, pt, px, rt) in enumerate(comments)
        ],
    }
    (root / f"{account}.json").write_text(json.dumps(payload))
    return root


def _reddit_cfg(path, **overrides):
    base = {
        "sources": [{"id": "reddit", "parser": "reddit", "path": str(path)}],
        # reddit replies are long-form; keep the synthetic floor from dropping them
        "min_record_chars": 0,
        "min_mean_message_chars": 0,
    }
    return LaneDConfig.model_validate(base | overrides)


def test_reddit_pairs_parent_with_owner_reply_owner_as_assistant(tmp_path):
    root = _reddit_file(
        tmp_path / "reddit",
        "sam_reads",
        OWNER,
        [
            (
                "2026-03-14T21:00:00Z",
                "stranger",
                "What got you into doom metal?",
                "Genuinely curious what the entry point was for people.",
                "Funeral doom, weirdly — the slowness forced me to sit still and actually listen.",
            )
        ],
    )
    recs = list(parse_all(_reddit_cfg(root), "t"))
    assert len(recs) == 1
    rec = recs[0]
    # The owner is the ASSISTANT here — the record ends on his reply, the modeled turn.
    assert [m.role for m in rec.messages] == ["user", "assistant"]
    assert rec.messages[0].speaker == "stranger"
    assert rec.messages[1].speaker == OWNER
    assert "entry point" in rec.messages[0].content  # parent title + text combined
    assert rec.provenance.family == "reddit/2026-03"


def test_reddit_skips_deleted_or_half_exchanges(tmp_path):
    root = _reddit_file(
        tmp_path / "reddit",
        "sam_reads",
        OWNER,
        [
            ("2026-03-01T10:00:00Z", "x", "gone", "[deleted]", "[removed]"),  # both gone
            ("2026-03-02T10:00:00Z", "y", "", "", "a reply with no parent to answer"),  # no parent
            ("2026-03-03T10:00:00Z", "z", "Real question", "with a body", "a real answer"),  # keep
        ],
    )
    recs = list(parse_all(_reddit_cfg(root), "t"))
    assert len(recs) == 1
    assert recs[0].messages[1].content == "a real answer"


def test_reddit_max_records_strides_across_accounts(tmp_path):
    root = tmp_path / "reddit"
    for acct in ("sam_a", "sam_b"):
        _reddit_file(
            root,
            acct,
            OWNER,
            [
                (f"2026-0{m}-01T10:00:00Z", "p", f"Q{m}", f"body {m}", f"reply {acct} {m}")
                for m in range(1, 6)
            ],
        )
    # 10 exchanges across two accounts -> capped to 4, strided in time order.
    recs = list(
        parse_all(
            _reddit_cfg(
                root,
                sources=[{"id": "reddit", "parser": "reddit", "path": str(root), "max_records": 4}],
            ),
            "t",
        )
    )
    assert len(recs) == 4


def test_reddit_rejects_a_config_role_anchor():
    with pytest.raises(ValueError, match="resolves the owner"):
        SourceConfig(id="reddit", parser="reddit", path="/x", assistant_sender=OWNER)
