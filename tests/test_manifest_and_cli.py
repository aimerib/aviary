from __future__ import annotations

from pathlib import Path

import pytest

from aviary.orchestrate import PROMPT_FILES, RunConfig, load_prompt_set
from aviary.schema.manifest import RunManifest, Teachers
from aviary.teacher.roster import Roster

REPO = Path(__file__).resolve().parents[1]


def test_manifest_roundtrip(tmp_path):
    m = RunManifest(
        run_id="2026-07-16-pilot",
        kind="pilot",
        teachers=Teachers(
            roster=[{"id": "glm-5-20260430", "provider": "zhipu", "route": "direct"}],
            assignments={"judge": {"primary": "glm-5-20260430"}},
        ),
    )
    m.counts.rollouts = 42
    m.spend_usd.by_lane["b"] = 1.23
    path = tmp_path / "m.manifest.yaml"
    m.save(path)
    loaded = RunManifest.load(path)
    assert loaded == m


def test_manifest_rejects_undated_teacher():
    with pytest.raises(ValueError, match="dated snapshot"):
        Teachers(
            roster=[{"id": "glm-latest", "provider": "zhipu", "route": "direct"}],
            assignments={},
        )


def test_tracked_roster_config_is_valid():
    roster = Roster.load(REPO / "datagen" / "configs" / "teachers.yaml")
    assert roster.hermes_pin == "v2026.7.7.2"  # release v0.18.2; upstream tags by date
    assert not any(t.provider == "anthropic" for t in roster.teachers)
    # every assignment resolves, and every generator has a cross-vendor judge
    for lane_key, roles in roster.assignments.items():
        for role in roles:
            assert roster.assigned(lane_key, role)
    for lane_key in ("lane_a", "lane_b", "lane_c"):
        for role, tid in roster.assignments[lane_key].items():
            judge = roster.judge_for(tid)
            assert judge.provider != roster.route_for(tid).provider, (lane_key, role)


def test_tracked_run_configs_load():
    for name in ("pilot", "burn"):
        cfg = RunConfig.load(REPO / "datagen" / "configs" / f"{name}.yaml")
        assert cfg.kind == name
        assert set(cfg.lanes) <= {"a", "b", "c"}


def test_prompt_set_covers_all_stages():
    for name, path in PROMPT_FILES.items():
        assert path.exists(), f"missing prompt file for {name}: {path}"
    ps = load_prompt_set()
    assert len(ps.hash) == 64


def test_cli_parses(capsys):
    from aviary.cli import main

    with pytest.raises(SystemExit):
        main([])  # no subcommand


def test_hash_tree_ignores_interpreter_artifacts(tmp_path):
    # Bytecode caches land INSIDE hashed trees when verifier plugins import; their
    # bytes are mtime-dependent, so hashing them makes gate_inputs_hash
    # unreproducible (bit run 2026-07-18-pilot-ao3scale). Only real inputs count.
    from aviary.hashing import hash_tree

    root = tmp_path / "verifiers"
    (root / "files").mkdir(parents=True)
    (root / "files" / "check.py").write_text("def verify(): ...\n")
    before = hash_tree(root)

    (root / "files" / "__pycache__").mkdir()
    (root / "files" / "__pycache__" / "check.cpython-313.pyc").write_bytes(b"\x00cache")
    (root / "files" / ".DS_Store").write_bytes(b"finder junk")
    assert hash_tree(root) == before

    (root / "files" / "check.py").write_text("def verify(): return 1\n")
    assert hash_tree(root) != before  # real input changes still change the hash


def test_schema_lane_d_additive():
    # Lane D additions are strictly additive: old records (no ts) parse unchanged,
    # ts survives a round-trip, and neither dedupe keys nor record ids see it.
    from aviary.gates.dedupe import exact_key
    from aviary.schema.records import (
        ConversationRecord,
        Message,
        Provenance,
        SourceRef,
        make_record_id,
    )

    def rec(ts):
        return ConversationRecord(
            system="s",
            messages=[
                Message(role="user", content="hi", ts=ts),
                Message(role="assistant", content="hey"),
            ],
            provenance=Provenance(
                record_id="r1",
                lane="d",
                run_id="t",
                family="chat/2026-03",
                source=SourceRef(kind="personal_stream", detail={"source": "chat"}),
            ),
        )

    with_ts, without_ts = rec("2026-03-14T21:07:03Z"), rec(None)
    round_tripped = ConversationRecord.model_validate_json(with_ts.model_dump_json())
    assert round_tripped.messages[0].ts == "2026-03-14T21:07:03Z"
    assert exact_key(with_ts) == exact_key(without_ts)  # dedupe never sees ts
    src = SourceRef(kind="personal_stream", detail={"source": "chat"})
    assert make_record_id("d", "t", src) == make_record_id("d", "t", src)
