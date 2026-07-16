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
    assert roster.hermes_pin == "v0.18.2"
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
