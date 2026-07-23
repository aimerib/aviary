"""Ship guards. Every test here runs under the autouse socket ban, so passing is
itself proof that the refusal paths and --dry-run never touch the network."""

from __future__ import annotations

import pytest

from aviary.schema.manifest import RunManifest, Teachers
from aviary.ship import ShipError, default_repo_name, plan_ship, ship_run

RUN = "2026-07-22-burn"


def _manifest(**artifacts) -> RunManifest:
    m = RunManifest(
        run_id=RUN,
        kind="burn",
        teachers=Teachers(
            roster=[{"id": "glm-5-20260430", "provider": "zhipu", "route": "direct"}],
            assignments={"judge": {"primary": "glm-5-20260430"}},
        ),
    )
    for k, v in artifacts.items():
        setattr(m.artifacts, k, v)
    return m


def _setup(tmp_path, monkeypatch, *, rendered=True, manifest=None):
    """Point the data dir + manifest at tmp_path and lay down a run store."""
    monkeypatch.setenv("AVIARY_DATA_DIR", str(tmp_path / "data"))
    mpath = tmp_path / f"{RUN}.manifest.yaml"
    (manifest or _manifest()).save(mpath)
    monkeypatch.setattr("aviary.ship.manifest_path", lambda run_id: mpath)

    root = tmp_path / "data" / RUN
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "raw" / "lane_a.jsonl").write_text('{"a":1}\n')
    (root / "gated").mkdir(parents=True, exist_ok=True)
    (root / "gated" / "kept.jsonl").write_text('{"k":1}\n')
    if rendered:
        (root / "rendered").mkdir(parents=True, exist_ok=True)
        for name in ("train_with_thoughts", "train_no_thoughts", "eval_with_thoughts"):
            (root / "rendered" / f"{name}.jsonl").write_text('{"text":"x"}\n')
    return root, mpath


def test_default_repo_name():
    assert default_repo_name(RUN) == "aviary-2026-07-22-burn"


def test_plan_ship_collects_the_whole_run_store(tmp_path, monkeypatch):
    root, _ = _setup(tmp_path, monkeypatch)
    got_root, files = plan_ship(RUN)
    assert got_root == root
    names = {p.relative_to(root).as_posix() for p in files}
    assert "rendered/train_with_thoughts.jsonl" in names
    assert "raw/lane_a.jsonl" in names  # full reproducibility chain, not just rendered
    assert "gated/kept.jsonl" in names


def test_ship_refuses_run_with_exempt_lane(tmp_path, monkeypatch):
    # Lane D is radioactive: rendered artifacts commingle lanes in one file, so a
    # run containing an exempt lane cannot be partially shipped — it refuses whole.
    _setup(tmp_path, monkeypatch, manifest=_manifest(ship_exempt_lanes=["d"]))
    with pytest.raises(ShipError, match="ship-exempt"):
        plan_ship(RUN)


def test_ship_refuses_unrendered_run(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, rendered=False)
    with pytest.raises(ShipError, match="no rendered"):
        plan_ship(RUN)


def test_ship_refuses_missing_run_store(tmp_path, monkeypatch):
    monkeypatch.setenv("AVIARY_DATA_DIR", str(tmp_path / "data"))
    mpath = tmp_path / f"{RUN}.manifest.yaml"
    _manifest().save(mpath)
    monkeypatch.setattr("aviary.ship.manifest_path", lambda run_id: mpath)
    with pytest.raises(ShipError, match="no run store"):
        plan_ship(RUN)


def test_ship_requires_token_rather_than_silently_noop(tmp_path, monkeypatch):
    # Scar tissue: a missing key that degrades instead of raising is how a corpus
    # goes quietly missing. Absent HF_TOKEN must be a hard refusal.
    _setup(tmp_path, monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    with pytest.raises(ShipError, match="HF_TOKEN"):
        ship_run(RUN)


def test_dry_run_validates_without_network(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)  # dry run must not even need one
    assert ship_run(RUN, dry_run=True) == "aviary-2026-07-22-burn"
    assert ship_run(RUN, "custom/name", dry_run=True) == "custom/name"


def test_dry_run_still_enforces_exempt_lanes(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, manifest=_manifest(ship_exempt_lanes=["d"]))
    with pytest.raises(ShipError, match="ship-exempt"):
        ship_run(RUN, dry_run=True)
