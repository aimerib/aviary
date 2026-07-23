"""Ship guards. Every test here runs under the autouse socket ban, so passing is
itself proof that the refusal paths and --dry-run never touch the network."""

from __future__ import annotations

import json

import pytest
import yaml

from aviary.schema.manifest import RunManifest, Teachers
from aviary.ship import (
    ShipError,
    _dataset_card,
    default_external_repo_name,
    default_repo_name,
    plan_ship,
    plan_ship_external,
    run_dataset_card,
    ship_external,
    ship_run,
)

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


def test_run_card_declares_configs_for_every_rendered_artifact():
    # The configs block is load-bearing: a run store has thousands of files and
    # the Hub cannot guess which are the corpus. Missing it => load_dataset fails.
    m = _manifest()
    m.counts.rendered_train, m.counts.rendered_eval = 1629, 514
    card = run_dataset_card(m)
    head = card.split("---")[1]
    for path in (
        "rendered/train_with_thoughts.jsonl",
        "rendered/train_no_thoughts.jsonl",
        "rendered/eval_with_thoughts.jsonl",
        "rendered/eval_no_thoughts.jsonl",
        "rendered/nsp.jsonl",
        "rendered/dpo.jsonl",
    ):
        assert path in head, path
    assert "default: true" in head  # one config must be the default


def test_run_card_yaml_frontmatter_parses():
    card = run_dataset_card(_manifest())
    assert card.startswith("---\n")
    front = yaml.safe_load(card.split("---")[1])
    names = [c["config_name"] for c in front["configs"]]
    assert names == ["with_thoughts", "no_thoughts", "nsp", "dpo"]
    with_thoughts = front["configs"][0]
    assert {d["split"] for d in with_thoughts["data_files"]} == {"train", "test"}


def test_run_card_warns_about_loss_masking():
    # Training on full `text` without masking to train_spans teaches the model to
    # generate user turns and system prompts; the card must say so.
    card = run_dataset_card(_manifest())
    assert "train_spans" in card
    assert "masking" in card.lower()


# --- cleaned external corpora ------------------------------------------------


def _external(tmp_path, monkeypatch, *, report=True, data=True):
    monkeypatch.setenv("AVIARY_DATA_DIR", str(tmp_path / "data"))
    root = tmp_path / "data" / "external" / "rp-reasoning-v2"
    root.mkdir(parents=True, exist_ok=True)
    if data:
        (root / "cleaned.jsonl").write_text('{"id":"a","conversations":[]}\n')
    if report:
        (root / "report.json").write_text(
            json.dumps(
                {
                    "source": "aimeri/rp-reasoning-v2",
                    "input_rows": 100,
                    "dropped": {"minor_coded": 11},
                    "survived_filters": 89,
                    "written": 89,
                    "seed": 1,
                    "drop_claude": True,
                }
            )
        )
    return root


def test_external_repo_name_is_namespaced_away_from_runs():
    # A foreign corpus must never collide with, or read as, an aviary run repo:
    # the same name used for both must resolve to two different repos.
    assert default_external_repo_name("rp-reasoning-v2") == "aviary-external-rp-reasoning-v2"
    for name in ("rp-reasoning-v2", "2026-07-22-burn", "x"):
        assert default_external_repo_name(name) != default_repo_name(name)
    assert "external" in default_external_repo_name("x")


def test_external_plan_returns_files_and_report(tmp_path, monkeypatch):
    root = _external(tmp_path, monkeypatch)
    got_root, files, report = plan_ship_external("rp-reasoning-v2")
    assert got_root == root
    assert {p.name for p in files} == {"cleaned.jsonl", "report.json"}
    assert report["source"] == "aimeri/rp-reasoning-v2"


def test_external_refuses_without_report(tmp_path, monkeypatch):
    # Cleaned data whose filters cannot be reconstructed invites unexamined reuse.
    _external(tmp_path, monkeypatch, report=False)
    with pytest.raises(ShipError, match="report.json"):
        plan_ship_external("rp-reasoning-v2")


def test_external_refuses_without_data(tmp_path, monkeypatch):
    _external(tmp_path, monkeypatch, data=False)
    with pytest.raises(ShipError, match="cleaned.jsonl"):
        plan_ship_external("rp-reasoning-v2")


def test_external_refuses_missing_corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("AVIARY_DATA_DIR", str(tmp_path / "data"))
    with pytest.raises(ShipError, match="no cleaned corpus"):
        plan_ship_external("nope")


def test_external_dry_run_offline(tmp_path, monkeypatch):
    _external(tmp_path, monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert ship_external("rp-reasoning-v2", dry_run=True) == "aviary-external-rp-reasoning-v2"


def test_external_requires_token(tmp_path, monkeypatch):
    _external(tmp_path, monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    with pytest.raises(ShipError, match="HF_TOKEN"):
        ship_external("rp-reasoning-v2")


def test_dataset_card_records_filters_and_caveat(tmp_path, monkeypatch):
    _external(tmp_path, monkeypatch)
    _, _, report = plan_ship_external("rp-reasoning-v2")
    card = _dataset_card("rp-reasoning-v2", report)
    assert "aimeri/rp-reasoning-v2" in card  # provenance
    assert "minor_coded" in card and "11" in card  # what was removed
    assert "false negatives" in card  # the honest caveat travels with the data
    assert "drop_claude" in card  # which optional filters ran
