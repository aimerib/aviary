from __future__ import annotations

import json
from pathlib import Path

import pytest

from aviary.lanes.a_agentic.hermes_config import emit_batch_config, emit_batch_inputs
from aviary.lanes.a_agentic.ingest import IngestContext, IngestError, ingest_hermes_record
from aviary.lanes.a_agentic.run import BurnBands, BurnGuardError, guard_burn
from aviary.lanes.a_agentic.taskbank import expand_all, load_taskbank
from aviary.schema.manifest import RunManifest, Teachers

REPO = Path(__file__).resolve().parents[1]


def test_taskbank_loads_and_expands():
    templates = load_taskbank(REPO / "tasks")
    assert any(t.id == "files.save_note" for t in templates)
    instances = expand_all(templates)
    save_note = [i for i in instances if i.template_id == "files.save_note"]
    assert len(save_note) == 6  # 3 kinds x 2 destinations
    assert all("{" not in i.prompt for i in save_note)
    keys = {i.instance_key() for i in save_note}
    assert len(keys) == 6


def test_emit_inputs_repeats_rollouts(tmp_path):
    instances = expand_all(load_taskbank(REPO / "tasks"))
    out = tmp_path / "inputs.jsonl"
    n = emit_batch_inputs(instances, out)
    assert n == sum(i.n_rollouts for i in instances)
    first = json.loads(out.read_text().splitlines()[0])
    assert set(first) == {"prompt"}


def test_emit_config_rejects_unknown_keys(tmp_path):
    path = emit_batch_config(
        toolsets=["file"],
        wire_model="deepseek-v4-flash-20260610",
        system_prompt="sys",
        output_dir=tmp_path / "out",
        num_workers=2,
        batch_size=4,
        out_path=tmp_path / "cfg.yaml",
    )
    assert path.exists()


HERMES_RECORD = {
    "prompt_index": 0,
    "conversations": [
        {"from": "human", "value": "hey liv, save something"},
        {
            "from": "gpt",
            "value": "On it.",
            "tool_calls": [
                {
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path": "notes/note.txt", "content": "hi"}',
                    }
                }
            ],
        },
        {"from": "tool", "value": '{"ok": true}'},
        {"from": "gpt", "value": "Saved."},
    ],
    "toolsets_used": ["file"],
    "tool_stats": {"file": {"count": 1, "success": 1}},
}


def make_ctx() -> IngestContext:
    from aviary.lanes.a_agentic.toolsets import load_toolset_schemas

    instances = expand_all(load_taskbank(REPO / "tasks"))
    return IngestContext(
        run_id="t",
        instances=instances,
        system_prompt="You are Olivia.",
        tools_schema_by_family=load_toolset_schemas(REPO / "datagen" / "toolsets"),
        teacher_id="deepseek-v4-flash-20260610",
        hermes_commit="v0.18.2",
        prompt_set_hash="h",
    )


def test_ingest_hermes_record():
    rec = ingest_hermes_record(HERMES_RECORD, make_ctx())
    assert rec.provenance.lane == "a"
    assert rec.provenance.template_id == "files.save_note"
    assert rec.provenance.sibling_group
    assert rec.messages[1].tool_calls[0].name == "write_file"
    assert rec.messages[1].tool_calls[0].arguments["path"] == "notes/note.txt"
    assert rec.messages[2].role == "tool"
    assert rec.messages[2].tool_call_id == rec.messages[1].tool_calls[0].id
    assert rec.system == "You are Olivia."
    assert rec.tools_schema_json and "write_file" in rec.tools_schema_json


def test_ingest_siblings_share_group():
    a = ingest_hermes_record(HERMES_RECORD, make_ctx())
    b = ingest_hermes_record({**HERMES_RECORD, "prompt_index": 1}, make_ctx())
    other = ingest_hermes_record({**HERMES_RECORD, "prompt_index": 4}, make_ctx())
    assert a.provenance.record_id != b.provenance.record_id
    assert a.provenance.sibling_group == b.provenance.sibling_group  # rollouts 0,1 of instance 0
    assert other.provenance.sibling_group != a.provenance.sibling_group


def test_ingest_rejects_unknown_role():
    bad = {**HERMES_RECORD, "conversations": [{"from": "narrator", "value": "x"}]}
    with pytest.raises(IngestError):
        ingest_hermes_record(bad, make_ctx())


def test_ingest_rejects_orphan_tool_result():
    # A tool result with no preceding tool call must drop the record, not
    # fabricate a call id.
    bad = {
        **HERMES_RECORD,
        "conversations": [
            {"from": "human", "value": "save it"},
            {"from": "tool", "value": '{"ok": true}'},
        ],
    }
    with pytest.raises(IngestError, match="no preceding tool call"):
        ingest_hermes_record(bad, make_ctx())


def test_ingest_matches_tool_result_by_explicit_id():
    # Out-of-order results pair by id, not arrival order.
    record = {
        **HERMES_RECORD,
        "conversations": [
            {"from": "human", "value": "do both"},
            {
                "from": "gpt",
                "value": "On it.",
                "tool_calls": [
                    {"id": "call_a", "function": {"name": "write_file", "arguments": "{}"}},
                    {"id": "call_b", "function": {"name": "write_file", "arguments": "{}"}},
                ],
            },
            {"from": "tool", "tool_call_id": "call_b", "value": '{"ok": true}'},
            {"from": "tool", "tool_call_id": "call_a", "value": '{"ok": true}'},
            {"from": "gpt", "value": "Done."},
        ],
    }
    rec = ingest_hermes_record(record, make_ctx())
    assert rec.messages[2].tool_call_id == "call_b"
    assert rec.messages[3].tool_call_id == "call_a"


def _pilot_manifest(**overrides) -> RunManifest:
    base = dict(
        run_id="2026-07-15-pilot",
        kind="pilot",
        finished="2026-07-15T10:00:00Z",
        teachers=Teachers(roster=[], assignments={}),
    )
    m = RunManifest.model_validate({**base, **overrides})
    m.provenance.datagen_config_hash = "hash1"
    m.keep_rates.verify = 0.4
    m.keep_rates.judge = 0.8
    return m


def test_burn_guard_paths(tmp_path):
    bands = BurnBands(max_age_days=14)
    with pytest.raises(BurnGuardError, match="no finished pilot"):
        guard_burn(tmp_path, "hash1", bands, "2026-07-16")

    m = _pilot_manifest()
    m.save(tmp_path / "2026-07-15-pilot.manifest.yaml")
    assert guard_burn(tmp_path, "hash1", bands, "2026-07-16").run_id == "2026-07-15-pilot"

    with pytest.raises(BurnGuardError, match="config changed"):
        guard_burn(tmp_path, "OTHER", bands, "2026-07-16")
    with pytest.raises(BurnGuardError, match="old"):
        guard_burn(tmp_path, "hash1", bands, "2026-08-16")

    m.keep_rates.verify = 0.05  # too hard: outside band
    m.save(tmp_path / "2026-07-15-pilot.manifest.yaml")
    with pytest.raises(BurnGuardError, match="verify keep rate"):
        guard_burn(tmp_path, "hash1", bands, "2026-07-16")
