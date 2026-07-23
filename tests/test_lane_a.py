from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from aviary.lanes.a_agentic.hermes_config import (
    build_batch_command,
    emit_batch_inputs,
    verify_hermes_interface,
    verify_hermes_python,
)
from aviary.lanes.a_agentic.ingest import IngestContext, IngestError, ingest_hermes_record
from aviary.lanes.a_agentic.run import BurnGuardError, guard_burn
from aviary.lanes.a_agentic.taskbank import expand_all, load_taskbank
from aviary.orchestrate import LaneBand
from aviary.schema.manifest import LaneKeepRate, RunManifest, Teachers

REPO = Path(__file__).resolve().parents[1]


def test_taskbank_loads_and_expands():
    templates = load_taskbank(REPO / "tasks")
    assert any(t.id == "files.save_note" for t in templates)
    instances = expand_all(templates)
    save_note = [i for i in instances if i.template_id == "files.save_note"]
    assert len(save_note) == 12  # zip mode: 12 kind/destination/index triples
    assert all("{" not in i.prompt for i in save_note)
    keys = {i.instance_key() for i in save_note}
    assert len(keys) == 12


def test_emit_inputs_repeats_rollouts(tmp_path):
    instances = expand_all(load_taskbank(REPO / "tasks"))
    out = tmp_path / "inputs.jsonl"
    n = emit_batch_inputs(instances, out)
    assert n == sum(i.n_rollouts for i in instances)
    first = json.loads(out.read_text().splitlines()[0])
    assert set(first) == {"prompt"}


def test_build_batch_command_flags_match_verified_set(tmp_path):
    from aviary.lanes.a_agentic.hermes_config import PASSED_FLAGS

    cmd = build_batch_command(
        dataset_file=tmp_path / "inputs.jsonl",
        run_name="r",
        distribution="terminal_web",
        wire_model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key="k",
        system_prompt="sys",
        num_workers=2,
        batch_size=4,
        max_turns=10,
    )
    assert cmd[:2] == ["python", "batch_runner.py"]
    emitted = {arg[2:].split("=", 1)[0] for arg in cmd[2:]}
    assert emitted == PASSED_FLAGS  # every emitted flag is verified, none forgotten


def _fake_hermes(tmp_path, main_args, distributions, toolsets):
    (tmp_path / "batch_runner.py").write_text(
        f"def main({', '.join(f'{a}=None' for a in main_args)}):\n    pass\n"
    )
    (tmp_path / "toolset_distributions.py").write_text(f"DISTRIBUTIONS = {distributions!r}\n")
    (tmp_path / "toolsets.py").write_text(f"TOOLSETS = {toolsets!r}\n")
    return tmp_path


_MAIN_ARGS = [
    "dataset_file",
    "batch_size",
    "run_name",
    "distribution",
    "model",
    "base_url",
    "api_key",
    "num_workers",
    "max_turns",
    "ephemeral_system_prompt",
    "resume",
]
_DISTS = {"terminal_web": {"toolsets": {"terminal": 100, "file": 100, "web": 80}}}
_TOOLSETS = {
    "terminal": {"tools": ["terminal"], "includes": ["file"]},
    "file": {"tools": ["read_file", "write_file"], "includes": []},
    "web": {"tools": ["web_search", "web_extract"], "includes": []},
}


def test_verify_hermes_interface(tmp_path):
    hd = _fake_hermes(tmp_path, _MAIN_ARGS, _DISTS, _TOOLSETS)
    # file is 100% (directly and via terminal's includes); web is only 80%.
    available = verify_hermes_interface(hd, "terminal_web", ["write_file", "read_file"])
    assert "write_file" in available and "web_search" not in available

    with pytest.raises(ValueError, match="not guaranteed"):
        verify_hermes_interface(hd, "terminal_web", ["web_search"])  # 80% != always
    with pytest.raises(ValueError, match="unknown hermes distribution"):
        verify_hermes_interface(hd, "nope", [])


def test_verify_hermes_interface_rejects_invented_flags(tmp_path):
    hd = _fake_hermes(tmp_path, ["dataset_file", "run_name"], _DISTS, _TOOLSETS)
    with pytest.raises(ValueError, match="does not accept"):
        verify_hermes_interface(hd)


# Mirrors the REAL pinned batch_runner output (verified 2026-07-19): structure
# lives inside text values — <think>, <tool_call> (no ids), <tool_response>
# (authoritative ids), and the harness system turn carrying the <tools> block.
HERMES_RECORD = {
    "prompt_index": 0,
    "completed": True,
    "partial": False,
    "conversations": [
        {
            "from": "system",
            "value": (
                "You are a function calling AI model. ... <tools>\n"
                '[{"name": "write_file", "description": "Write a file.", "parameters": '
                '{"type": "object", "properties": {"path": {"type": "string"}, '
                '"content": {"type": "string"}}, "required": ["path", "content"]}]'
                "\n</tools> ..."
            ),
        },
        {"from": "human", "value": "hey liv, save something"},
        {
            "from": "gpt",
            "value": (
                "<think>\nSimple write. Do it.\n</think>\nOn it.\n"
                '<tool_call>\n{"name": "write_file", "arguments": '
                '{"path": "notes/note.txt", "content": "hi"}}\n</tool_call>'
            ),
        },
        {
            "from": "tool",
            "value": (
                "<tool_response>\n"
                '{"tool_call_id": "call_00_abc", "name": "write_file", '
                '"content": {"bytes_written": 2}}\n'
                "</tool_response>"
            ),
        },
        {"from": "gpt", "value": "<think>\nDone.\n</think>\nSaved."},
    ],
    "toolsets_used": ["file"],
    "tool_stats": {"file": {"count": 1, "success": 1}},
}


def make_ctx() -> IngestContext:
    instances = expand_all(load_taskbank(REPO / "tasks"))
    return IngestContext(
        run_id="t",
        instances=instances,
        system_prompt="You are Olivia.",
        persona_speaker="Olivia",
        teacher_id="deepseek-v4-flash-20260610",
        hermes_commit="v2026.7.7.2",
        prompt_set_hash="h",
    )


def test_ingest_hermes_record():
    ctx = make_ctx()
    # HERMES_RECORD's synthetic trajectory is decoupled from any instance's real
    # prompt text; point prompt_index at whichever instance is files.save_note
    # rather than assuming load order puts it first (family dirs sort
    # alphabetically, and other lane A families now sort earlier).
    idx = next(i for i, inst in enumerate(ctx.instances) if inst.template_id == "files.save_note")
    rec = ingest_hermes_record({**HERMES_RECORD, "prompt_index": idx}, ctx)
    assert rec.provenance.lane == "a"
    assert rec.provenance.template_id == "files.save_note"
    assert rec.provenance.sibling_group
    assert rec.messages[0].role == "user"  # harness system turn replaced, not kept
    gpt = rec.messages[1]
    assert gpt.thought == "Simple write. Do it."
    assert gpt.content == "On it."  # think + tool_call blocks stripped from prose
    assert gpt.tool_calls[0].name == "write_file"
    assert gpt.tool_calls[0].arguments["path"] == "notes/note.txt"
    assert gpt.tool_calls[0].id == "call_00_abc"  # authoritative id from the response
    tool = rec.messages[2]
    assert tool.role == "tool" and tool.tool_call_id == "call_00_abc"
    assert tool.content == '{"bytes_written": 2}'  # payload only, verifier-parseable
    assert rec.system == "You are Olivia."
    # tools schema extracted VERBATIM from the recorded harness turn
    assert rec.tools_schema_json.startswith('[{"name": "write_file"')


def test_ingest_siblings_share_group():
    a = ingest_hermes_record(HERMES_RECORD, make_ctx())
    instances = expand_all(load_taskbank(REPO / "tasks"))
    n = len(instances)  # interleaved order: indices 0 and n are rollouts of instance 0
    b = ingest_hermes_record({**HERMES_RECORD, "prompt_index": n}, make_ctx())
    other = ingest_hermes_record({**HERMES_RECORD, "prompt_index": 1}, make_ctx())
    assert a.provenance.record_id != b.provenance.record_id
    assert a.provenance.sibling_group == b.provenance.sibling_group
    assert other.provenance.sibling_group != a.provenance.sibling_group


def test_ingest_rejects_unknown_role_and_partials():
    bad = {**HERMES_RECORD, "conversations": [{"from": "narrator", "value": "x"}]}
    with pytest.raises(IngestError):
        ingest_hermes_record(bad, make_ctx())
    with pytest.raises(IngestError, match="incomplete"):
        ingest_hermes_record({**HERMES_RECORD, "partial": True}, make_ctx())
    with pytest.raises(IngestError, match="incomplete"):
        ingest_hermes_record({**HERMES_RECORD, "completed": False}, make_ctx())


def test_ingest_rejects_orphan_tool_result():
    # A tool result with no open call must drop the record, not fabricate a pairing.
    bad = {
        **HERMES_RECORD,
        "conversations": [
            {"from": "human", "value": "save it"},
            {
                "from": "tool",
                "value": '<tool_response>\n{"tool_call_id": "x", "name": "write_file", '
                '"content": {}}\n</tool_response>',
            },
        ],
    }
    with pytest.raises(IngestError, match="no open call"):
        ingest_hermes_record(bad, make_ctx())


def test_ingest_pairs_parallel_calls_in_one_tool_turn():
    # Real batch output answers parallel calls with SEVERAL <tool_response> blocks
    # in one tool turn; ids come from the responses, matched by name in order.
    record = {
        **HERMES_RECORD,
        "conversations": [
            {"from": "human", "value": "do both"},
            {
                "from": "gpt",
                "value": (
                    '<tool_call>\n{"name": "write_file", "arguments": {"path": "a"}}\n'
                    "</tool_call>\n"
                    '<tool_call>\n{"name": "read_file", "arguments": {"path": "b"}}\n'
                    "</tool_call>"
                ),
            },
            {
                "from": "tool",
                "value": (
                    "<tool_response>\n"
                    '{"tool_call_id": "call_w", "name": "write_file", "content": {"bytes_written": 1}}\n'
                    "</tool_response>\n<tool_response>\n"
                    '{"tool_call_id": "call_r", "name": "read_file", "content": "text"}\n'
                    "</tool_response>"
                ),
            },
            {"from": "gpt", "value": "Done."},
        ],
    }
    rec = ingest_hermes_record(record, make_ctx())
    calls = {tc.name: tc.id for tc in rec.messages[1].tool_calls}
    assert calls == {"write_file": "call_w", "read_file": "call_r"}
    assert [m.tool_call_id for m in rec.messages if m.role == "tool"] == ["call_w", "call_r"]
    assert rec.messages[3].content == "text"  # string payloads pass through


def _pilot_manifest(**overrides) -> RunManifest:
    base = dict(
        run_id="2026-07-15-pilot",
        kind="pilot",
        finished="2026-07-15T10:00:00Z",
        teachers=Teachers(roster=[], assignments={}),
    )
    m = RunManifest.model_validate({**base, **overrides})
    m.provenance.datagen_config_hash = "hash1"
    # Lane B judges low by design; lane A high. A per-lane guard must accept both.
    m.keep_rates.by_lane = {
        "a": LaneKeepRate(verify=0.6, judge=0.8),
        "b": LaneKeepRate(verify=0.7, judge=0.25),
    }
    return m


# Bands mirroring the shipped burn.yaml shape: lane B's judge band sits far below A's.
_BANDS = {
    "a": LaneBand(verify=(0.30, 0.85), judge=(0.50, 0.95)),
    "b": LaneBand(verify=(0.40, 0.95), judge=(0.15, 0.50)),
}


def test_burn_guard_paths(tmp_path):
    with pytest.raises(BurnGuardError, match="no finished pilot"):
        guard_burn(tmp_path, "hash1", _BANDS, ["a", "b"], 14, "2026-07-16")

    m = _pilot_manifest()
    m.save(tmp_path / "2026-07-15-pilot.manifest.yaml")
    ok = guard_burn(tmp_path, "hash1", _BANDS, ["a", "b"], 14, "2026-07-16")
    assert ok.run_id == "2026-07-15-pilot"

    with pytest.raises(BurnGuardError, match="config changed"):
        guard_burn(tmp_path, "OTHER", _BANDS, ["a", "b"], 14, "2026-07-16")
    with pytest.raises(BurnGuardError, match="old"):
        guard_burn(tmp_path, "hash1", _BANDS, ["a", "b"], 14, "2026-08-16")


def test_burn_guard_lane_b_low_judge_is_in_band(tmp_path):
    # The whole point: lane B's ~25% judge rate is HEALTHY, not a failure. A single
    # global judge floor would have rejected this pilot; the per-lane band accepts it.
    m = _pilot_manifest()
    m.save(tmp_path / "2026-07-15-pilot.manifest.yaml")
    assert guard_burn(tmp_path, "hash1", _BANDS, ["b"], 14, "2026-07-16").run_id


def test_burn_guard_refuses_unpiloted_lane(tmp_path):
    # Burning a lane the pilot never measured (lane A here) is a hard stop.
    m = _pilot_manifest()
    m.keep_rates.by_lane = {"b": LaneKeepRate(verify=0.7, judge=0.25)}
    m.save(tmp_path / "2026-07-15-pilot.manifest.yaml")
    with pytest.raises(BurnGuardError, match="never measured"):
        guard_burn(tmp_path, "hash1", _BANDS, ["a", "b"], 14, "2026-07-16")


def test_burn_guard_out_of_band_rejected(tmp_path):
    m = _pilot_manifest()
    m.keep_rates.by_lane["a"] = LaneKeepRate(verify=0.05, judge=0.8)  # too hard
    m.save(tmp_path / "2026-07-15-pilot.manifest.yaml")
    with pytest.raises(BurnGuardError, match="verify keep rate"):
        guard_burn(tmp_path, "hash1", _BANDS, ["a", "b"], 14, "2026-07-16")


def test_burn_guard_missing_band_rejected(tmp_path):
    m = _pilot_manifest()
    m.save(tmp_path / "2026-07-15-pilot.manifest.yaml")
    with pytest.raises(BurnGuardError, match="no burn_bands entry"):
        guard_burn(tmp_path, "hash1", _BANDS, ["a", "b", "c"], 14, "2026-07-16")


def test_zip_param_mode_and_interleaved_rollout_order():
    from aviary.lanes.a_agentic.taskbank import TaskTemplate, expand, rollout_order

    t = TaskTemplate(
        id="files.t",
        family="files",
        prompt="write {kind} to {destination}",
        param_mode="zip",
        params={"kind": ["a", "b"], "destination": ["x.txt", "y.txt"]},
        tools=["write_file"],
        n_rollouts=2,
        verifier="v.py",
    )
    insts = expand(t)
    assert [(i.params["kind"], i.params["destination"]) for i in insts] == [
        ("a", "x.txt"),
        ("b", "y.txt"),
    ]  # zipped, not cartesian
    order = rollout_order(insts)
    # Interleaved round-robin: same-instance rollouts never adjacent (shared
    # hermes workspace -> same-destination rollouts must not run back-to-back).
    keys = [i.instance_key() for i in order]
    assert keys[0] != keys[1] and keys == [keys[0], keys[1]] * 2

    with pytest.raises(ValueError, match="equal-length"):
        TaskTemplate(
            id="files.bad",
            family="files",
            prompt="{kind} {destination}",
            param_mode="zip",
            params={"kind": ["a"], "destination": ["x", "y"]},
            tools=["write_file"],
            verifier="v.py",
        )


# --- persona neutrality of the task bank -------------------------------------

PERSONA_NAMES = ("olivia", "liv", "sorcha")


def test_no_task_prompt_hardcodes_a_persona_name():
    """Scar tissue: every task prompt opened with "hey liv" — Olivia's NICKNAME —
    so the first Sorcha lane A batch had the user addressing Sorcha as Olivia. A
    grep for "Olivia" found nothing. One task bank serves every target, so the
    address has to come from the target, not the template."""
    import re

    from aviary.lanes.a_agentic.taskbank import load_taskbank

    offenders = []
    for template in load_taskbank(REPO / "tasks"):
        for name in PERSONA_NAMES:
            if re.search(rf"\b{name}\b", template.prompt, re.I):
                offenders.append((template.id, name))
    assert not offenders, f"task prompts naming a persona: {offenders}"


def test_persona_placeholder_is_filled_from_the_target():
    from aviary.lanes.a_agentic.taskbank import expand_all, load_taskbank

    templates = load_taskbank(REPO / "tasks")
    for address in ("liv", "Sorcha"):
        instances = expand_all(templates, address)
        assert instances
        assert any(address in i.prompt for i in instances)
        assert not any("{persona}" in i.prompt for i in instances)


def test_flash_task_prompts_are_byte_identical_to_the_hardcoded_era():
    """flash-v2_2 must reproduce the pre-target pipeline byte-for-byte, so
    substituting its address back in has to reproduce the literal old text."""
    from aviary.lanes.a_agentic.taskbank import expand_all, load_taskbank
    from aviary.targets import Target

    flash = Target.load("flash-v2_2")
    assert flash.address == "liv"
    prompts = [i.prompt for i in expand_all(load_taskbank(REPO / "tasks"), flash.address)]
    assert any(p.startswith("hey liv,") for p in prompts)


def test_every_target_resolves_an_address():
    from aviary.targets import Target

    for name in ("flash-v2_2", "sorcha-v1"):
        assert Target.load(name).address


def test_verify_hermes_python_rejects_an_interpreter_missing_batch_runner_deps(tmp_path):
    # The 2026-07-23 failure: AVIARY_HERMES_PYTHON unset -> hermes ran under an
    # interpreter without its deps and died *inside the subprocess*, 40 minutes in.
    (tmp_path / "batch_runner.py").write_text(
        "import json\nimport definitely_not_a_real_module\n\ndef main():\n    pass\n"
    )
    with pytest.raises(ValueError, match="cannot import"):
        verify_hermes_python(tmp_path, sys.executable)


def test_verify_hermes_python_accepts_an_interpreter_that_has_them(tmp_path):
    (tmp_path / "batch_runner.py").write_text("import json, pathlib\n\ndef main():\n    pass\n")
    verify_hermes_python(tmp_path, sys.executable)  # must not raise


def test_verify_hermes_python_rejects_a_missing_interpreter(tmp_path):
    (tmp_path / "batch_runner.py").write_text("import json\n")
    with pytest.raises(ValueError, match="not runnable"):
        verify_hermes_python(tmp_path, str(tmp_path / "no-such-python"))


def test_verify_hermes_python_ignores_guarded_imports(tmp_path):
    # Only top-level imports are hard requirements; one inside try/except or a
    # function is the checkout's own business, not a reason to refuse to run.
    (tmp_path / "batch_runner.py").write_text(
        "import json\n"
        "try:\n    import definitely_not_a_real_module\nexcept ImportError:\n    pass\n"
        "def main():\n    import another_fake_module\n"
    )
    verify_hermes_python(tmp_path, sys.executable)  # must not raise
