"""Coverage for compute_stats/format_stats: per-template pass rates and the
difficulty-band violation surface that `just stats` reports."""

from __future__ import annotations

from aviary.io.jsonl import write_jsonl
from aviary.io.store import RunStore
from aviary.lanes.a_agentic.taskbank import TaskTemplate
from aviary.schema.records import (
    ConversationRecord,
    GateState,
    Message,
    Provenance,
    SourceRef,
)
from aviary.stats import compute_stats, format_stats


def _rec(rid: str, tid: str, *, verified: bool, dropped: bool, reason=None) -> ConversationRecord:
    return ConversationRecord(
        system="s",
        messages=[Message(role="user", content="q"), Message(role="assistant", content="a")],
        provenance=Provenance(
            record_id=rid,
            lane="a",
            run_id="s",
            family="files",
            template_id=tid,
            source=SourceRef(kind="task_instance", detail={}),
        ),
        gate_state=GateState(verified=verified, dropped=dropped, drop_reason=reason),
    )


def _template(target: tuple[float, float]) -> TaskTemplate:
    return TaskTemplate(
        id="files.save_note",
        family="files",
        prompt="hi",
        tools=["write_file"],
        verifier="verifiers/files/save_note.py",
        difficulty_target=target,
    )


def _write(store: RunStore) -> None:
    write_jsonl(store.gated_kept(), [_rec("k1", "files.save_note", verified=True, dropped=False)])
    write_jsonl(
        store.gated_rejected(),
        [
            _rec("r1", "files.save_note", verified=False, dropped=True, reason="verify"),
            _rec("r2", "files.save_note", verified=True, dropped=True, reason="judge"),
        ],
    )


def test_compute_stats_counts_and_no_violation(tmp_path):
    store = RunStore("s", root=tmp_path)
    _write(store)
    stats = compute_stats(store, [_template((0.3, 0.8))])
    t = stats.per_template["files.save_note"]
    assert (t.rollouts, t.verified, t.kept) == (3, 2, 1)
    assert abs(t.pass_rate - 2 / 3) < 1e-9
    assert not stats.band_violations  # 0.67 within [0.3, 0.8]
    out = format_stats(stats, None)
    assert "files.save_note" in out and "0.67" in out


def test_compute_stats_flags_band_violation(tmp_path):
    store = RunStore("s2", root=tmp_path)
    _write(store)
    stats = compute_stats(store, [_template((0.3, 0.5))])  # 0.67 above the band
    assert any("files.save_note" in v for v in stats.band_violations)
