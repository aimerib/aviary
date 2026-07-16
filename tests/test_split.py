from __future__ import annotations

import pytest

from aviary.render.split import HoldoutViolation, assert_no_holdout_leak, split_records
from aviary.schema.records import ConversationRecord, Message, Provenance, SourceRef


def make_record(family: str, holdout: bool, params: dict) -> ConversationRecord:
    return ConversationRecord(
        system="s",
        messages=[
            Message(role="user", content="q"),
            Message(role="assistant", content="a"),
        ],
        provenance=Provenance(
            record_id=f"{family}-{sorted(params.items())}",
            lane="a",
            run_id="t",
            family=family,
            template_id=f"{family}.t1",
            holdout=holdout,
            source=SourceRef(kind="task_instance", detail={"params": params}),
        ),
    )


def test_holdout_families_go_to_eval():
    records = [make_record("web", False, {"i": i}) for i in range(10)]
    records += [make_record("secret", True, {"i": i}) for i in range(3)]
    train, eval_ = split_records(records, eval_param_fraction=0.0, seed=1)
    assert all(not r.provenance.holdout for r in train)
    assert sum(r.provenance.holdout for r in eval_) == 3
    assert_no_holdout_leak(train)


def test_leak_raises():
    leaked = [make_record("secret", True, {"i": 1})]
    with pytest.raises(HoldoutViolation):
        assert_no_holdout_leak(leaked)


def test_eval_param_split_is_deterministic_and_instance_level():
    records = [make_record("web", False, {"i": i}) for i in range(20)]
    t1, e1 = split_records(records, eval_param_fraction=0.25, seed=42)
    t2, e2 = split_records(records, eval_param_fraction=0.25, seed=42)
    assert [r.provenance.record_id for r in e1] == [r.provenance.record_id for r in e2]
    assert len(e1) == 5
    t3, e3 = split_records(records, eval_param_fraction=0.25, seed=43)
    assert {r.provenance.record_id for r in e3} != {r.provenance.record_id for r in e1}


def test_siblings_of_same_instance_split_together():
    records = [make_record("web", False, {"i": i % 4}) for i in range(16)]
    train, eval_ = split_records(records, eval_param_fraction=0.5, seed=7)
    train_keys = {r.provenance.source.detail["params"]["i"] for r in train}
    eval_keys = {r.provenance.source.detail["params"]["i"] for r in eval_}
    assert not train_keys & eval_keys
