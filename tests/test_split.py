from __future__ import annotations

import hashlib

import pytest

from aviary.hashing import canonical_json
from aviary.render.split import HoldoutViolation, assert_no_holdout_leak, split_records
from aviary.schema.records import ConversationRecord, Message, Provenance, SourceRef


def _sibling_group(template_id: str, params: dict) -> str:
    # Mirror TaskInstance.instance_key(): stable across an instance's rollouts.
    payload = canonical_json({"template": template_id, "params": params})
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def make_record(family: str, holdout: bool, params: dict, rollout: int = 0) -> ConversationRecord:
    template_id = f"{family}.t1"
    return ConversationRecord(
        system="s",
        messages=[
            Message(role="user", content="q"),
            Message(role="assistant", content="a"),
        ],
        provenance=Provenance(
            record_id=f"{family}-{sorted(params.items())}-{rollout}",
            lane="a",
            run_id="t",
            family=family,
            template_id=template_id,
            holdout=holdout,
            # Real lane-A records: siblings share sibling_group but each rollout
            # carries a distinct prompt_index in source.detail.
            sibling_group=_sibling_group(template_id, params),
            source=SourceRef(
                kind="task_instance", detail={"params": params, "prompt_index": rollout}
            ),
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


def test_mixed_holdout_within_family_raises():
    # One record of a held-out family mislabeled holdout=False.
    records = [
        make_record("secret", True, {"i": 0}),
        make_record("secret", False, {"i": 1}),
    ]
    with pytest.raises(HoldoutViolation, match="inconsistent holdout"):
        split_records(records, eval_param_fraction=0.0, seed=1)


def test_eval_param_split_is_deterministic_and_instance_level():
    records = [make_record("web", False, {"i": i}) for i in range(20)]
    t1, e1 = split_records(records, eval_param_fraction=0.25, seed=42)
    t2, e2 = split_records(records, eval_param_fraction=0.25, seed=42)
    assert [r.provenance.record_id for r in e1] == [r.provenance.record_id for r in e2]
    assert len(e1) == 5
    t3, e3 = split_records(records, eval_param_fraction=0.25, seed=43)
    assert {r.provenance.record_id for r in e3} != {r.provenance.record_id for r in e1}


def test_siblings_of_same_instance_split_together():
    # 4 instances x 4 rollouts each. Each rollout has a distinct prompt_index in
    # source.detail (as real lane-A records do); siblings must not split.
    records = [
        make_record("web", False, {"i": inst}, rollout=r) for inst in range(4) for r in range(4)
    ]
    train, eval_ = split_records(records, eval_param_fraction=0.5, seed=7)
    train_keys = {r.provenance.source.detail["params"]["i"] for r in train}
    eval_keys = {r.provenance.source.detail["params"]["i"] for r in eval_}
    assert not train_keys & eval_keys
    # Every instance's 4 rollouts land wholly on one side of the split.
    eval_ids = {r.provenance.record_id for r in eval_}
    for inst in range(4):
        sides = {
            r.provenance.record_id in eval_ids
            for r in records
            if r.provenance.source.detail["params"]["i"] == inst
        }
        assert len(sides) == 1
