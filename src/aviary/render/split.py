"""Family-level train/eval split. Holdout violations are a hard failure, never a warning."""

from __future__ import annotations

import hashlib
import random

from aviary.hashing import canonical_json
from aviary.schema.records import ConversationRecord


class HoldoutViolation(Exception):
    """A holdout family appeared in a train render. Nothing may be written."""


def instance_key(rec: ConversationRecord) -> str:
    payload = canonical_json(
        {"template": rec.provenance.template_id, "detail": rec.provenance.source.detail}
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def split_records(
    records: list[ConversationRecord],
    eval_param_fraction: float,
    seed: int,
) -> tuple[list[ConversationRecord], list[ConversationRecord]]:
    """Holdout families -> eval. Additionally, a deterministic fraction of instance
    parameterizations of non-holdout templates -> eval (unseen params of seen templates)."""
    train: list[ConversationRecord] = []
    eval_: list[ConversationRecord] = []

    instance_keys = sorted({instance_key(r) for r in records if not r.provenance.holdout})
    rng = random.Random(seed)
    n_eval = int(len(instance_keys) * eval_param_fraction)
    eval_instances = set(rng.sample(instance_keys, n_eval)) if n_eval else set()

    for rec in records:
        if rec.provenance.holdout or instance_key(rec) in eval_instances:
            eval_.append(rec)
        else:
            train.append(rec)
    return train, eval_


def assert_no_holdout_leak(train: list[ConversationRecord]) -> None:
    leaked = sorted(
        {r.provenance.family for r in train if r.provenance.holdout}
        | {r.provenance.template_id or "" for r in train if r.provenance.holdout}
    )
    if leaked:
        raise HoldoutViolation(
            f"holdout families/templates in train split: {', '.join(x for x in leaked if x)}"
        )
