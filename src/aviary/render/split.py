"""Family-level train/eval split. Holdout violations are a hard failure, never a warning."""

from __future__ import annotations

import hashlib
import random

from aviary.hashing import canonical_json
from aviary.schema.records import ConversationRecord


class HoldoutViolation(Exception):
    """A holdout family appeared in a train render. Nothing may be written."""


def instance_key(rec: ConversationRecord) -> str:
    """The split unit: MUST be byte-identical across all sibling rollouts of one
    instance, or siblings leak across the train/eval boundary (some rollouts land
    in eval, their twins in train, exposing eval prompts during SFT).

    When a lane sets `sibling_group` (lane A: hash of template+params), that value
    IS the split unit — it is stable across rollouts, whereas `source.detail`
    carries per-rollout fields (prompt_index, toolsets_used, tool_stats) that
    would otherwise give each rollout its own key. Lanes without rollout siblings
    (B/C, sibling_group=None) fall back to the full source ref, one key per record.
    """
    if rec.provenance.sibling_group:
        return rec.provenance.sibling_group
    payload = canonical_json(
        {"template": rec.provenance.template_id, "detail": rec.provenance.source.detail}
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def assert_family_holdout_consistent(records: list[ConversationRecord]) -> None:
    """Holdout is family-level (split rule). Every record of a family must carry the
    same holdout flag; a family split across True/False means a record was mislabeled
    (e.g. a lane-C record that inherited a held-out lane-B work but lost the flag),
    which the per-record guard alone would let slip into train."""
    by_family: dict[str, set[bool]] = {}
    for r in records:
        by_family.setdefault(r.provenance.family, set()).add(r.provenance.holdout)
    mixed = sorted(f for f, flags in by_family.items() if len(flags) > 1)
    if mixed:
        raise HoldoutViolation(
            f"families with inconsistent holdout flags (mislabeled records): {', '.join(mixed)}"
        )


def split_records(
    records: list[ConversationRecord],
    eval_param_fraction: float,
    seed: int,
) -> tuple[list[ConversationRecord], list[ConversationRecord]]:
    """Holdout families -> eval. Additionally, a deterministic fraction of instance
    parameterizations of non-holdout templates -> eval (unseen params of seen templates)."""
    assert_family_holdout_consistent(records)
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
