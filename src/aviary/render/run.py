"""`just render`: corpus-wide dedupe -> split -> serialize -> write artifacts.

Hard-fails (writes NOTHING) on a holdout violation. Counts land in the manifest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from aviary.gates.dedupe import find_duplicates
from aviary.io.jsonl import read_jsonl, write_jsonl
from aviary.io.store import RunStore
from aviary.render.dpo import pair_siblings
from aviary.render.serializer import (
    ThoughtMode,
    render_conversation,
    render_next_speaker_samples,
)
from aviary.render.split import assert_no_holdout_leak, split_records
from aviary.schema.records import ConversationRecord

log = logging.getLogger(__name__)


@dataclass
class RenderCounts:
    train: int = 0
    eval: int = 0
    nsp: int = 0
    dpo_pairs: int = 0
    deduped: int = 0


def render_run(
    store: RunStore,
    eval_param_fraction: float,
    split_seed: int,
    dpo_min_margin: float,
    extra_corpus: list[ConversationRecord] | None = None,
    include_lanes: set[str] | None = None,
) -> RenderCounts:
    kept = list(read_jsonl(store.gated_kept(), ConversationRecord))
    rejected = list(read_jsonl(store.gated_rejected(), ConversationRecord))

    if include_lanes is not None:
        # Target record-set boundary: a render may only include the lanes its build
        # target declares (e.g. lane D never enters a flash render; nothing voiced
        # by another target's persona enters this one). Exclusions logged, never silent.
        before = len(kept) + len(extra_corpus or [])
        kept = [r for r in kept if r.provenance.lane in include_lanes]
        extra_corpus = [r for r in (extra_corpus or []) if r.provenance.lane in include_lanes]
        rejected = [r for r in rejected if r.provenance.lane in include_lanes]
        excluded = before - len(kept) - len(extra_corpus)
        if excluded:
            log.info(
                "render: %d records outside target lanes %s excluded",
                excluded,
                sorted(include_lanes),
            )

    # corpus-wide dedupe: prior runs' kept records (extra_corpus) come first so
    # new duplicates lose to existing corpus members
    counts = RenderCounts()
    pool = (extra_corpus or []) + kept
    dupes = find_duplicates(pool)
    kept = [r for r in kept if r.provenance.record_id not in dupes]
    counts.deduped = len(pool) - len(extra_corpus or []) - len(kept)

    train, eval_ = split_records(kept, eval_param_fraction, split_seed)
    assert_no_holdout_leak(train)  # raises HoldoutViolation before anything is written

    artifacts: dict[str, list] = {
        "train_with_thoughts": [],
        "train_no_thoughts": [],
        "eval_with_thoughts": [],
        "eval_no_thoughts": [],
        "nsp": [],
    }
    for split_name, records in (("train", train), ("eval", eval_)):
        for rec in records:
            artifacts[f"{split_name}_with_thoughts"].append(
                render_conversation(rec, ThoughtMode.WITH)
            )
            artifacts[f"{split_name}_no_thoughts"].append(
                render_conversation(rec, ThoughtMode.WITHOUT)
            )
            if split_name == "train":
                artifacts["nsp"].extend(render_next_speaker_samples(rec))

    train_ids = {r.provenance.record_id for r in train}
    pairs = pair_siblings(
        [r for r in kept if r.provenance.record_id in train_ids], rejected, dpo_min_margin
    )

    for name, samples in artifacts.items():
        write_jsonl(store.rendered(name), samples)
    write_jsonl(store.rendered("dpo"), pairs)

    counts.train = len(train)
    counts.eval = len(eval_)
    counts.nsp = len(artifacts["nsp"])
    counts.dpo_pairs = len(pairs)
    log.info(
        "render: %d train, %d eval, %d nsp, %d dpo pairs (%d deduped)",
        counts.train,
        counts.eval,
        counts.nsp,
        counts.dpo_pairs,
        counts.deduped,
    )
    return counts
