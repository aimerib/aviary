"""`just gate`: verify -> judge -> scrub -> dedupe -> harmonize over a run's raw records.

Rejected records are written to gated/rejected.jsonl WITH their gate state — the
judge-rejected ones are the DPO source, the rest feed difficulty tuning stats.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from aviary.gates import dedupe as dedupe_mod
from aviary.gates.harmonize import harmonize_record, is_olivia_voiced
from aviary.gates.judge import Rubric, judge_record
from aviary.gates.scrub import ScrubPattern, scan_record
from aviary.gates.verify import run_verifier, verifier_id
from aviary.io.jsonl import read_jsonl, write_jsonl
from aviary.io.store import RunStore
from aviary.paths import REPO_ROOT
from aviary.schema.records import ConversationRecord
from aviary.teacher.client import TeacherClient
from aviary.teacher.prompts import PromptSet
from aviary.teacher.roster import Roster

log = logging.getLogger(__name__)

# which recorded teacher role carries the record's "voice" (for cross-vendor judging)
GENERATOR_ROLE_PRECEDENCE = ("character", "dialogue", "assistant")

VerifierResolver = Callable[[ConversationRecord], list[Path]]


def default_resolver(template_verifiers: dict[str, str]) -> VerifierResolver:
    """lane a: the template's declared verifier; lanes b/c: every structural verifier
    in verifiers/laneb|lanec/."""

    def resolve(rec: ConversationRecord) -> list[Path]:
        if rec.provenance.lane == "a":
            tid = rec.provenance.template_id or ""
            if tid not in template_verifiers:
                raise KeyError(f"no verifier declared for template {tid!r}")
            return [REPO_ROOT / template_verifiers[tid]]
        lane_dir = REPO_ROOT / "verifiers" / f"lane{rec.provenance.lane}"
        return sorted(lane_dir.glob("*.py"))

    return resolve


@dataclass
class GateStats:
    total: int = 0
    verified: int = 0
    judged: int = 0
    kept: int = 0
    drops: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        self.drops[reason] = self.drops.get(reason, 0) + 1

    @property
    def verify_rate(self) -> float:
        return self.verified / self.total if self.total else 0.0

    @property
    def judge_rate(self) -> float:
        return self.judged / self.verified if self.verified else 0.0


def _generator_id(rec: ConversationRecord) -> str:
    for role in GENERATOR_ROLE_PRECEDENCE:
        if role in rec.provenance.teachers:
            return rec.provenance.teachers[role]
    return next(iter(rec.provenance.teachers.values()))


def _rubric_for(rec: ConversationRecord, rubrics: dict[str, Rubric]) -> Rubric:
    """Pick the rubric by VOICE, not just lane. Lane C is mixed: Olivia simple-chats
    use the Olivia quality rubric, but character-RP records must be judged on
    character fidelity — scoring them on olivia_voice is meaningless (Olivia is
    transparent in roleplay). Same voice signal the harmonizer keys off."""
    if rec.provenance.lane == "c" and not is_olivia_voiced(rec):
        return rubrics["c_character"]
    return rubrics[rec.provenance.lane]


def run_gates(
    store: RunStore,
    resolver: VerifierResolver,
    rubric_by_lane: dict[str, Rubric],
    scrub_patterns: list[ScrubPattern],
    roster: Roster,
    client: TeacherClient,
    prompts: PromptSet,
    dedupe_threshold: float = 0.8,
) -> GateStats:
    records: list[ConversationRecord] = []
    for path in store.raw_files():
        records.extend(read_jsonl(path, ConversationRecord))

    stats = GateStats(total=len(records))
    kept: list[ConversationRecord] = []
    rejected: list[ConversationRecord] = []

    def reject(rec: ConversationRecord, reason: str, **gate_updates) -> None:
        stats.drop(reason)
        rejected.append(
            rec.model_copy(
                update={
                    "gate_state": rec.gate_state.model_copy(
                        update={"dropped": True, "drop_reason": reason, **gate_updates}
                    )
                }
            )
        )

    # 1. verify (outcome only). A verifier import/resolve failure drops THAT record,
    # never the whole run (paid pipeline: one bad plugin must not lose all progress).
    verified: list[ConversationRecord] = []
    for rec in records:
        try:
            paths = resolver(rec)  # resolve once: recorded vid == verifiers that ran
            results = [run_verifier(p, rec) for p in paths]
        except Exception as e:
            log.warning("verify errored for %s: %s", rec.provenance.record_id, e)
            reject(rec, "error")
            continue
        vid = ",".join(verifier_id(p, REPO_ROOT) for p in paths)
        passed = bool(results) and all(r.passed for r in results)
        details = {r.verifier_id: r.details for r in results}
        rec = rec.model_copy(
            update={
                "gate_state": rec.gate_state.model_copy(
                    update={"verified": passed, "verifier_id": vid, "verifier_details": details}
                )
            }
        )
        if passed:
            stats.verified += 1
            verified.append(rec)
        else:
            reject(rec, "verify")

    # 2. judge (cross-vendor); rejects retained for DPO
    judged: list[ConversationRecord] = []
    for rec in verified:
        try:
            rubric = _rubric_for(rec, rubric_by_lane)
            judge_model = roster.judge_for(_generator_id(rec)).id
            scores = judge_record(rec, rubric, client, judge_model, prompts)
        except Exception as e:
            # A malformed judge reply / transient LLM error drops one record.
            log.warning("judge errored for %s: %s", rec.provenance.record_id, e)
            reject(rec, "error")
            continue
        rec = rec.model_copy(
            update={"gate_state": rec.gate_state.model_copy(update={"judge": scores})}
        )
        if scores.passed:
            stats.judged += 1
            judged.append(rec)
        else:
            reject(rec, "judge")

    # 3. scrub
    clean: list[ConversationRecord] = []
    for rec in judged:
        hits = scan_record(rec, scrub_patterns)
        flags = [h for h in hits if h.startswith("flag:")]
        rec = rec.model_copy(
            update={"gate_state": rec.gate_state.model_copy(update={"scrub_flags": hits})}
        )
        if any(h.startswith("drop:") for h in hits):
            reject(rec, "scrub")
        else:
            if flags:
                log.info("record %s flagged: %s", rec.provenance.record_id, flags)
            clean.append(rec)

    # 4. dedupe (within run; corpus-wide pass happens again at render)
    dupes = dedupe_mod.find_duplicates(clean, threshold=dedupe_threshold)
    unique = []
    for rec in clean:
        dup_of = dupes.get(rec.provenance.record_id)
        if dup_of:
            reject(rec, "dedupe", deduped_against=dup_of)
        else:
            unique.append(rec)

    # 5. harmonize (span-protected; lane policy)
    harmonizer_model = roster.assigned("harmonizer", "primary").id
    for rec in unique:
        try:
            outcome = harmonize_record(rec, client, prompts, harmonizer_model)
        except Exception as e:
            # A transient LLM error in harmonize drops one record, not the run.
            log.warning("harmonize errored for %s: %s", rec.provenance.record_id, e)
            reject(rec, "error")
            continue
        if outcome.dropped:
            stats.drop("span_violation")
            rejected.append(outcome.record)
        else:
            kept.append(outcome.record)

    stats.kept = len(kept)
    write_jsonl(store.gated_kept(), kept)
    write_jsonl(store.gated_rejected(), rejected)
    log.info(
        "gates: %d in, %d kept (verify %.0f%%, judge %.0f%%), drops=%s",
        stats.total,
        stats.kept,
        stats.verify_rate * 100,
        stats.judge_rate * 100,
        stats.drops,
    )
    return stats
