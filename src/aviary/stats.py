"""`just stats`: keep rates, pass@N per template, spend, difficulty-band violations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from aviary.io.jsonl import read_jsonl
from aviary.io.store import RunStore
from aviary.lanes.a_agentic.taskbank import TaskTemplate
from aviary.schema.manifest import RunManifest
from aviary.schema.records import ConversationRecord


@dataclass
class TemplateStats:
    rollouts: int = 0
    verified: int = 0
    kept: int = 0

    @property
    def pass_rate(self) -> float:
        return self.verified / self.rollouts if self.rollouts else 0.0


@dataclass
class RunStats:
    per_template: dict[str, TemplateStats] = field(default_factory=dict)
    band_violations: list[str] = field(default_factory=list)


def compute_stats(store: RunStore, templates: list[TaskTemplate]) -> RunStats:
    stats = RunStats(per_template=defaultdict(TemplateStats))
    seen: dict[str, ConversationRecord] = {}
    for path in [store.gated_kept(), store.gated_rejected()]:
        if not path.exists():
            continue
        for rec in read_jsonl(path, ConversationRecord):
            seen[rec.provenance.record_id] = rec
    for rec in seen.values():
        key = rec.provenance.template_id or f"lane_{rec.provenance.lane}:{rec.provenance.family}"
        t = stats.per_template[key]
        t.rollouts += 1
        if rec.gate_state.verified:
            t.verified += 1
        if not rec.gate_state.dropped:
            t.kept += 1

    by_id = {t.id: t for t in templates}
    for key, t in stats.per_template.items():
        template = by_id.get(key)
        if template is None:
            continue
        low, high = template.difficulty_target
        if t.rollouts and not (low <= t.pass_rate <= high):
            stats.band_violations.append(
                f"{key}: pass rate {t.pass_rate:.2f} outside [{low}, {high}] — "
                "revise the TEMPLATE (difficulty), never the verifier"
            )
    return stats


def format_stats(stats: RunStats, manifest: RunManifest | None) -> str:
    lines = ["template                                rollouts  verified  kept  pass"]
    for key in sorted(stats.per_template):
        t = stats.per_template[key]
        lines.append(f"{key:<40}{t.rollouts:>8}{t.verified:>10}{t.kept:>6}  {t.pass_rate:.2f}")
    if manifest:
        lines.append("")
        lines.append(
            f"keep rates: verify={manifest.keep_rates.verify:.2f} "
            f"judge={manifest.keep_rates.judge:.2f}  spend=${manifest.spend_usd.total:.2f}"
        )
    if stats.band_violations:
        lines.append("\nDIFFICULTY BAND VIOLATIONS:")
        lines.extend(f"  - {v}" for v in stats.band_violations)
    return "\n".join(lines)
