"""PII + teacher-identity scrub. Drop or flag; the only rewriting is lane-policy
pseudonymization (gates/pseudonym.py) — paraphrase stays harmonize's job. Pure
functions over records.

Scrub policy is PER-LANE. The default posture (drop on PII shapes) fits lanes
whose data should contain no real people. Lane D inverts it: the corpus IS one
real person's life, the owner's own identifiers are signal and must survive,
and third parties are pseudonymized instead of dropped. A LaneScrubPolicy says
which pattern labels don't apply to a lane and what transform (if any) runs on
records the scan kept.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from aviary.schema.records import ConversationRecord


@dataclass(frozen=True)
class ScrubPattern:
    label: str
    pattern: re.Pattern[str]
    action: str  # "drop" | "flag"


@dataclass
class LaneScrubPolicy:
    # Pattern labels that do NOT apply to this lane (e.g. lane D exempts the PII
    # shapes that match the owner's own identifiers — they are signal there).
    exempt_labels: frozenset[str] = frozenset()
    # Applied to records the scan kept, before they continue down the funnel
    # (lane D: the pseudonymizer). Records it changes get flag:transformed.
    transform: Callable[[ConversationRecord], ConversationRecord] | None = None


def load_lane_policies(path: Path) -> dict[str, LaneScrubPolicy]:
    """gates/scrub/lane_policy.yaml: per-lane label exemptions. Transforms are
    attached by the orchestrator (they need run-dir state), not by config."""
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    return {
        lane: LaneScrubPolicy(exempt_labels=frozenset(spec.get("exempt_labels", [])))
        for lane, spec in (raw or {}).items()
    }


def load_patterns(*paths: Path) -> list[ScrubPattern]:
    patterns: list[ScrubPattern] = []
    for path in paths:
        raw = yaml.safe_load(path.read_text()) or []
        for entry in raw:
            patterns.append(
                ScrubPattern(
                    label=entry["label"],
                    pattern=re.compile(entry["pattern"], re.I),
                    action=entry.get("action", "drop"),
                )
            )
    return patterns


def scan_record(rec: ConversationRecord, patterns: list[ScrubPattern]) -> list[str]:
    """Returns hit labels, 'drop:'-prefixed for drop-severity hits.

    Scans conversational prose and thoughts — the fields that end up as training
    targets. Tool results are synthetic content from our own sandboxes; they are
    scanned too, but only for teacher self-identification (not PII shapes, which
    legitimately occur in fixtures/tool output like file listings).
    """
    hits: list[str] = []
    for m in rec.messages:
        candidates: list[tuple[str, bool]] = []
        if m.role in ("user", "assistant"):
            candidates.append((m.content, True))
            if m.thought:
                candidates.append((m.thought, True))
        elif m.role == "tool":
            candidates.append((m.content, False))
        for text, full_scan in candidates:
            for p in patterns:
                if not full_scan and not p.label.startswith("teacher_"):
                    continue
                if p.pattern.search(text):
                    hits.append(f"{p.action}:{p.label}")
    return sorted(set(hits))
