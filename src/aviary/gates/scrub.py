"""PII + teacher-identity scrub. Drop or flag, never rewrite (rewriting is harmonize's
job and it may not touch protected text either). Pure functions over records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from aviary.schema.records import ConversationRecord


@dataclass(frozen=True)
class ScrubPattern:
    label: str
    pattern: re.Pattern[str]
    action: str  # "drop" | "flag"


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
