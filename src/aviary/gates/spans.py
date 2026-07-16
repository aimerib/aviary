"""Protected-span model for the harmonizer (span-protection rule).

Structural protection happens upstream (tool args/results/system/schemas are never
paraphrase candidates). This module protects sub-spans INSIDE conversational prose:
code, URLs, paths, quoted strings, numbers, identifiers. The paraphraser only ever
sees masked text; restore must be byte-perfect or the record is dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Order = priority; earlier kinds win overlaps.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("fenced_code", re.compile(r"```.*?```", re.S)),
    ("inline_code", re.compile(r"`[^`\n]+`")),
    ("url", re.compile(r"\bhttps?://[^\s)\]}>\"']+")),
    ("abs_path", re.compile(r"(?<![\w.])(?:~?/[\w.\-]+){2,}/?")),
    ("quoted", re.compile(r"\"[^\"\n]*\"|'[^'\n]*'(?![a-z])")),
    ("identifier", re.compile(r"\b[a-zA-Z_][\w]*(?:_[\w]+)+\b|\b[a-z]+(?:[A-Z][a-z0-9]+)+\b")),
    ("number", re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?%?(?![\w.])", re.I)),
]

_PLACEHOLDER = "⟦S{i}⟧"  # ⟦S0⟧
_PLACEHOLDER_RE = re.compile(r"⟦S(\d+)⟧")


@dataclass(frozen=True)
class ProtectedSpan:
    start: int
    end: int
    kind: str
    text: str


def extract_protected_spans(text: str) -> list[ProtectedSpan]:
    taken: list[tuple[int, int]] = []
    spans: list[ProtectedSpan] = []
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            if any(m.start() < e and m.end() > s for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            spans.append(ProtectedSpan(m.start(), m.end(), kind, m.group()))
    return sorted(spans, key=lambda s: s.start)


def mask_spans(text: str, spans: list[ProtectedSpan]) -> tuple[str, dict[str, str]]:
    mapping: dict[str, str] = {}
    out: list[str] = []
    pos = 0
    for i, span in enumerate(spans):
        placeholder = _PLACEHOLDER.format(i=i)
        mapping[placeholder] = span.text
        out.append(text[pos : span.start])
        out.append(placeholder)
        pos = span.end
    out.append(text[pos:])
    return "".join(out), mapping


def restore_spans(paraphrased: str, mapping: dict[str, str]) -> str | None:
    """Byte-perfect restore, or None. Every placeholder must appear exactly once and
    no unknown/leftover placeholder may remain (never bend the boundary)."""
    found = _PLACEHOLDER_RE.findall(paraphrased)
    expected = sorted(_PLACEHOLDER_RE.findall(" ".join(mapping)))
    if sorted(found) != expected:
        return None
    result = paraphrased
    for placeholder, original in mapping.items():
        result = result.replace(placeholder, original)
    if _PLACEHOLDER_RE.search(result) or "⟦" in result:
        return None
    return result
