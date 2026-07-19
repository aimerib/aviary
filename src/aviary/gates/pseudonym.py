"""Stable third-party pseudonymization for lane D (stub mechanism, config rules).

The owner's own identifiers are signal and are preserved; third-party names are
replaced with stable pseudonyms so relationships survive ("Alex said..." stays
one consistent person) while identities don't. The name->pseudonym mapping is
persisted under the run dir ($AVIARY_DATA_DIR/<run_id>/scrub/pseudonyms.json) so
re-gating the same run reuses it — pseudonyms are stable per run, and the mapping
file ships nowhere (lane D run data never leaves the run store).

Rules are CONFIG, not code, and personal: they name the owner and their people.
They therefore live in a LOCAL file outside the repo (lane_d.yaml points to it —
same never-committed convention as lane D sources). This module is deliberately a
stub: word-boundary replacement of configured name patterns. Smarter matching
(nicknames, possessives, NER) is the owner's call later; the mapping mechanics
won't change.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from aviary.schema.records import ConversationRecord, Message


class PseudonymRules(BaseModel):
    # Identifiers that must SURVIVE untouched (the owner's names/handles).
    # Checked before name_patterns, so an owner name matched by a broad pattern
    # is still preserved.
    preserve: list[str] = Field(default_factory=list)
    # Regexes matching third-party names to pseudonymize. Word boundaries are
    # added by the stub; patterns should match bare names ("Alex", "A(lex|l)").
    name_patterns: list[str] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None) -> PseudonymRules:
        if path is None or not path.exists():
            return cls()
        return cls.model_validate(yaml.safe_load(path.read_text()) or {})


class Pseudonymizer:
    def __init__(self, rules: PseudonymRules, mapping_path: Path):
        self.rules = rules
        self.mapping_path = mapping_path
        self.mapping: dict[str, str] = {}
        if mapping_path.exists():
            self.mapping = json.loads(mapping_path.read_text())
        self._preserve = {p.lower() for p in rules.preserve}
        self._patterns = [re.compile(rf"\b(?:{p})\b") for p in rules.name_patterns]

    def _pseudonym_for(self, name: str) -> str:
        key = name.lower()
        if key not in self.mapping:
            # First-seen order; persisted immediately so stability survives crashes.
            self.mapping[key] = f"Person-{len(self.mapping) + 1}"
            self.mapping_path.parent.mkdir(parents=True, exist_ok=True)
            self.mapping_path.write_text(json.dumps(self.mapping, indent=1))
        return self.mapping[key]

    def _sub_text(self, text: str) -> str:
        for pattern in self._patterns:
            text = pattern.sub(
                lambda m: m.group(0)
                if m.group(0).lower() in self._preserve
                else self._pseudonym_for(m.group(0)),
                text,
            )
        return text

    def _sub_message(self, m: Message) -> Message:
        updates: dict = {}
        if m.content:
            new = self._sub_text(m.content)
            if new != m.content:
                updates["content"] = new
        if m.thought:
            new = self._sub_text(m.thought)
            if new != m.thought:
                updates["thought"] = new
        if m.speaker and m.speaker.lower() not in self._preserve:
            for pattern in self._patterns:
                if pattern.fullmatch(m.speaker):
                    updates["speaker"] = self._pseudonym_for(m.speaker)
                    break
        return m.model_copy(update=updates) if updates else m

    def apply(self, rec: ConversationRecord) -> ConversationRecord:
        """Pseudonymize conversational fields (content/thought/speaker). Provenance,
        tool payloads, and system text are untouched — lane D records carry none."""
        new_messages = [self._sub_message(m) for m in rec.messages]
        if all(a is b for a, b in zip(new_messages, rec.messages, strict=True)):
            return rec
        return rec.model_copy(update={"messages": new_messages})
