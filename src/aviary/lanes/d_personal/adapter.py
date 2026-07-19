"""Lane D: personal streams normalized into ConversationRecords.

The owner's private corpus (chat exports, journals, notes) enters the same funnel
as every other lane. Nothing here calls a model — lane D "generation" is pure
normalization; gates do the judging.

Privacy contract (radioactive-data rule): source files live at local paths named
in lane_d.yaml and are NEVER committed — no samples in git, tests, fixtures,
goldens, or docs. Every lane D test runs on synthetic data authored in the test
itself. Run output stays under $AVIARY_DATA_DIR and ships nowhere (the manifest
records the exemption).

Family scheme: `<source>/<YYYY-MM>` of each record's first message. Rationale:
the leakage failure mode for personal logs is temporal adjacency — nearby
messages share context, in-jokes, and ongoing threads — so holdout must be
time-blocked (whole months), never message-sampled. Holdout months are declared
in lane_d.yaml and flow through the existing family-level split rule unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import BaseModel, Field

from aviary.schema.records import (
    ConversationRecord,
    Message,
    Provenance,
    SourceRef,
    make_record_id,
)


class SourceConfig(BaseModel):
    id: str  # source name; becomes the family prefix (`<id>/<YYYY-MM>`)
    parser: str  # key into PARSERS
    path: Path  # local, never committed
    # Sender whose messages become assistant turns (the voice being modeled);
    # every other sender maps to user turns. Speaker names are preserved on both
    # sides — the scrub stage owns pseudonymization, not the adapter.
    assistant_sender: str


class LaneDConfig(BaseModel):
    sources: list[SourceConfig] = Field(default_factory=list)
    # A silence longer than this splits a stream into separate records: beyond a
    # session boundary, adjacent messages stop being one conversation.
    session_gap_minutes: int = 240
    max_messages_per_record: int = 200
    min_messages_per_record: int = 2
    holdout_families: list[str] = Field(default_factory=list)  # e.g. "chat/2026-05"

    @classmethod
    def load(cls, path: Path) -> LaneDConfig:
        return cls.model_validate(yaml.safe_load(path.read_text()) or {})


class SourceParser(Protocol):
    """source config → normalized records. Implementations must preserve
    per-message timestamps (Message.ts) and never fabricate content."""

    def parse(
        self, source: SourceConfig, cfg: LaneDConfig, run_id: str
    ) -> Iterator[ConversationRecord]: ...


def _parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


class GenericJsonlParser:
    """Reference parser for the generic chat-export format (the one format lane D
    defines rather than inherits). One message per line:

        {"conversation_id": "<stream id>",
         "ts": "2026-03-14T21:07:03Z",      # ISO-8601 UTC, non-decreasing per stream
         "sender": "<display name>",
         "text": "<message body>"}

    Messages are grouped by conversation_id in file order, then split into records
    at session gaps (> session_gap_minutes) and at max_messages_per_record.
    Records shorter than min_messages_per_record are dropped (a lone message is
    not a conversation). Real exports (imessage, discord, journal) get their own
    parsers later — registered below, formats not guessed at.
    """

    def parse(
        self, source: SourceConfig, cfg: LaneDConfig, run_id: str
    ) -> Iterator[ConversationRecord]:
        streams: dict[str, list[dict]] = {}
        with source.path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                missing = {"conversation_id", "ts", "sender", "text"} - set(raw)
                if missing:
                    raise ValueError(f"{source.path}:{line_no}: missing fields {sorted(missing)}")
                streams.setdefault(raw["conversation_id"], []).append(raw)

        for conv_id, msgs in streams.items():
            for seg_idx, segment in enumerate(_sessionize(msgs, cfg)):
                if len(segment) < cfg.min_messages_per_record:
                    continue
                yield self._record(source, cfg, run_id, conv_id, seg_idx, segment)

    def _record(
        self,
        source: SourceConfig,
        cfg: LaneDConfig,
        run_id: str,
        conv_id: str,
        seg_idx: int,
        segment: list[dict],
    ) -> ConversationRecord:
        month = _parse_ts(segment[0]["ts"]).strftime("%Y-%m")
        family = f"{source.id}/{month}"
        ref = SourceRef(
            kind="personal_stream",
            detail={"source": source.id, "conversation_id": conv_id, "segment": seg_idx},
        )
        messages = [
            Message(
                role="assistant" if m["sender"] == source.assistant_sender else "user",
                speaker=m["sender"],
                content=m["text"],
                ts=m["ts"],
            )
            for m in segment
        ]
        return ConversationRecord(
            system="",  # persona attachment is a build-target concern, not the adapter's
            messages=messages,
            provenance=Provenance(
                record_id=make_record_id("d", run_id, ref),
                lane="d",
                run_id=run_id,
                family=family,
                holdout=family in cfg.holdout_families,
                source=ref,
            ),
        )


def _sessionize(msgs: list[dict], cfg: LaneDConfig) -> Iterator[list[dict]]:
    segment: list[dict] = []
    for m in msgs:
        if segment:
            gap_min = (_parse_ts(m["ts"]) - _parse_ts(segment[-1]["ts"])).total_seconds() / 60
            if gap_min > cfg.session_gap_minutes or len(segment) >= cfg.max_messages_per_record:
                yield segment
                segment = []
        segment.append(m)
    if segment:
        yield segment


class _StubParser:
    """Registration point for a real-format parser that doesn't exist yet.
    Formats are learned from real exports at implementation time, never guessed."""

    def __init__(self, name: str):
        self.name = name

    def parse(
        self, source: SourceConfig, cfg: LaneDConfig, run_id: str
    ) -> Iterator[ConversationRecord]:
        raise NotImplementedError(
            f"lane D parser {self.name!r} is a registered stub — implement it against a "
            "real export before pointing lane_d.yaml at one"
        )


PARSERS: dict[str, SourceParser] = {
    "generic_jsonl": GenericJsonlParser(),
    "imessage": _StubParser("imessage"),
    "discord": _StubParser("discord"),
    "journal": _StubParser("journal"),
}


def parse_all(cfg: LaneDConfig, run_id: str) -> Iterator[ConversationRecord]:
    for source in cfg.sources:
        if source.parser not in PARSERS:
            raise KeyError(f"unknown lane D parser {source.parser!r} for source {source.id!r}")
        yield from PARSERS[source.parser].parse(source, cfg, run_id)
