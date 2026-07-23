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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

import yaml
from pydantic import BaseModel, Field, model_validator

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
    path: Path  # local, never committed; a file or (imessage) a directory of threads

    # Exactly one of these decides the role mapping. Speaker names are preserved on
    # both sides — the scrub stage owns pseudonymization, not the adapter.
    #
    # `assistant_sender`: that sender becomes assistant turns (the voice being
    #   modeled), everyone else user turns.
    # `owner_sender`: that sender becomes USER turns, everyone else assistant.
    #   This is the mapping a companion build wants: the owner sits where the owner
    #   actually sits at inference, and the model is never trained to produce the
    #   owner's own turns — SOUL.md's hard line ("never impersonates the user or
    #   writes their turns") is a training-data property before it is a rubric axis.
    assistant_sender: str = ""
    owner_sender: str = ""

    # imessage only: which threads to read from a directory export.
    include_threads: list[str] = Field(default_factory=list)  # file stems; empty = all
    exclude_threads: list[str] = Field(default_factory=list)
    min_thread_messages: int = 0  # skip threads thinner than this before sessionizing
    # Cap any one thread's share of this source's records (0 = uncapped). A single
    # dominant relationship would otherwise BE the lane's assistant voice.
    max_thread_share: float = 0.0

    @model_validator(mode="after")
    def _exactly_one_role_anchor(self) -> SourceConfig:
        if bool(self.assistant_sender) == bool(self.owner_sender):
            raise ValueError(
                f"source {self.id!r}: set exactly one of assistant_sender / owner_sender"
            )
        return self

    def role_for(self, sender: str) -> str:
        if self.assistant_sender:
            return "assistant" if sender == self.assistant_sender else "user"
        return "user" if sender == self.owner_sender else "assistant"


class LaneDConfig(BaseModel):
    sources: list[SourceConfig] = Field(default_factory=list)
    # A silence longer than this splits a stream into separate records: beyond a
    # session boundary, adjacent messages stop being one conversation.
    session_gap_minutes: int = 240
    max_messages_per_record: int = 200
    min_messages_per_record: int = 2
    # A record with only one side is a monologue, not a conversation — and a
    # one-sided run is usually an export artifact (blank outgoing windows), not a
    # real exchange. Dropped rather than trained on.
    require_both_roles: bool = True
    # Keep only the scheme://host/path of shared links. Message exports carry live
    # secrets in query strings and fragments — the raw corpus here contains a
    # 1Password share link whose fragment IS the credential. Scrub runs later and
    # looks for PII, not for capability URLs; this is the cheaper guarantee.
    keep_url_query: bool = False
    # Structural substance floor, applied before any model sees the record. The
    # dominant content of a six-year message export is not bad conversation, it is
    # EMPTY conversation ("k", "on my way", "yup"). The sorcha_lane_d rubric scores
    # exactly this, but paying a judge call per acknowledgement token is waste — so
    # the obvious cases die here and the rubric handles the judgement calls.
    min_record_chars: int = 0  # total content chars across the record
    min_mean_message_chars: int = 0  # guards long threads made entirely of tokens
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
                record = _record_from_segment(source, cfg, run_id, conv_id, seg_idx, segment)
                if record is not None:
                    yield record


def _record_from_segment(
    source: SourceConfig,
    cfg: LaneDConfig,
    run_id: str,
    conv_id: str,
    seg_idx: int,
    segment: list[dict],
) -> ConversationRecord | None:
    """One sessionized segment -> one record, or None if it isn't a conversation."""
    if len(segment) < cfg.min_messages_per_record:
        return None
    messages = [
        Message(
            role=source.role_for(m["sender"]),
            speaker=m["sender"],
            content=m["text"],
            ts=m["ts"],
        )
        for m in segment
    ]
    if cfg.require_both_roles and len({m.role for m in messages}) < 2:
        return None
    total = sum(len(m.content or "") for m in messages)
    if total < cfg.min_record_chars:
        return None
    if cfg.min_mean_message_chars and total / len(messages) < cfg.min_mean_message_chars:
        return None

    month = _parse_ts(segment[0]["ts"]).strftime("%Y-%m")
    family = f"{source.id}/{month}"
    ref = SourceRef(
        kind="personal_stream",
        detail={"source": source.id, "conversation_id": conv_id, "segment": seg_idx},
    )
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


APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=UTC)
OBJECT_REPLACEMENT = "￼"  # Apple's inline-attachment placeholder inside text


def _apple_ts(nanos: int) -> datetime:
    """imessage-exporter timestamps are nanoseconds since the Core Data epoch."""
    return APPLE_EPOCH + timedelta(seconds=nanos / 1_000_000_000)


def _clean_url(raw: str, keep_query: bool) -> str:
    if keep_query:
        return raw
    split = urlsplit(raw)
    return urlunsplit((split.scheme, split.netloc, split.path, "", ""))


def _part_text(part: dict, keep_query: bool) -> str:
    """Text carried by one message part, '' if the part is not textual.

    Attachments ('segments') carry no text and are deliberately not stubbed with a
    marker: a placeholder would teach the model to emit '[image]'.
    """
    kind = part.get("type")
    if kind == "text":
        return (part.get("text") or "").replace(OBJECT_REPLACEMENT, "").strip()
    if kind == "edited":
        # The last revision is what the sender meant; earlier ones are keystrokes.
        history = part.get("history") or []
        return (history[-1].get("text") or "").strip() if history else ""
    if kind == "url":
        url = part.get("url") or ""
        return _clean_url(url, keep_query) if url else ""
    return ""


class IMessageParser:
    """`imessage-exporter` JSONL. One file per thread, one message per line:

        {"guid": …, "timestamp": 612011036240000000,   # ns since 2001-01-01
         "sender": "Me" | "<display name>", "is_from_me": bool,
         "service": "iMessage" | "SMS" | "RCS", "type": "message" | "announcement",
         "parts": [{"type": "text", "text": …} | {"type": "segments", …} | …]}

    `source.path` is the export DIRECTORY; each `<thread>.jsonl` becomes one
    conversation stream, sessionized and role-mapped like every other lane D source.
    Announcements (renames, joins) are skipped, as are messages that reduce to no
    text at all — a bare attachment or a tapback is not a turn.
    """

    def parse(
        self, source: SourceConfig, cfg: LaneDConfig, run_id: str
    ) -> Iterator[ConversationRecord]:
        by_thread: dict[str, list[ConversationRecord]] = {}
        for thread, msgs in self._threads(source, cfg):
            kept = [
                record
                for seg_idx, segment in enumerate(_sessionize(msgs, cfg))
                if (record := _record_from_segment(source, cfg, run_id, thread, seg_idx, segment))
                is not None
            ]
            if kept:
                by_thread[thread] = kept
        yield from _apply_thread_cap(by_thread, source.max_thread_share)

    def _threads(self, source: SourceConfig, cfg: LaneDConfig) -> Iterator[tuple[str, list[dict]]]:
        paths = sorted(source.path.glob("*.jsonl")) if source.path.is_dir() else [source.path]
        include, exclude = set(source.include_threads), set(source.exclude_threads)
        for path in paths:
            thread = path.stem
            if (include and thread not in include) or thread in exclude:
                continue
            msgs = list(self._messages(path, cfg))
            if len(msgs) < source.min_thread_messages:
                continue
            if msgs:
                yield thread, msgs

    def _messages(self, path: Path, cfg: LaneDConfig) -> Iterator[dict]:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                if raw.get("type") != "message":
                    continue
                sender, ts = raw.get("sender"), raw.get("timestamp")
                if not sender or ts is None:
                    continue
                text = "\n".join(
                    t
                    for t in (
                        _part_text(p, cfg.keep_url_query)
                        for p in (raw.get("parts") or [])
                        if isinstance(p, dict)
                    )
                    if t
                )
                if not text:
                    continue
                yield {"ts": _apple_ts(ts).isoformat(), "sender": sender, "text": text}


def _apply_thread_cap(
    by_thread: dict[str, list[ConversationRecord]], max_share: float
) -> Iterator[ConversationRecord]:
    """Downsample threads that exceed `max_share` of the total.

    Solved by repeated tightening rather than one pass: capping the biggest thread
    shrinks the total, which lowers the cap for the next one. Survivors are taken on
    an even stride across the thread's whole span, so a capped thread stays
    representative of six years instead of collapsing to its first months.
    """
    if max_share <= 0 or not by_thread:
        yield from (r for records in by_thread.values() for r in records)
        return

    counts = {thread: len(records) for thread, records in by_thread.items()}
    while True:
        total = sum(counts.values())
        limit = max(1, int(total * max_share))
        over = {t: c for t, c in counts.items() if c > limit}
        if not over:
            break
        for thread in over:
            counts[thread] = limit

    for thread, records in by_thread.items():
        keep = counts[thread]
        if keep >= len(records):
            yield from records
            continue
        step = len(records) / keep
        yield from (records[int(i * step)] for i in range(keep))


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
    "imessage": IMessageParser(),
    "discord": _StubParser("discord"),
    "journal": _StubParser("journal"),
}


def parse_all(cfg: LaneDConfig, run_id: str) -> Iterator[ConversationRecord]:
    for source in cfg.sources:
        if source.parser not in PARSERS:
            raise KeyError(f"unknown lane D parser {source.parser!r} for source {source.id!r}")
        yield from PARSERS[source.parser].parse(source, cfg, run_id)
