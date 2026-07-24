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
import re
from collections import Counter
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

# Parsers that identify the owner from the export itself — Teams/Reddit name them in
# each file's `me`/`account` field, Instagram's owner is the one participant common to
# every thread. They set message roles directly, so a config anchor would be dead
# config AND, being real display names, exactly what the radioactive rule keeps out of
# git. imessage stays anchor-based: its export labels the owner a literal "Me".
_STRUCTURAL_OWNER = {"instagram", "teams", "reddit"}


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

    # Directory-export sources (imessage/instagram/teams): which threads to read.
    include_threads: list[str] = Field(default_factory=list)  # file/dir stems; empty = all
    exclude_threads: list[str] = Field(default_factory=list)
    min_thread_messages: int = 0  # skip threads thinner than this before sessionizing
    # Cap any one thread's share of this source's records (0 = uncapped). A single
    # dominant relationship would otherwise BE the lane's assistant voice.
    max_thread_share: float = 0.0
    # instagram: skip threads with more than N participants (0 = keep all). A group
    # thread collapses several speakers into one "assistant" voice; a companion build
    # usually wants a single relationship per thread.
    max_participants: int = 0
    # Absolute cap on this source's records, strided evenly across them in the order
    # the parser yields (0 = uncapped). For a source that can dwarf the others —
    # reddit's metal account alone is >1600 comments — this takes a representative
    # slice in time order rather than letting one source become the lane.
    max_records: int = 0

    @model_validator(mode="after")
    def _exactly_one_role_anchor(self) -> SourceConfig:
        if self.parser in _STRUCTURAL_OWNER:
            if self.assistant_sender or self.owner_sender:
                raise ValueError(
                    f"source {self.id!r}: parser {self.parser!r} resolves the owner from the "
                    "export itself — leave assistant_sender/owner_sender unset"
                )
            return self
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
            # Structural-owner parsers (instagram/teams) resolve the role from the
            # export and set it on the row; anchor-based sources map it from config.
            role=m.get("role") or source.role_for(m["sender"]),
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
        yield from _threaded_records(self._threads(source, cfg), source, cfg, run_id)

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


def _threaded_records(
    threads: Iterator[tuple[str, list[dict]]],
    source: SourceConfig,
    cfg: LaneDConfig,
    run_id: str,
) -> Iterator[ConversationRecord]:
    """Shared driver for directory-of-threads sources (imessage, instagram, teams):
    sessionize each thread, build records, then apply the thread-dominance cap.

    `threads` yields (thread_name, rows), each row a dict with `ts`/`sender`/`text`
    and optionally a pre-resolved `role` (structural-owner formats). The cap runs
    across the whole source, so it must see every thread before yielding — hence the
    materialized `by_thread` rather than a straight passthrough."""
    by_thread: dict[str, list[ConversationRecord]] = {}
    for thread, rows in threads:
        kept = [
            record
            for seg_idx, segment in enumerate(_sessionize(rows, cfg))
            if (record := _record_from_segment(source, cfg, run_id, thread, seg_idx, segment))
            is not None
        ]
        if kept:
            by_thread[thread] = kept
    yield from _apply_thread_cap(by_thread, source.max_thread_share)


def _demojibake(s: str) -> str:
    """Undo Meta's UTF-8-as-latin-1 double-encoding: bytes that are really UTF-8 get
    stored reinterpreted as latin-1 and JSON-escaped, so 'Ã©' should read 'é' and
    '\\u00f0\\u009f\\u0098\\u0080' is '😀'. Only round-trip when it round-trips cleanly
    — text already correct (any codepoint > 255) or byte runs that aren't valid UTF-8
    are returned untouched. Applied ONCE, and before owner detection, since
    sender_name is mojibaked too."""
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


# Meta auto-generated, non-conversational `content`. Measured in the owner's export:
# "X sent an attachment." (1384), "X liked a message" (101), share stubs (80), and a
# few group-membership system lines. These are events, not turns — training on them
# teaches the model to say "sent an attachment".
_IG_AUTO = re.compile(
    r"sent an attachment\.?\s*$"
    r"|\bliked a message\b"
    r"|\bshared (?:a|an) (?:reel|post|story|link)\b"
    r"|^Reacted .+ to your message\s*$"
    r"|\b(?:named the group|changed the (?:theme|group name|nickname)"
    r"|set (?:the|your) nickname|added|left the group|created the group|removed)\b",
    re.I,
)


def _detect_owner(thread_participants: list[list[str]]) -> str:
    """The owner is whoever is in the most threads: they appear in every one, each
    friend in only their own. Keeps a real display name out of git (unlike a config
    anchor) and survives a malformed thread.

    Refuses to guess: if the top two participants are tied (e.g. a lone DM, where
    owner and friend each appear once), role assignment would be a coin flip — and a
    flipped owner trains the model to produce the owner's own turns. A real DYI export
    has dozens of threads, so a tie means the path is wrong, not that we should pick."""
    counts: Counter[str] = Counter()
    for names in thread_participants:
        counts.update(set(names))
    ranked = counts.most_common(2)
    if not ranked:
        return ""
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        raise ValueError(
            "instagram owner is ambiguous: no participant appears in more threads than "
            "the rest. A real export has many threads sharing one owner — check the path."
        )
    return ranked[0][0]


class InstagramParser:
    """Meta 'Download Your Information' message threads (Instagram/Messenger).

    `source.path` is the export directory; each `<thread>/message_*.json` is one
    conversation (Meta shards long threads across message_1, message_2, …). Three
    format quirks live here and nowhere else:

      * every text field is UTF-8 double-encoded as latin-1 (`_demojibake`);
      * messages are listed newest-first (re-sorted ascending);
      * photo/share/reaction/system rows carry an auto `content` string that is an
        event, not a turn (`_IG_AUTO`), and are dropped like a bare attachment.

    The owner is detected structurally (in every thread) and takes the USER seat —
    the companion mapping, matching imessage's `owner_sender`."""

    def parse(
        self, source: SourceConfig, cfg: LaneDConfig, run_id: str
    ) -> Iterator[ConversationRecord]:
        loaded = self._load(source)
        owner = _detect_owner([participants for _, _, participants in loaded])
        yield from _threaded_records(self._threads(loaded, owner, source), source, cfg, run_id)

    def _load(self, source: SourceConfig) -> list[tuple[str, list[dict], list[str]]]:
        """(thread_name, raw messages, participant names) per thread dir, honoring the
        include/exclude filters and max_participants. Participants are read first so
        the owner can be detected before any role is assigned."""
        include, exclude = set(source.include_threads), set(source.exclude_threads)
        out: list[tuple[str, list[dict], list[str]]] = []
        for tdir in sorted(p for p in source.path.iterdir() if p.is_dir()):
            if (include and tdir.name not in include) or tdir.name in exclude:
                continue
            files = sorted(tdir.glob("message_*.json"))
            if not files:
                continue
            participants: list[str] = []
            messages: list[dict] = []
            for f in files:
                raw = json.loads(f.read_text(encoding="utf-8"))
                if raw.get("participants"):
                    participants = [_demojibake(p.get("name", "")) for p in raw["participants"]]
                messages.extend(raw.get("messages", []))
            if source.max_participants and len(participants) > source.max_participants:
                continue
            out.append((tdir.name, messages, participants))
        return out

    def _threads(
        self, loaded: list[tuple[str, list[dict], list[str]]], owner: str, source: SourceConfig
    ) -> Iterator[tuple[str, list[dict]]]:
        for name, messages, _ in loaded:
            rows = [r for r in (self._row(m, owner) for m in messages) if r]
            rows.sort(key=lambda r: r["ts"])  # Meta lists newest-first
            if len(rows) >= source.min_thread_messages:
                yield name, rows

    @staticmethod
    def _row(m: dict, owner: str) -> dict | None:
        content, sender, ts = m.get("content"), m.get("sender_name"), m.get("timestamp_ms")
        if not content or sender is None or ts is None:
            return None
        content = _demojibake(content)
        if _IG_AUTO.search(content):
            return None
        text = content.strip()
        if not text:
            return None
        sender = _demojibake(sender)
        return {
            "ts": datetime.fromtimestamp(ts / 1000, UTC).isoformat(),
            "sender": sender,
            "text": text,
            "role": "user" if sender == owner else "assistant",
        }


class TeamsParser:
    """Pre-normalized Teams 1-on-1 export: one JSON file per chat,

        {"me": {"name": …}, "participant": {"name": …},
         "messages": [{"from": <name>, "sent": <iso8601 Z>, "text": …}]}

    `me.name` names the owner structurally (no real name in config), and the owner
    takes the USER seat. Each file is one conversation stream, sessionized like every
    other lane D source."""

    def parse(
        self, source: SourceConfig, cfg: LaneDConfig, run_id: str
    ) -> Iterator[ConversationRecord]:
        yield from _threaded_records(self._threads(source), source, cfg, run_id)

    def _threads(self, source: SourceConfig) -> Iterator[tuple[str, list[dict]]]:
        include, exclude = set(source.include_threads), set(source.exclude_threads)
        paths = sorted(source.path.glob("*.json")) if source.path.is_dir() else [source.path]
        for path in paths:
            if (include and path.stem not in include) or path.stem in exclude:
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            owner = (raw.get("me") or {}).get("name", "")
            rows = [
                {
                    "ts": m["sent"],
                    "sender": m["from"],
                    "text": (m.get("text") or "").strip(),
                    "role": "user" if m.get("from") == owner else "assistant",
                }
                for m in raw.get("messages", [])
                if m.get("from") and m.get("sent") and (m.get("text") or "").strip()
            ]
            if len(rows) >= source.min_thread_messages:
                yield path.stem, rows


class RedditParser:
    """Reddit comment exchanges. Each of the owner's comments pairs with the parent it
    replied to: [parent author's post/comment] -> [owner's reply]. `source.path` is a
    directory of per-account export files: {"account", "me", "comments": [...]}.

    Unlike the DM sources, the owner takes the ASSISTANT seat: the record ends on his
    substantive reply, which is the turn worth modeling (owner-directed 2026-07-24, to
    absorb topic engagement, not to seat strangers as the assistant). Records are
    yielded oldest-first across all accounts so `max_records` slices an even sample of
    the whole span rather than one account's recent tail. A deleted/empty parent or
    reply is skipped — half an exchange is not a conversation."""

    def parse(
        self, source: SourceConfig, cfg: LaneDConfig, run_id: str
    ) -> Iterator[ConversationRecord]:
        rows = self._exchanges(source)
        rows.sort(key=lambda r: r["sent"])
        for r in rows:
            segment = [
                {
                    "ts": r["sent"],
                    "sender": r["parent_author"],
                    "text": r["parent_text"],
                    "role": "user",
                },
                {
                    "ts": r["sent"],
                    "sender": r["owner"],
                    "text": r["reply_text"],
                    "role": "assistant",
                },
            ]
            record = _record_from_segment(source, cfg, run_id, r["permalink"], 0, segment)
            if record is not None:
                yield record

    @staticmethod
    def _exchanges(source: SourceConfig) -> list[dict]:
        paths = sorted(source.path.glob("*.json")) if source.path.is_dir() else [source.path]
        deleted = {"", "[deleted]", "[removed]"}
        out: list[dict] = []
        for path in paths:
            raw = json.loads(path.read_text(encoding="utf-8"))
            owner = (raw.get("me") or {}).get("name") or raw.get("account") or "me"
            for i, c in enumerate(raw.get("comments", [])):
                sent = c.get("sent")
                reply = (c.get("text") or "").strip()
                if not sent or reply in deleted:
                    continue
                parent = c.get("in_reply_to") or {}
                parent_text = "\n\n".join(
                    t
                    for t in (
                        (parent.get("title") or "").strip(),
                        (parent.get("text") or "").strip(),
                    )
                    if t
                )
                if parent_text in deleted:
                    continue
                out.append(
                    {
                        "sent": sent,
                        "owner": owner,
                        "parent_author": (parent.get("author") or "").strip() or "reddit",
                        "parent_text": parent_text,
                        "reply_text": reply,
                        "permalink": c.get("permalink") or f"{path.stem}#{i}",
                    }
                )
        return out


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
    "instagram": InstagramParser(),
    "teams": TeamsParser(),
    "reddit": RedditParser(),
    "discord": _StubParser("discord"),
    "journal": _StubParser("journal"),
}


def _stride_cap(records: list[ConversationRecord], limit: int) -> list[ConversationRecord]:
    """An even slice of `records` (in yield order) when it exceeds `limit`."""
    if limit <= 0 or len(records) <= limit:
        return records
    step = len(records) / limit
    return [records[int(i * step)] for i in range(limit)]


def parse_all(cfg: LaneDConfig, run_id: str) -> Iterator[ConversationRecord]:
    for source in cfg.sources:
        if source.parser not in PARSERS:
            raise KeyError(f"unknown lane D parser {source.parser!r} for source {source.id!r}")
        records = PARSERS[source.parser].parse(source, cfg, run_id)
        if source.max_records:
            records = _stride_cap(list(records), source.max_records)
        yield from records
