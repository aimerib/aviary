"""Ingest AO3 works (otwarchive-downloader JSONL) into lane B.

The downloader emits per-range JSONL, one work per line:
`{"id", "title", "metadata": {Rating, Archive Warning, Fandom, Characters,
Relationship, Additional Tags, author, words, ...}, "text"}` (text is plain, not
HTML). This filters, samples fandom-diverse, writes each kept work as `.txt`, and
emits lane_b.yaml entries — the same shape as fetch.py's `score_local`: pure
filter/select logic plus a thin file-writing pass, so the suite exercises the
selection offline.

Content policy (owner-directed 2026-07-24): include everything — all ratings, all
warnings, every fandom — EXCEPT works AO3 itself tags Underage. That single
exclusion keys on AO3's own explicit warning/tag, never a learned classifier, so
downstream classification of the kept corpus is unaffected (the excluded slice is
identifiable by metadata alone). Near-empty works are dropped as invalid input, not
as a content judgment.

AO3 is an RP-appropriate corpus and feeds lane C's RP seeds, so it belongs in its
own designated lane B run, kept apart from the published-fiction prose corpus (see
lanes/c_selfplay/seeds.py). This tool just produces the corpus; placement is a
config choice.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

from aviary.lanes.b_fiction.books import BookConfig
from aviary.lanes.b_fiction.fetch import slugify

# AO3's own explicit signal for the one hard exclusion: the "Underage" archive
# warning or an "Underage" additional tag. Word-boundary so "Minor Character Death"
# (not underage, not sexual) and "under-the-weather" don't match.
_UNDERAGE = re.compile(r"\bunderage\b", re.I)


def is_underage(metadata: dict) -> bool:
    """True if AO3 tags the work Underage — the single hard exclusion (module docs)."""
    blob = f"{metadata.get('Archive Warning', '')} , {metadata.get('Additional Tags', '')}"
    return bool(_UNDERAGE.search(blob))


def primary_fandom(metadata: dict) -> str:
    """First listed fandom, the diversity key. AO3 comma-joins multi-fandom works."""
    fandom = str(metadata.get("Fandom", "") or "").split(",")[0].strip()
    return fandom or "unknown"


def work_id_for(work: dict) -> str:
    return slugify(f"ao3-{work.get('id', '?')}", 80)


def ao3_book_config(work: dict, path: Path) -> BookConfig:
    """A kept AO3 work as a lane B BookConfig. author is already a display name."""
    m = work.get("metadata", {})
    return BookConfig(
        work_id=work_id_for(work),
        path=path,
        author=str(m.get("author", "") or "").strip(),
        title=str(work.get("title", "") or "untitled").strip(),
    )


def iter_works(jsonl_dir: Path) -> Iterator[dict]:
    """Every work across the range files, in id order. Skips blank/corrupt lines so
    one bad line can't abort a 240k-work ingest."""
    for fp in sorted(jsonl_dir.glob("ao3_works_*.jsonl")):
        with fp.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def select_works(
    works: Iterator[dict],
    *,
    limit: int,
    min_words: int,
    per_fandom_cap: int,
    existing: set[str] | None = None,
) -> Iterator[tuple[dict, str]]:
    """Fandom-diverse, underage-excluded selection, deterministic in stream order.

    Yields (work, text) up to `limit`. Caps each fandom at `per_fandom_cap` so a mega
    fandom (Supernatural, Harry Potter) can't dominate an RP corpus that wants breadth
    across 8,900 fandoms. Underage is excluded first; near-empty and already-registered
    works are skipped as invalid/duplicate, not as content judgments."""
    existing = existing or set()
    per_fandom: dict[str, int] = {}
    kept = 0
    for work in works:
        if kept >= limit:
            return
        meta = work.get("metadata", {})
        if is_underage(meta):
            continue
        text = (work.get("text") or "").strip()
        if len(text.split()) < min_words:
            continue
        wid = work_id_for(work)
        title = str(work.get("title", "") or "").strip()
        if wid in existing or title.lower() in existing:
            continue
        fandom = primary_fandom(meta)
        if per_fandom.get(fandom, 0) >= per_fandom_cap:
            continue
        per_fandom[fandom] = per_fandom.get(fandom, 0) + 1
        kept += 1
        yield work, text


def ingest_ao3(
    jsonl_dir: Path,
    out: Path,
    *,
    limit: int,
    min_words: int = 300,
    per_fandom_cap: int = 5,
    existing: set[str] | None = None,
    log=print,
) -> list[BookConfig]:
    """Select a fandom-diverse, underage-excluded sample and write each work as `.txt`
    to `out` (OUTSIDE the repo). Returns lane B BookConfigs for the kept works."""
    out.mkdir(parents=True, exist_ok=True)
    kept: list[BookConfig] = []
    fandoms: set[str] = set()
    for work, text in select_works(
        iter_works(jsonl_dir),
        limit=limit,
        min_words=min_words,
        per_fandom_cap=per_fandom_cap,
        existing=existing,
    ):
        cfg = ao3_book_config(work, out / f"{work_id_for(work)}.txt")
        cfg.path.write_text(text, encoding="utf-8")
        kept.append(cfg)
        fandoms.add(primary_fandom(work.get("metadata", {})))
        if len(kept) % 250 == 0:
            log(f"  {len(kept)}/{limit} kept, {len(fandoms)} fandoms")
    log(
        f"\nkept {len(kept)} AO3 works across {len(fandoms)} fandoms -> {out}\n"
        "(Underage-tagged works excluded per policy; everything else included.)"
    )
    return kept
