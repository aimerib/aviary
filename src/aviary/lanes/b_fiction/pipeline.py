"""Lane B stage runner: books -> profiles -> scenes -> dialogue -> records.

Each stage checkpoints to $AVIARY_DATA_DIR/<run_id>/laneb/; the teacher response
cache makes reruns cheap, so the pipeline is resumable and idempotent.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel

from aviary.hashing import canonical_json
from aviary.io.jsonl import read_jsonl, write_jsonl
from aviary.io.store import RunStore
from aviary.lanes.b_fiction.assemble import assemble_record
from aviary.lanes.b_fiction.books import BookConfig, chunk_text, load_book_text
from aviary.lanes.b_fiction.extract import extract_dialogue, extract_profiles, find_scenes
from aviary.lanes.b_fiction.models import ExtractedScene, ProfileSet
from aviary.lanes.common import ExtractionError
from aviary.teacher.client import TeacherClient
from aviary.teacher.prompts import PromptSet

log = logging.getLogger(__name__)

def _fingerprint(payload: dict) -> str:
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:16]


def _read_checkpoint[M: BaseModel](path: Path, fingerprint: str, model: type[M]) -> list[M] | None:
    """Reuse a stage checkpoint ONLY when its recorded fingerprint matches the
    current inputs (prompt hash, model, source text, config). A missing sidecar or
    a mismatch returns None so the stage re-extracts — never trust stale extracted
    data under a new run/config (the "idempotent" claim holds only for equal inputs)."""
    meta = path.with_suffix(".meta.json")
    if not path.exists() or not meta.exists():
        return None
    try:
        recorded = json.loads(meta.read_text()).get("fingerprint")
    except json.JSONDecodeError:
        return None
    if recorded != fingerprint:
        log.warning("lane B checkpoint %s stale (inputs changed) — re-extracting", path.name)
        return None
    return list(read_jsonl(path, model))


def _write_checkpoint(path: Path, fingerprint: str, records: list) -> None:
    write_jsonl(path, records)
    path.with_suffix(".meta.json").write_text(json.dumps({"fingerprint": fingerprint}))


@dataclass
class LaneBConfig:
    books: list[BookConfig]
    profile_sample_chars: int = 60_000
    chunk_target_chars: int = 16_000
    min_turns: int = 6
    min_thought_coverage: float = 0.5

    @classmethod
    def load(cls, path: Path) -> LaneBConfig:
        raw = yaml.safe_load(path.read_text())
        books = [
            BookConfig(
                work_id=b["work_id"],
                path=Path(b["path"]).expanduser(),
                author=b.get("author", ""),
                title=b.get("title", ""),
                holdout=b.get("holdout", False),
                max_chunks=b.get("max_chunks"),
            )
            for b in raw["books"]
        ]
        opts = {k: v for k, v in raw.items() if k != "books"}
        return cls(books=books, **opts)


def _stride(items: list[BookConfig], n: int) -> list[BookConfig]:
    """`n` items spread evenly across `items`, deterministically."""
    if n <= 0:
        return []
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def sample_books(books: list[BookConfig], max_books: int) -> list[BookConfig]:
    """Cap the corpus to `max_books`, keeping the holdout ratio and the author mix.

    A pilot does not need all 482 books — lane C samples 20 seeds off the back of
    it — but it does need a book list shaped like the real one. Taking the first N
    would take N books by one or two authors, and could take zero holdout books,
    which leaves the split rule untested precisely where it hard-fails.

    So: stride each stratum separately, then restore config order. Deterministic,
    no seed — the same cap always yields the same corpus, which is what makes the
    response cache worth anything across pilot reruns.
    """
    if max_books <= 0 or len(books) <= max_books:
        return list(books)
    held = [b for b in books if b.holdout]
    train = [b for b in books if not b.holdout]
    if not held or not train:
        return _stride(books, max_books)
    # Round the holdout share, but always leave room for at least one of each.
    n_held = max(1, round(max_books * len(held) / len(books)))
    n_held = min(n_held, max_books - 1, len(held))
    picked = _stride(train, max_books - n_held) + _stride(held, n_held)
    order = {b.work_id: i for i, b in enumerate(books)}
    return sorted(picked, key=lambda b: order[b.work_id])


def run_lane_b(
    cfg: LaneBConfig,
    store: RunStore,
    client: TeacherClient,
    prompts: PromptSet,
    models: dict[str, str],  # {"profiles": id, "scenes": id, "dialogue": id}
    max_workers: int = 1,
) -> int:
    """Returns number of records written to raw/lane_b.jsonl. Books are independent
    (separate per-work checkpoints), so they run concurrently when max_workers > 1;
    default 1 preserves serial behavior."""
    from aviary.teacher.pool import TeacherPool

    pool = TeacherPool(client, max_workers=max_workers)
    results = pool.run(
        [lambda b=book: _run_book(b, cfg, store, client, prompts, models) for book in cfg.books]
    )
    records = []
    for book, res in zip(cfg.books, results, strict=True):
        if isinstance(res, ExtractionError):
            log.warning("book %s failed extraction: %s", book.work_id, res)
        elif isinstance(res, Exception):
            raise res  # non-extraction failures are bugs, not skippable rejects
        else:
            records.extend(res)
    n = write_jsonl(store.raw("b"), records)
    log.info("lane B: %d records from %d books", n, len(cfg.books))
    return n


def _run_book(
    book: BookConfig,
    cfg: LaneBConfig,
    store: RunStore,
    client: TeacherClient,
    prompts: PromptSet,
    models: dict[str, str],
) -> list:
    text = load_book_text(book.path)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    chunks = chunk_text(text, target_chars=cfg.chunk_target_chars)
    if book.max_chunks:
        chunks = chunks[: book.max_chunks]

    profiles_path = store.stage("b", f"profiles_{book.work_id}")
    profiles_fp = _fingerprint(
        {
            "stage": "profiles",
            "prompt_set_hash": prompts.hash,
            "model": models["profiles"],
            "text_hash": text_hash,
            "profile_sample_chars": cfg.profile_sample_chars,
        }
    )
    cached_profiles = _read_checkpoint(profiles_path, profiles_fp, ProfileSet)
    if cached_profiles:
        profiles = cached_profiles[0]
    else:
        profiles = extract_profiles(
            book.work_id, text[: cfg.profile_sample_chars], client, prompts, models["profiles"]
        )
        _write_checkpoint(profiles_path, profiles_fp, [profiles])

    scenes_path = store.stage("b", f"scenes_{book.work_id}")
    scenes_fp = _fingerprint(
        {
            "stage": "scenes",
            "prompt_set_hash": prompts.hash,
            "scenes_model": models["scenes"],
            "dialogue_model": models["dialogue"],
            "text_hash": text_hash,
            "profiles_fp": profiles_fp,
            "chunk_target_chars": cfg.chunk_target_chars,
            "max_chunks": book.max_chunks,
            "min_turns": cfg.min_turns,
            "min_thought_coverage": cfg.min_thought_coverage,
        }
    )
    cached_scenes = _read_checkpoint(scenes_path, scenes_fp, ExtractedScene)
    if cached_scenes is not None:
        extracted = cached_scenes
    else:
        extracted = []
        for chunk in chunks:
            for scene in find_scenes(chunk, profiles, client, prompts, models["scenes"]):
                try:
                    result = extract_dialogue(
                        chunk,
                        scene,
                        profiles,
                        client,
                        prompts,
                        models["dialogue"],
                        min_turns=cfg.min_turns,
                        min_thought_coverage=cfg.min_thought_coverage,
                    )
                except ExtractionError as e:
                    log.warning("scene %s/%s dropped: %s", book.work_id, scene.scene_idx, e)
                    continue
                if result is not None:
                    extracted.append(result)
        _write_checkpoint(scenes_path, scenes_fp, extracted)
    log.info("book %s: %d chunks -> %d scenes kept", book.work_id, len(chunks), len(extracted))

    return [
        assemble_record(
            scene,
            profiles,
            run_id=store.run_id,
            holdout=book.holdout,
            teacher_ids={
                "profiles": models["profiles"],
                "scenes": models["scenes"],
                "dialogue": models["dialogue"],
            },
            prompt_set_hash=prompts.hash,
        )
        for scene in extracted
    ]
