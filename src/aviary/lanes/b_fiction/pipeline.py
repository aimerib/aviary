"""Lane B stage runner: books -> profiles -> scenes -> dialogue -> records.

Each stage checkpoints to $AVIARY_DATA_DIR/<run_id>/laneb/; the teacher response
cache makes reruns cheap, so the pipeline is resumable and idempotent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

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


def run_lane_b(
    cfg: LaneBConfig,
    store: RunStore,
    client: TeacherClient,
    prompts: PromptSet,
    models: dict[str, str],  # {"profiles": id, "scenes": id, "dialogue": id}
) -> int:
    """Returns number of records written to raw/lane_b.jsonl."""
    records = []
    for book in cfg.books:
        try:
            records.extend(_run_book(book, cfg, store, client, prompts, models))
        except ExtractionError as e:
            log.warning("book %s failed extraction: %s", book.work_id, e)
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
    chunks = chunk_text(text, target_chars=cfg.chunk_target_chars)
    if book.max_chunks:
        chunks = chunks[: book.max_chunks]

    profiles_path = store.stage("b", f"profiles_{book.work_id}")
    if profiles_path.exists():
        profiles = next(read_jsonl(profiles_path, ProfileSet))
    else:
        profiles = extract_profiles(
            book.work_id, text[: cfg.profile_sample_chars], client, prompts, models["profiles"]
        )
        write_jsonl(profiles_path, [profiles])

    scenes_path = store.stage("b", f"scenes_{book.work_id}")
    if scenes_path.exists():
        extracted = list(read_jsonl(scenes_path, ExtractedScene))
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
        write_jsonl(scenes_path, extracted)
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
