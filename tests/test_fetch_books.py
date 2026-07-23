from __future__ import annotations

from pathlib import Path

from aviary.lanes.b_fiction.books import BookConfig
from aviary.lanes.b_fiction.fetch import (
    book_config,
    dump_book_yaml,
    existing_keys,
    interiority_score,
    is_epub,
    percentiles,
    slugify,
)

# Prose with named inner states vs prose that is pure external action. The scorer
# must rank the first far above the second — that ordering IS the feature.
INTERIOR = (
    "She wondered whether he had noticed. She felt a flush of shame, then remembered "
    "the letter and knew, suddenly, that she had been wrong. He thought so too, she "
    "suspected, though she doubted he would ever say it. She hoped, she feared, she "
    "considered the whole tangled business and resented how much she still cared. "
) * 40
EXTERNAL = (
    "The cart rolled down the hill. He opened the gate and walked through it. The dog "
    "ran across the yard. She picked up the bucket and carried it to the well. Water "
    "spilled on the stones. The sun set behind the ridge and the lamps came on. "
) * 40


def test_interiority_ranks_inner_life_above_external_action():
    assert interiority_score(INTERIOR) > interiority_score(EXTERNAL)
    # And the gap is decisive, not marginal — the threshold ~7 must fall between them.
    assert interiority_score(INTERIOR) > 7.0 > interiority_score(EXTERNAL)


def test_interiority_is_zero_for_too_short_text():
    assert interiority_score("She thought and felt and wondered.") == 0.0


def test_is_epub_rejects_html_interstitial():
    assert is_epub(b"PK\x03\x04rest-of-a-zip")
    assert not is_epub(b"<!DOCTYPE html><html>gated</html>")
    assert not is_epub(b"")


def test_slugify_is_filesystem_safe_and_bounded():
    assert slugify("Pride & Prejudice: A Novel!") == "pride-prejudice-a-novel"
    assert len(slugify("x" * 200, n=40)) <= 40
    assert slugify("!!!") == "untitled"


def test_book_config_flips_gutenberg_author_order_and_stabilizes_work_id():
    b = book_config(
        source="gutenberg",
        ident="1342",
        path=Path("/tmp/x.txt"),
        author="Austen, Jane",
        title="Pride and Prejudice",
    )
    assert isinstance(b, BookConfig)
    assert b.author == "Jane Austen"  # 'Last, First' -> 'First Last'
    assert b.work_id == "gutenberg-1342"  # deterministic across reruns
    assert b.holdout is False  # left for human review, never auto-held


def test_percentiles_are_monotonic():
    p = percentiles([float(i) for i in range(100)])
    assert p["p10"] <= p["p50"] <= p["p90"]


def test_dump_book_yaml_round_trips_through_lane_b_loader(tmp_path):
    import yaml

    entries = [
        book_config(
            source="gutenberg",
            ident="1342",
            path=tmp_path / "a.txt",
            author="Austen, Jane",
            title="Pride and Prejudice",
        ),
        book_config(
            source="gutenberg",
            ident="2641",
            path=tmp_path / "b.txt",
            author="Forster, E. M.",
            title="A Room with a View",
        ),
    ]
    block = "books:\n" + "\n".join(dump_book_yaml(entries).splitlines()[1:])  # drop the comment
    parsed = yaml.safe_load(block)
    assert [b["work_id"] for b in parsed["books"]] == ["gutenberg-1342", "gutenberg-2641"]
    assert parsed["books"][0]["author"] == "Jane Austen"


def test_existing_keys_reads_titles_and_ids_for_dedupe(tmp_path):
    y = tmp_path / "lane_b.yaml"
    y.write_text(
        'books:\n  - work_id: gutenberg-1342\n    path: "/x"\n    title: "Pride and Prejudice"\n'
    )
    keys = existing_keys(y)
    assert "gutenberg-1342" in keys
    assert "pride and prejudice" in keys  # normalized lowercase


def test_existing_keys_empty_when_no_file(tmp_path):
    assert existing_keys(tmp_path / "nope.yaml") == set()
