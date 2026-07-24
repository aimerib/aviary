"""AO3 ingestion: filter + fandom-diverse selection. All works here are synthetic."""

from __future__ import annotations

import json

from aviary.lanes.b_fiction.ao3 import (
    ao3_book_config,
    ingest_ao3,
    is_underage,
    primary_fandom,
    select_works,
    work_id_for,
)

BODY = "prose word " * 400  # clears a 300-word floor comfortably


def _work(
    wid,
    *,
    fandom="Generic Fandom",
    warning="No Archive Warnings Apply",
    tags="Fluff",
    title=None,
    author="anon",
    text=BODY,
):
    return {
        "id": str(wid),
        "title": title or f"Work {wid}",
        "metadata": {
            "Fandom": fandom,
            "Archive Warning": warning,
            "Additional Tags": tags,
            "Rating": "Mature",
            "author": author,
        },
        "text": text,
    }


def _select(works, *, limit=100, min_words=300, per_fandom_cap=99, existing=None):
    return list(
        select_works(
            iter(works),
            limit=limit,
            min_words=min_words,
            per_fandom_cap=per_fandom_cap,
            existing=existing,
        )
    )


def test_is_underage_matches_warning_and_tag_not_minor_death():
    assert is_underage({"Archive Warning": "Underage Sex"})
    assert is_underage({"Additional Tags": "Underage, Angst"})
    # "Minor Character Death" is not underage-sexual and must NOT be caught.
    assert not is_underage({"Archive Warning": "Major Character Death"})
    assert not is_underage({"Additional Tags": "Minor Character Death, Slow Burn"})


def test_select_excludes_underage_tagged_works():
    works = [
        _work(1, warning="No Archive Warnings Apply"),
        _work(2, warning="Underage Sex"),  # the one hard exclusion
        _work(3, tags="Underage"),
        _work(4, tags="Fluff, Hurt/Comfort"),
    ]
    kept_ids = [w["id"] for w, _ in _select(works)]
    assert kept_ids == ["1", "4"]  # everything included EXCEPT the two underage-tagged


def test_select_keeps_full_adult_range():
    # No rating ceiling and no other warning is a hard line: non-con, violence, death,
    # and chose-not-to-warn all pass (owner-directed; downstream classifier filters).
    works = [
        _work(1, warning="Rape/Non-Con"),
        _work(2, warning="Creator Chose Not To Use Archive Warnings"),
        _work(3, warning="Graphic Depictions Of Violence"),
    ]
    assert len(_select(works)) == 3


def test_per_fandom_cap_spreads_across_fandoms():
    works = [_work(i, fandom="Supernatural") for i in range(10)] + [
        _work(100 + i, fandom="Stargate") for i in range(10)
    ]
    kept = _select(works, per_fandom_cap=3)
    fandoms = [primary_fandom(w["metadata"]) for w, _ in kept]
    assert fandoms.count("Supernatural") == 3  # capped
    assert fandoms.count("Stargate") == 3
    assert len(kept) == 6


def test_min_words_drops_stubs_and_limit_bounds_output():
    works = [_work(1, text="too short"), _work(2), _work(3), _work(4)]
    assert [w["id"] for w, _ in _select(works, min_words=300)] == ["2", "3", "4"]
    assert len(_select(works, limit=2, min_words=300)) == 2


def test_select_dedupes_against_existing():
    works = [_work(1), _work(2)]
    kept = _select(works, existing={"ao3-1"})
    assert [w["id"] for w, _ in kept] == ["2"]


def test_ao3_book_config_stable_id_and_fields(tmp_path):
    b = ao3_book_config(
        _work(1342, title="A Study in Fic", author="someauthor"), tmp_path / "x.txt"
    )
    assert b.work_id == work_id_for({"id": "1342"}) == "ao3-1342"
    assert b.title == "A Study in Fic"
    assert b.author == "someauthor"
    assert b.holdout is False


def test_ingest_ao3_is_idempotent_by_output_dir(tmp_path):
    src = tmp_path / "jsonl"
    src.mkdir()
    works = [_work(i, fandom=f"Fandom {i}") for i in range(5)]
    (src / "ao3_works_1-10000.jsonl").write_text("\n".join(json.dumps(w) for w in works))
    out = tmp_path / "corpus"

    first = ingest_ao3(src, out, limit=5, log=lambda *a: None)
    assert len(first) == 5
    assert len(list(out.glob("ao3-*.txt"))) == 5

    # Re-run over the same out: every work is already materialized -> nothing new,
    # nothing rewritten, nothing re-emitted. Even a larger limit adds none (source dry).
    assert ingest_ao3(src, out, limit=5, log=lambda *a: None) == []
    assert ingest_ao3(src, out, limit=50, log=lambda *a: None) == []


def test_ingest_ao3_limit_is_target_total_not_per_run(tmp_path):
    src = tmp_path / "jsonl"
    src.mkdir()
    works = [_work(i, fandom=f"Fandom {i}") for i in range(10)]
    (src / "ao3_works_1-10000.jsonl").write_text("\n".join(json.dumps(w) for w in works))
    out = tmp_path / "corpus"

    assert len(ingest_ao3(src, out, limit=4, log=lambda *a: None)) == 4  # fresh: writes 4
    assert ingest_ao3(src, out, limit=4, log=lambda *a: None) == []  # same limit: no-op
    assert len(ingest_ao3(src, out, limit=7, log=lambda *a: None)) == 3  # raise: tops up to 7
    assert len(list(out.glob("ao3-*.txt"))) == 7  # total, not 4+4+7
