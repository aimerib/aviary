"""Lane A Auto Evol-Instruct: surface generation is the teacher's; ground truth is
recomputed deterministically. These tests pin the recompute + the safe merge."""

from __future__ import annotations

import json

import pytest

from aviary.lanes.a_agentic.evolve import (
    build_prompt,
    derive_count_tagged,
    derive_instances,
    evolve_template,
    existing_instance_keys,
    merge_instances,
    parse_surface_candidates,
)
from aviary.lanes.a_agentic.taskbank import TaskTemplate, expand

LIST_4_2 = "1. Alpha #x\n2. Beta\n3. Gamma #x\n4. Delta"  # 2 of 4 tagged


def _surface(name="Trip Kit", block=LIST_4_2, tag="#x"):
    return {"name": name, "entries_block": block, "tag": tag}


# --- the safety property: ground truth is recomputed, never trusted ----------


def test_derive_recomputes_true_count_and_never_trusts_a_supplied_one():
    # Even if a surface smuggles in a wrong true_count, the deriver ignores it and
    # counts the list itself — the whole point.
    inst = derive_count_tagged(_surface() | {"true_count": 999})
    assert inst is not None
    assert inst["true_count"] == 2  # recomputed from the two '#x' lines, not 999
    assert inst["destination"] == "lists/trip-kit-log.txt"
    assert inst["count_destination"] == "lists/trip-kit-count.txt"
    assert inst["tag"] == "#x"


def test_derive_rejects_degenerate_and_malformed():
    assert derive_count_tagged(_surface(tag="#none")) is None  # tag on nothing
    assert derive_count_tagged(_surface(block="1. A #x\n2. B #x")) is None  # tag on all (also <3)
    assert derive_count_tagged(_surface(block="not a list at all")) is None
    assert derive_count_tagged({"name": "x", "tag": "#x"}) is None  # missing entries_block


def test_derive_instances_dedupes_and_skips_rejects():
    cands = [_surface(), _surface(), _surface(tag="#none")]  # dup + a reject
    got = derive_instances("count.tagged_count", cands)
    assert len(got) == 1


def test_unregistered_template_is_an_error():
    with pytest.raises(KeyError, match="no evolve deriver"):
        derive_instances("web.compare_sources", [{}])


# --- parsing a teacher reply -------------------------------------------------


def test_parse_handles_fences_prose_and_junk():
    arr = json.dumps([_surface()])
    assert len(parse_surface_candidates(f"here you go:\n```json\n{arr}\n```")) == 1
    assert len(parse_surface_candidates(f"sure! {arr} done")) == 1
    assert parse_surface_candidates("no json here") == []
    assert parse_surface_candidates('{"not": "an array"}') == []


# --- safe merge into a zip template ------------------------------------------


def _template() -> dict:
    return {
        "id": "count.tagged_count",
        "param_mode": "zip",
        "params": {
            "entries_block": ["1. A #x\n2. B\n3. C #x\n"],
            "tag": ["#x"],
            "true_count": [2],
            "destination": ["lists/a-log.txt"],
            "count_destination": ["lists/a-count.txt"],
        },
    }


def test_merge_appends_zip_aligned_and_dedupes():
    inst = derive_count_tagged(_surface())
    merged, added = merge_instances(_template(), [inst, inst])  # same instance twice
    assert added == 1
    assert all(len(v) == 2 for v in merged["params"].values())  # every column grew by 1
    # re-merging an instance already present adds nothing
    _, again = merge_instances(merged, [inst])
    assert again == 0


def test_merge_refuses_non_zip_templates():
    with pytest.raises(ValueError, match="zip-mode"):
        merge_instances({"id": "x", "param_mode": "product", "params": {}}, [_surface()])


def test_existing_keys_roundtrip():
    keys = existing_instance_keys(_template())
    assert len(keys) == 1


# --- the loop, offline (fake teacher) ----------------------------------------


def test_evolve_template_generates_derives_and_bounds():
    replies = iter(
        [
            json.dumps(
                [
                    _surface(
                        "Groceries", "1. Rice (low)\n2. Oil\n3. Beans (low)\n4. Salt", "(low)"
                    ),
                    _surface("Chores", "1. Vacuum (done)\n2. Dishes\n3. Laundry (done)", "(done)"),
                ]
            ),
            json.dumps(
                [_surface("Trails", "1. Ridge #done\n2. Loop\n3. Spur #done\n4. Vale", "#done")]
            ),
        ]
    )

    def fake_generate(_prompt: str) -> str:
        return next(replies, "[]")

    out = evolve_template(_template(), fake_generate, n=3, rounds=3)
    assert len(out) == 3
    assert all(o["true_count"] == 2 for o in out)  # each list has 2 tagged, recomputed
    assert len({o["destination"] for o in out}) == 3  # unique destinations


def test_evolve_template_terminates_when_teacher_dries_up():
    out = evolve_template(_template(), lambda _p: "[]", n=10, rounds=3)
    assert out == []


# --- the payoff: an evolved template is valid and expandable -----------------


def test_evolved_template_still_validates_and_expands():
    # Merge two evolved instances, then prove the result is a legal TaskTemplate whose
    # instances the UNCHANGED verifier will judge (params carry the recomputed truth).
    base = _template() | {
        "family": "count",
        "prompt": "Save {entries_block} to {destination}; count {tag} into {count_destination}.",
        "tools": ["write_file"],
        "verifier": "verifiers/count/tagged_count.py",
    }
    evolved = [
        derive_count_tagged(
            _surface("Books", "1. Dune [r]\n2. Sapiens\n3. Emma [r]\n4. 1984", "[r]")
        ),
        derive_count_tagged(
            _surface("Plants", "1. Fern *dry*\n2. Pothos\n3. Basil *dry*\n4. Ivy", "*dry*")
        ),
    ]
    merged, added = merge_instances(base, evolved)
    assert added == 2
    tmpl = TaskTemplate.model_validate(merged)  # zip lists equal length, placeholders ok
    instances = expand(tmpl)
    assert len(instances) == 3  # 1 original + 2 evolved
    # the evolved instances carry their recomputed ground truth into provenance params
    assert {"lists/books-log.txt", "lists/plants-log.txt"} <= {
        i.params["destination"] for i in instances
    }


def test_build_prompt_names_fields_and_withholds_the_answer():
    p = build_prompt(
        _template() | {"description": "count tagged items"}, ("name", "entries_block", "tag"), 5
    )
    assert "entries_block" in p and "5" in p
    assert "do not compute" in p.lower() or "not compute" in p.lower()
