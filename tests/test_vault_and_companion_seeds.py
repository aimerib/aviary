"""Vault reader + companion seed generation.

Radioactive-data rule: every vault here is SYNTHETIC, authored in this file. No
real note, name, or fact from the owner's vault may appear in the repo — which is
also why the tests assert on structure and invariants rather than on content.
"""

from __future__ import annotations

import pytest

from aviary.lanes.c_selfplay.companion_seeds import (
    companion_seeds,
    event_seeds,
    obsession_seeds,
    person_seeds,
    seed_mix,
    topic_seeds,
)
from aviary.lanes.d_personal.vault import Vault, VaultConfig

PERSONA = "You are Fictional Persona, a synthetic test character."
SPEAKER = "Fictional"

VOICE = "Writes in two gears: terse logistics, then 200-word essays. Says 'lol' a lot."


def build_vault(tmp_path, *, people=2, notes=3, events=4, me_bullets=3):
    root = tmp_path / "vault"
    kb = root / "KB"
    for sub in ("Me", "People", "Timeline", "Voice"):
        (kb / sub).mkdir(parents=True, exist_ok=True)
    (root / "Notes").mkdir(parents=True, exist_ok=True)

    (kb / "Voice" / "style.md").write_text(f"---\ntype: voice-guide\n---\n\n{VOICE}\n")

    bullets = "\n".join(
        f"- Facet {i}: a synthetic detail long enough to clear the length floor applied "
        f"by the bullet extractor, number {i}."
        for i in range(me_bullets)
    )
    (kb / "Me" / "temperament.md").write_text(
        f"---\ntype: fact-sheet\ntopic: temperament\n---\n\nSummary line.\n\n{bullets}\n"
    )

    for i in range(people):
        (kb / "People" / f"Person {i}.md").write_text(
            f"---\ntype: person\nname: Person {i}\nrelationship: synthetic friend {i}\n---\n\n"
            f"## Who\nPerson {i} is a fictional test contact.\n\n## Dynamics\nThey banter.\n"
        )

    dated = "\n".join(
        f"- **Mar {i + 1}:** synthetic event number {i} happened." for i in range(events)
    )
    (kb / "Timeline" / "Timeline 2024.md").write_text(
        f"---\ntype: timeline\nyear: 2024\n---\n\n# 2024\n\n**Year in brief:** synthetic.\n\n{dated}\n"
    )

    for i in range(notes):
        (root / "Notes" / f"Obsession {i}.md").write_text(
            f"# Obsession {i}\n\nA thing they love.\n"
        )

    return VaultConfig(path=root, kb_dir="KB", notes_dirs=["Notes"], voice_note="Voice/style.md")


# --- vault reading -----------------------------------------------------------


def test_vault_loads_every_section(tmp_path):
    vault = Vault.load(build_vault(tmp_path))
    assert len(vault.me()) == 1
    assert len(vault.people()) == 2
    assert len(vault.timeline()) == 1
    assert len(vault.notes) == 3
    assert VOICE in vault.voice
    assert vault.stats()["notes"] == 3


def test_frontmatter_is_parsed_and_stripped(tmp_path):
    vault = Vault.load(build_vault(tmp_path))
    person = vault.people()[0]
    assert person.frontmatter["type"] == "person"
    assert person.title == "Person 0"
    assert "---" not in person.body.split("\n")[0]


def test_dated_bullets_and_sections_extract(tmp_path):
    vault = Vault.load(build_vault(tmp_path, events=4))
    events = vault.timeline()[0].dated_events()
    assert len(events) == 4
    assert events[0][0] == "Mar 1"
    assert "synthetic event number 0" in events[0][1]
    assert set(vault.people()[0].sections()) == {"Who", "Dynamics"}


def test_family_keys_are_hashed_not_named(tmp_path):
    """Titles are real names and private topics; families are declared in configs
    and recorded in provenance, so they must never carry the title."""
    vault = Vault.load(build_vault(tmp_path))
    for note in vault.people() + vault.notes:
        assert note.title not in note.family_key
        assert note.family_key.split("-")[-1].isalnum()
    # Stable across reloads.
    again = Vault.load(build_vault(tmp_path))
    assert [n.family_key for n in vault.people()] == [n.family_key for n in again.people()]


def test_missing_vault_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="vault not found"):
        Vault.load(VaultConfig(path=tmp_path / "nope"))


def test_absent_optional_pieces_degrade_quietly(tmp_path):
    root = tmp_path / "bare"
    root.mkdir()
    vault = Vault.load(VaultConfig(path=root, kb_dir="KB", notes_dirs=["Notes"]))
    assert vault.me() == [] and vault.notes == [] and vault.voice == ""


# --- seed generation ---------------------------------------------------------


def test_all_four_seed_kinds_are_produced(tmp_path):
    vault = Vault.load(build_vault(tmp_path))
    seeds = companion_seeds(vault, PERSONA, SPEAKER)
    mix = seed_mix(seeds)
    assert set(mix) == {"event", "person", "topic", "obsession"}
    assert all(count > 0 for count in mix.values())


def test_every_seed_carries_grounding_and_voice(tmp_path):
    vault = Vault.load(build_vault(tmp_path))
    for seed in companion_seeds(vault, PERSONA, SPEAKER):
        assert seed.grounding, seed.seed_id
        assert VOICE in seed.user_style
        assert seed.character_name == SPEAKER
        assert PERSONA in seed.card


def test_grounding_is_generation_only_by_default(tmp_path):
    """SOUL.md wants owner knowledge in the weights, not recited from a prompt the
    model will not have at inference. The card the RECORD keeps must stay clean."""
    vault = Vault.load(build_vault(tmp_path))
    for seed in companion_seeds(vault, PERSONA, SPEAKER):
        assert seed.render_grounding is False
        assert seed.grounding not in seed.card


def test_families_are_holdout_consistent_and_partial(tmp_path):
    vault = Vault.load(build_vault(tmp_path, people=40, notes=40, events=40))
    seeds = companion_seeds(vault, PERSONA, SPEAKER)

    # Holdout is a property of the family, never of individual seeds within one:
    # otherwise two seeds about the same person straddle the train/eval boundary.
    by_family: dict[str, set[bool]] = {}
    for seed in seeds:
        by_family.setdefault(seed.family, set()).add(seed.holdout)
    assert all(len(flags) == 1 for flags in by_family.values())

    held = {f for f, flags in by_family.items() if True in flags}
    assert 0 < len(held) < len(by_family)  # some, not all


def test_seed_ids_are_unique(tmp_path):
    vault = Vault.load(build_vault(tmp_path, people=5, notes=5, events=5))
    seeds = companion_seeds(vault, PERSONA, SPEAKER)
    assert len({s.seed_id for s in seeds}) == len(seeds)


def test_caps_bound_each_kind_independently(tmp_path):
    vault = Vault.load(build_vault(tmp_path, people=10, notes=10, events=10, me_bullets=10))
    assert len(event_seeds(vault, PERSONA, SPEAKER, cap=3)) == 3
    assert len(person_seeds(vault, PERSONA, SPEAKER, cap=4)) == 4
    assert len(topic_seeds(vault, PERSONA, SPEAKER, cap=5)) == 5
    assert len(obsession_seeds(vault, PERSONA, SPEAKER, cap=6)) == 6

    seeds = companion_seeds(
        vault, PERSONA, SPEAKER, max_events=2, max_people=2, max_topics=2, max_obsessions=2
    )
    assert seed_mix(seeds) == {"event": 2, "person": 2, "topic": 2, "obsession": 2}


def test_empty_vault_yields_no_seeds(tmp_path):
    root = tmp_path / "bare"
    root.mkdir()
    vault = Vault.load(VaultConfig(path=root))
    assert companion_seeds(vault, PERSONA, SPEAKER) == []
