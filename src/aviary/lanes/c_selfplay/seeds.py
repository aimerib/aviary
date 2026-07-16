"""Self-play seeds: what conversation to have and in whose skin.

Sources: inline scenarios from lane_c.yaml (Olivia seeds), and lane B output from a
prior run (character cards + scenes). Lane-B-derived seeds INHERIT the work's family
so holdout can never leak across lanes.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from aviary.io.jsonl import read_jsonl
from aviary.io.store import RunStore
from aviary.schema.records import ConversationRecord


class Seed(BaseModel):
    seed_id: str
    family: str
    holdout: bool = False
    character_name: str  # who the character-side plays ("Olivia" or a card name)
    card: str  # character-side system prompt body (persona/card + scenario)
    scenario: str  # shown to the user-sim
    user_goal: str = ""  # what the simulated user wants out of the chat


def load_inline_seeds(path: Path, olivia_system: str) -> list[Seed]:
    raw = yaml.safe_load(path.read_text()) or {}
    seeds = []
    for s in raw.get("seeds", []):
        seeds.append(
            Seed(
                seed_id=s["seed_id"],
                family=s["family"],
                holdout=s.get("holdout", False),
                character_name="Olivia",
                card=f"{olivia_system}\n\n## Scenario\n{s['scenario']}",
                scenario=s["scenario"],
                user_goal=s.get("user_goal", ""),
            )
        )
    return seeds


def seeds_from_lane_b(store: RunStore, max_per_work: int = 3) -> list[Seed]:
    """Reuse extracted scenes as self-play seeds: the character side plays one
    participant, the user-sim plays the other."""
    path = store.raw("b")
    if not path.exists():
        return []
    seeds: list[Seed] = []
    per_work: dict[str, int] = {}
    for rec in read_jsonl(path, ConversationRecord):
        work = rec.provenance.family
        if per_work.get(work, 0) >= max_per_work:
            continue
        speakers = rec.speakers
        if len(speakers) < 2:
            continue
        per_work[work] = per_work.get(work, 0) + 1
        detail = rec.provenance.source.detail
        seeds.append(
            Seed(
                seed_id=f"laneb-{work}-{detail.get('chunk_idx')}-{detail.get('scene_idx')}",
                family=work,  # family inheritance: lane B holdout stays held out
                holdout=rec.provenance.holdout,
                character_name=speakers[0],
                card=rec.system + f"\n\nYou play {speakers[0]} and only {speakers[0]}.",
                scenario=rec.system.split("## Scene", 1)[-1].strip(),
                user_goal=f"Play {speakers[1]} in this scene.",
            )
        )
    return seeds
