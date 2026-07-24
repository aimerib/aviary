"""Self-play seeds: what conversation to have and in whose skin.

Sources: inline scenarios from lane_c.yaml (persona simple-chat seeds), and lane B
output from a prior run (character cards + scenes). Lane-B-derived seeds INHERIT the
work's family so holdout can never leak across lanes.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from aviary.io.jsonl import read_jsonl
from aviary.io.store import RunStore
from aviary.schema.records import ConversationRecord


class Seed(BaseModel):
    seed_id: str
    family: str
    holdout: bool = False
    character_name: str  # who the character-side plays (the persona or a card name)
    card: str  # character-side system prompt body (persona/card + scenario)
    scenario: str  # shown to the user-sim
    user_goal: str = ""  # what the simulated user wants out of the chat

    # Facts the character-side teacher may use to be specific. Shown at GENERATION
    # time; rendered into the record's system prompt only if `render_grounding`.
    # SOUL.md wants owner knowledge in the weights rather than recited from a
    # prompt, so the default keeps grounding out of the training text — the model
    # sees the informed reply without being handed the notes.
    grounding: str = ""
    render_grounding: bool = False
    # How the simulated user actually writes. Beats any amount of "write casually"
    # instruction, because it is that person's real, observed style.
    user_style: str = ""

    # Which user-sim personas may drive THIS seed. Empty = every configured persona.
    #
    # Not an optimization — a correctness fix. Lane C used to cross every seed with
    # every persona, which puts an `nsfw_rper` on a seed about the user's
    # mother-in-law dying and a `lazy_texter` on a companion scene that only makes
    # sense in his own voice. Those records are incoherent AND they multiply the
    # run: seeds x personas x conversations_per_seed was 8,865 conversations for a
    # burn, ~157 hours of teacher calls, most of them mismatched.
    user_sim_personas: list[str] = Field(default_factory=list)


PERSONA_PLACEHOLDER = "{persona}"

# User-sim personas appropriate to character roleplay (never the owner persona).
RP_PERSONAS = ("lazy_texter", "engaged_rper", "nsfw_rper", "task_asker")


def load_inline_seeds(path: Path, persona_system: str, persona_speaker: str) -> list[Seed]:
    """Inline seeds are PERSONA simple-chats: the assistant side plays the build
    target's persona. RP records must never carry this speaker — the harmonizer
    keys off it to know whose voice it may rewrite.

    Seed text is persona-NEUTRAL: it refers to the assistant as `{persona}` and
    that token is substituted with the build target's speaker here. A seed that
    hardcodes a name would put that name in the user-sim's prompt and in the
    character card, which is how Olivia would end up inside a Sorcha corpus —
    the one thing the sorcha target exists to prevent.
    """
    raw = yaml.safe_load(path.read_text()) or {}

    def fill(text: str) -> str:
        return text.replace(PERSONA_PLACEHOLDER, persona_speaker)

    seeds = []
    for s in raw.get("seeds", []):
        scenario = fill(s["scenario"])
        seeds.append(
            Seed(
                seed_id=s["seed_id"],
                family=s["family"],
                holdout=s.get("holdout", False),
                character_name=persona_speaker,
                card=f"{persona_system}\n\n## Scenario\n{scenario}",
                scenario=scenario,
                user_goal=fill(s.get("user_goal", "")),
            )
        )
    return seeds


def seeds_from_lane_b(
    store: RunStore,
    max_per_work: int = 3,
    max_seeds: int | None = None,
    rp_personas: list[str] | None = None,
) -> list[Seed]:
    """Reuse extracted scenes as self-play seeds: the character side plays one
    participant, the user-sim plays the other. `store` is the lane B run whose
    characters seed the RP — point it at an RP-appropriate corpus (AO3), NOT the
    published-fiction prose corpus (novel characters are off-distribution for RP).
    `max_seeds` caps the total via a deterministic even stride across works.
    `rp_personas` is the allowed user-sim pool (defaults to RP_PERSONAS); orchestrate
    passes every `kind: rp` persona so the pool grows without editing this file."""
    path = store.raw("b")
    if not path.exists():
        return []
    rp = list(rp_personas) if rp_personas else list(RP_PERSONAS)
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
                # RP scenes are SillyTavern-shaped: a roleplayer driving a
                # character. The `owner` persona belongs to companion seeds only —
                # it would put the owner's real texting voice into fiction.
                user_sim_personas=rp,
            )
        )
    if max_seeds is not None and len(seeds) > max_seeds:
        step = len(seeds) / max_seeds
        seeds = [seeds[int(i * step)] for i in range(max_seeds)]
    return seeds
