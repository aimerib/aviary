"""Companion self-play seeds built from the owner's vault.

The blandness failure mode for a companion model is not the persona prompt — it is
the seed set. Four hand-written scenarios ("the user can't sleep and complains
about their day") produce four flavours of generic warmth no matter how good the
system prompt is. A companion who has known someone fifteen years is specific:
she remembers the move, the job, the grief, the running joke about soup.

So seeds come from what actually happened. Four kinds, all derived from the
knowledge base the owner already built for this persona:

  event      — a dated entry from the Timeline notes
  person     — someone from the People profiles
  topic      — a facet from the Me/ fact sheets
  obsession  — a title from his own notes; the thing he'd explain unprompted,
               which is the exact situation SOUL.md says makes her light up

Each seed carries `grounding` (facts the character-side teacher sees so Sorcha can
be specific) and `user_style` (the vault's voice guide, so the simulated user texts
like the actual person rather than like an LLM). Grounding is generation-only
unless the seed says otherwise — SOUL.md wants this knowledge in the weights, not
recited from a prompt.

Families are HASHED (`Note.family_key`): person and note titles are real names and
private topics, holdout is declared by family, and configs live in git.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from aviary.lanes.c_selfplay.seeds import Seed
from aviary.lanes.d_personal.vault import Note, Vault

# Fraction of companion families held out. Deterministic by family hash rather than
# declared in a config, because declaring them would put real names in git.
HOLDOUT_FRACTION = 0.1

# The user-sim persona for companion seeds: the owner himself.
OWNER_PERSONA = "owner"


def _is_holdout(family: str, fraction: float = HOLDOUT_FRACTION) -> bool:
    bucket = int(hashlib.sha256(f"holdout:{family}".encode()).hexdigest()[:8], 16) % 1000
    return bucket < fraction * 1000


def _seed(
    *,
    seed_id: str,
    family: str,
    persona_system: str,
    persona_speaker: str,
    scenario: str,
    user_goal: str,
    grounding: str,
    user_style: str,
    render_grounding: bool,
) -> Seed:
    return Seed(
        seed_id=seed_id,
        family=f"companion/{family}",
        holdout=_is_holdout(family),
        character_name=persona_speaker,
        card=f"{persona_system}\n\n## Scenario\n{scenario}",
        scenario=scenario,
        user_goal=user_goal,
        grounding=grounding,
        user_style=user_style,
        render_grounding=render_grounding,
        # Companion scenes are the owner talking to his own companion. An RP
        # persona here would be someone else wearing his life as a costume.
        user_sim_personas=[OWNER_PERSONA],
    )


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _variant(options: tuple[str, ...], key: str) -> str:
    """Pick one option by hashing `key`. Deterministic (reruns hit the same cache),
    and spread evenly because the key is a per-seed id rather than a counter."""
    bucket = int(hashlib.sha256(f"variant:{key}".encode()).hexdigest()[:8], 16)
    return options[bucket % len(options)]


# HOW a thing comes up, not what it is. Measured 2026-07-23: 200 of 527 seeds opened
# with one identical framing sentence and 189 with another — 74% of the corpus on two
# templates. The grounding underneath differed, and that turned out to be enough to
# keep the transcripts from repeating verbatim (0.992 distinct 5-grams), so this is
# not a repair of something broken. It widens the range of CONVERSATIONS she has had:
# every one of those 200 was someone calmly raising a memory, and a companion should
# also have been snapped at, bored, interrupted, and told something at 2am.
EVENT_FRAMINGS: tuple[str, ...] = (
    "It comes up sideways, mid-conversation — he is not briefing you, you were there.",
    "Something today rhymed with it and he is still working out why it landed so hard.",
    "He is telling it as a joke. It is funny. It is also not only funny, and you know that.",
    "He is annoyed about it in retrospect, and a bit annoyed at himself for still caring.",
    "He circles it without naming it directly. Let him get there; do not name it for him.",
    "He mentions it in passing while doing something else, and does not expect a response.",
    "He is misremembering a detail. You remember it differently. That is worth a gentle argument.",
    "He brings it up because he wants to be talked out of a conclusion he has drawn from it.",
)

OBSESSION_FRAMINGS: tuple[str, ...] = (
    "He explains it unprompted, at length, because it is live in his head right now.",
    "He is three days in and starting to doubt the whole premise. Do not rescue him too fast.",
    "He is trying to explain it simply and keeps failing, which is annoying him.",
    "He assumes you already know the basics. You do not. Say so and make him back up.",
    "He is excited and it is 2am and he is not going to bed. You are not his minder.",
    "He wants to argue about it. Push back on something specific rather than agreeing.",
    "He drifts off it halfway through into something adjacent he cares about more.",
    "He is bored of it and will not admit that yet.",
)


def event_seeds(vault: Vault, persona_system: str, speaker: str, *, cap: int) -> list[Seed]:
    """One seed per dated timeline entry: the user brings up something that really
    happened, and she is expected to already know the shape of it."""
    out: list[Seed] = []
    for note in vault.timeline():
        year = str(note.frontmatter.get("year") or note.title)
        context = note.summary(900)
        for i, (when, what) in enumerate(note.dated_events()):
            if len(out) >= cap:
                return out
            seed_id = f"companion-event-{note.family_key}-{i}"
            out.append(
                _seed(
                    seed_id=seed_id,
                    family=note.family_key,
                    persona_system=persona_system,
                    persona_speaker=speaker,
                    scenario=(
                        f"The user brings up something from {when} ({year}): {_short(what, 300)} "
                        f"{_variant(EVENT_FRAMINGS, seed_id)}"
                    ),
                    user_goal="talk about it with someone who already knows the background",
                    grounding=(
                        f"## What you remember about {year}\n{context}\n\n"
                        f"## This particular thing\n{when}: {what}"
                    ),
                    user_style=vault.voice,
                    render_grounding=False,
                )
            )
    return out


def person_seeds(vault: Vault, persona_system: str, speaker: str, *, cap: int) -> list[Seed]:
    """One seed per person profile: he mentions someone, she knows who they are."""
    out: list[Seed] = []
    for note in vault.people()[:cap]:
        relationship = str(note.frontmatter.get("relationship") or "someone in his life")
        out.append(
            _seed(
                seed_id=f"companion-person-{note.family_key}",
                family=note.family_key,
                persona_system=persona_system,
                persona_speaker=speaker,
                scenario=(
                    f"The user is talking about {note.title} ({_short(relationship, 120)}). "
                    "Something recent, unresolved, or funny. You know this person through him — "
                    "years of stories — so react like it, and never recite a dossier back at him."
                ),
                user_goal="think out loud about this person with someone who knows them",
                grounding=f"## {note.title}\n{note.summary(1400)}",
                render_grounding=False,
                user_style=vault.voice,
            )
        )
    return out


def topic_seeds(vault: Vault, persona_system: str, speaker: str, *, cap: int) -> list[Seed]:
    """One seed per facet of the Me/ fact sheets — health, work, family, tastes."""
    out: list[Seed] = []
    for note in vault.me():
        topic = str(note.frontmatter.get("topic") or note.title)
        for i, bullet in enumerate(note.bullets()):
            if len(out) >= cap:
                return out
            out.append(
                _seed(
                    seed_id=f"companion-topic-{note.family_key}-{i}",
                    family=note.family_key,
                    persona_system=persona_system,
                    persona_speaker=speaker,
                    scenario=(
                        f"A conversation that lands on {topic}: {_short(bullet, 280)} "
                        "He is not asking for advice unless he asks for advice. Be an accomplice, "
                        "not a minder."
                    ),
                    user_goal="be met where he is on this, not managed",
                    grounding=f"## {topic}\n{note.summary(1200)}",
                    render_grounding=False,
                    user_style=vault.voice,
                )
            )
    return out


def obsession_seeds(vault: Vault, persona_system: str, speaker: str, *, cap: int) -> list[Seed]:
    """One seed per note he wrote himself.

    SOUL.md: 'when someone explains something niche and loved — waveguides, Minoan
    religion, GPU memory layouts — she lights up, not because she knows it but
    because she loves hearing it.' His own note titles ARE that list, so this is the
    single most on-persona scenario available, and it is the one that most needs
    her to ask questions instead of performing expertise.
    """
    out: list[Seed] = []
    for note in vault.notes[:cap]:
        seed_id = f"companion-obsession-{note.family_key}"
        out.append(
            _seed(
                seed_id=seed_id,
                family=note.family_key,
                persona_system=persona_system,
                persona_speaker=speaker,
                scenario=(
                    f'The user is deep in something he cares about: "{note.title}". '
                    f"{_variant(OBSESSION_FRAMINGS, seed_id)} You do "
                    "not know this subject well — that is the point. Light up, interrupt with "
                    "real questions, connect it to things he has said before. Do not pretend to "
                    "be an expert and do not summarize it back at him."
                ),
                user_goal="explain the thing he loves to someone who wants to hear it",
                grounding=f"## What he's working through\n{note.summary(1100)}",
                render_grounding=False,
                user_style=vault.voice,
            )
        )
    return out


def companion_seeds(
    vault: Vault,
    persona_system: str,
    persona_speaker: str,
    *,
    max_events: int = 200,
    max_people: int = 40,
    max_topics: int = 120,
    max_obsessions: int = 200,
) -> list[Seed]:
    """All four kinds. Caps are per-kind so no single kind can swamp the mix."""
    kinds: Iterable[list[Seed]] = (
        event_seeds(vault, persona_system, persona_speaker, cap=max_events),
        person_seeds(vault, persona_system, persona_speaker, cap=max_people),
        topic_seeds(vault, persona_system, persona_speaker, cap=max_topics),
        obsession_seeds(vault, persona_system, persona_speaker, cap=max_obsessions),
    )
    return [seed for kind in kinds for seed in kind]


def seed_mix(seeds: list[Seed]) -> dict[str, int]:
    """Counts by kind — for the manifest and for noticing a collapsed mix."""
    mix: dict[str, int] = {}
    for seed in seeds:
        kind = seed.seed_id.split("-")[1] if seed.seed_id.startswith("companion-") else "other"
        mix[kind] = mix.get(kind, 0) + 1
    return mix


__all__ = [
    "Note",
    "companion_seeds",
    "event_seeds",
    "obsession_seeds",
    "person_seeds",
    "seed_mix",
    "topic_seeds",
]
