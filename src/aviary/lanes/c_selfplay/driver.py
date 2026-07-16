"""Self-play loop: user-sim turn -> character turn -> repeat until a stop condition.

Stop conditions: sampled max_turns, user-sim <END_CHAT> sentinel, char budget,
degenerate-loop detection (n-gram overlap across recent character turns).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from aviary.lanes.c_selfplay.seeds import Seed
from aviary.lanes.c_selfplay.usersim import UserSimPersona, apply_style
from aviary.schema.records import (
    ConversationRecord,
    Message,
    Provenance,
    SourceRef,
    make_record_id,
)
from aviary.teacher.client import ChatRequest, TeacherClient
from aviary.teacher.prompts import PromptSet

END_SENTINEL = "<END_CHAT>"


@dataclass
class LaneCConfig:
    min_turns: int = 8
    max_turns: int = 24
    max_total_chars: int = 60_000
    degenerate_ngram: int = 4
    degenerate_overlap: float = 0.6
    length_targets: tuple[str, ...] = ("brief (1-3 sentences)", "moderate (1-2 paragraphs)")


@dataclass
class LoopState:
    messages: list[Message] = field(default_factory=list)
    total_chars: int = 0


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = text.lower().split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def is_degenerate(state: LoopState, cfg: LaneCConfig) -> bool:
    replies = [m.content for m in state.messages if m.role == "assistant"][-3:]
    if len(replies) < 2:
        return False
    a, b = _ngrams(replies[-2], cfg.degenerate_ngram), _ngrams(replies[-1], cfg.degenerate_ngram)
    if not a or not b:
        return False
    return len(a & b) / min(len(a), len(b)) >= cfg.degenerate_overlap


def _history(messages: list[Message], for_side: str) -> list[dict]:
    """Both sides see the same transcript with roles flipped appropriately."""
    out = []
    for m in messages:
        if for_side == "character":
            role = "user" if m.role == "user" else "assistant"
        else:  # user-sim: their own turns are 'assistant'
            role = "assistant" if m.role == "user" else "user"
        out.append({"role": role, "content": m.content})
    return out


def run_selfplay(
    seed: Seed,
    persona: UserSimPersona,
    cfg: LaneCConfig,
    client: TeacherClient,
    models: dict[str, str],  # {"user_sim": id, "character": id}
    prompts: PromptSet,
    run_id: str,
    prompt_set_hash: str,
    rng: random.Random,
) -> ConversationRecord:
    rng_seed = rng.randrange(2**31)
    local = random.Random(rng_seed)
    max_turns = local.randint(cfg.min_turns, cfg.max_turns)
    length_target = local.choice(cfg.length_targets)

    usersim_system = prompts["lanec_user_sim"].format(
        description=persona.description,
        min_words=persona.target_len_words[0],
        max_words=persona.target_len_words[1],
        goal=seed.user_goal or (local.choice(persona.goals) if persona.goals else "chat"),
        scenario=seed.scenario,
        sentinel=END_SENTINEL,
    )
    character_system = (
        f"{seed.card}\n\n"
        f"Response length for this conversation: {length_target}. "
        f"Never write the other person's messages or actions."
    )

    state = LoopState()
    stop_reason = "max_turns"
    for _turn in range(max_turns):
        user_raw = client.complete(
            ChatRequest(
                model=models["user_sim"],
                system=usersim_system,
                messages=_history(state.messages, "user_sim")
                or [{"role": "user", "content": "(start the conversation)"}],
                temperature=0.9,
                max_tokens=300,
            ),
            lane="c",
        ).text.strip()
        if END_SENTINEL in user_raw:
            stop_reason = "user_ended"
            break
        user_text = apply_style(user_raw, persona, local)
        state.messages.append(Message(role="user", content=user_text))

        char_text = client.complete(
            ChatRequest(
                model=models["character"],
                system=character_system,
                messages=_history(state.messages, "character"),
                temperature=0.85,
                max_tokens=1024,
            ),
            lane="c",
        ).text.strip()
        state.messages.append(
            Message(role="assistant", speaker=seed.character_name, content=char_text)
        )
        state.total_chars += len(user_text) + len(char_text)

        if is_degenerate(state, cfg):
            stop_reason = "degenerate"
            break
        if state.total_chars > cfg.max_total_chars:
            stop_reason = "budget"
            break

    source = SourceRef(
        kind="selfplay_seed",
        detail={
            "seed_id": seed.seed_id,
            "usersim_id": persona.id,
            "rng_seed": rng_seed,
            "stop_reason": stop_reason,
            "length_target": length_target,
        },
    )
    return ConversationRecord(
        system=character_system,
        messages=state.messages,
        provenance=Provenance(
            record_id=make_record_id("c", run_id, source),
            lane="c",
            run_id=run_id,
            family=seed.family,
            holdout=seed.holdout,
            teachers={"user_sim": models["user_sim"], "character": models["character"]},
            prompt_set_hash=prompt_set_hash,
            source=source,
        ),
    )
