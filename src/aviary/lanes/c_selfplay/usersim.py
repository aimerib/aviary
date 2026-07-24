"""User-simulator personas + deterministic typo injection.

LLMs write unrealistically clean 'user' turns; personas constrain style and the
typo injector applies reproducible post-hoc corruption (rng seed recorded in
provenance) so realism doesn't depend on the teacher's acting skills.
"""

from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

_ADJACENT = {
    "a": "sq",
    "b": "vn",
    "c": "xv",
    "d": "sf",
    "e": "wr",
    "f": "dg",
    "g": "fh",
    "h": "gj",
    "i": "uo",
    "j": "hk",
    "k": "jl",
    "l": "k",
    "m": "n",
    "n": "bm",
    "o": "ip",
    "p": "o",
    "q": "wa",
    "r": "et",
    "s": "ad",
    "t": "ry",
    "u": "yi",
    "v": "cb",
    "w": "qe",
    "x": "zc",
    "y": "tu",
    "z": "x",
}


class UserSimPersona(BaseModel):
    id: str
    description: str = ""
    target_len_words: tuple[int, int] = (3, 25)
    capitalization: str = "normal"  # normal | sloppy
    typo_rate: float = 0.0
    ooc_frequency: float = 0.0
    patience: str = "medium"  # low | medium | high
    # rp = a character-roleplay archetype (drives AO3-seeded RP); companion = the
    # owner's own voice (companion seeds only). RP membership is derived from this,
    # so the pool grows by adding a `kind: rp` file — no hand-synced id list.
    kind: str = "rp"
    goals: list[str] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> UserSimPersona:
        return cls.model_validate(yaml.safe_load(path.read_text()))


def load_personas(dir_: Path) -> dict[str, UserSimPersona]:
    return {p.stem: UserSimPersona.load(p) for p in sorted(dir_.glob("*.yaml"))}


def rp_persona_ids(personas: dict[str, UserSimPersona]) -> list[str]:
    """The RP-appropriate personas — everything that isn't a companion (owner) voice.
    RP seeds draw from this; the owner persona belongs to companion seeds only."""
    return [pid for pid, p in personas.items() if p.kind == "rp"]


def sample_personas(seed_id: str, allowed: list[str], k: int) -> list[str]:
    """A deterministic per-seed down-sample of the allowed persona pool. k<=0 or
    k>=len returns the whole pool (the cross-product default); otherwise a stable
    draw of k, seeded by seed_id so reruns hit the response cache and each seed keeps
    the same personas. This is what lets a ~16-persona RP pool spread across the
    corpus without crossing every seed with every persona."""
    if k <= 0 or k >= len(allowed):
        return allowed
    picker = random.Random(int(hashlib.sha256(seed_id.encode()).hexdigest()[:8], 16))
    return picker.sample(allowed, k)


def inject_typos(text: str, rate: float, rng: random.Random) -> str:
    """Deterministic, seeded, word-level corruption: adjacent-key swaps, dropped or
    doubled letters, dropped apostrophes. Never touches URLs or code-ish tokens."""
    if rate <= 0:
        return text

    def corrupt_word(word: str) -> str:
        if len(word) < 3 or word.startswith(("http", "`", "/")) or rng.random() > rate:
            return word
        ops = ["swap", "drop", "double", "apostrophe"]
        op = rng.choice(ops)
        i = rng.randrange(1, len(word) - 1)
        ch = word[i].lower()
        if op == "apostrophe" and "'" in word:
            return word.replace("'", "")
        if op == "swap" and ch in _ADJACENT:
            repl = rng.choice(_ADJACENT[ch])
            return word[:i] + repl + word[i + 1 :]
        if op == "drop":
            return word[:i] + word[i + 1 :]
        if op == "double":
            return word[: i + 1] + word[i] + word[i + 1 :]
        return word

    return " ".join(corrupt_word(w) for w in text.split(" "))


def apply_style(text: str, persona: UserSimPersona, rng: random.Random) -> str:
    out = text
    if persona.capitalization == "sloppy":
        out = out[0].lower() + out[1:] if out else out
        out = re.sub(r"[.!]+$", "", out)
    return inject_typos(out, persona.typo_rate, rng)
