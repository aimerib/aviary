"""User-simulator personas + deterministic typo injection.

LLMs write unrealistically clean 'user' turns; personas constrain style and the
typo injector applies reproducible post-hoc corruption (rng seed recorded in
provenance) so realism doesn't depend on the teacher's acting skills.
"""

from __future__ import annotations

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
    goals: list[str] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> UserSimPersona:
        return cls.model_validate(yaml.safe_load(path.read_text()))


def load_personas(dir_: Path) -> dict[str, UserSimPersona]:
    return {p.stem: UserSimPersona.load(p) for p in sorted(dir_.glob("*.yaml"))}


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
