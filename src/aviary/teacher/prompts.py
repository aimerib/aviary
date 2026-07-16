"""PromptSet: every prompt template a run uses, frozen and hashed at run start.

Cache-discipline rule: system prompts and tool schemas must be byte-stable across
a run. The PromptSet is loaded once, hashed, recorded in the manifest, and every
LLM-touching stage asserts the hash before calling out. A mid-run edit changes
the hash and the run refuses to continue.
"""

from __future__ import annotations

from pathlib import Path

from aviary.hashing import sha256_text


class PromptSetError(RuntimeError):
    pass


class PromptSet:
    def __init__(self, texts: dict[str, str]):
        self._texts = dict(sorted(texts.items()))
        joined = "\0".join(f"{k}\1{v}" for k, v in self._texts.items())
        self.hash = sha256_text(joined)

    def __getitem__(self, name: str) -> str:
        try:
            return self._texts[name]
        except KeyError as e:
            raise PromptSetError(f"prompt {name!r} not in frozen PromptSet") from e

    def assert_hash(self, expected: str) -> None:
        if expected and self.hash != expected:
            raise PromptSetError(
                "PromptSet hash mismatch: prompts were edited mid-run. "
                "Treat prompt edits as a new run (cache-discipline rule)."
            )

    @classmethod
    def load(cls, files: dict[str, Path]) -> PromptSet:
        return cls({name: path.read_text() for name, path in files.items()})
