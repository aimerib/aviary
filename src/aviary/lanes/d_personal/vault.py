"""Obsidian vault reader: the owner's notes as lane D grounding.

CLAUDE.md scopes lane D to "the owner's private streams (chat exports, journals,
notes)" — the vault is the journals-and-notes half. It is read for two things:

1. **The knowledge base** (`Sorcha/`): fact sheets, people profiles, a timeline,
   and a voice guide, already synthesized from the same iMessage export. This is
   what turns a generically-Irish assistant into someone who has known the owner
   for fifteen years. It grounds GENERATION; SOUL.md is explicit that owner
   knowledge "enters via training data and memory, never via prompt text", so
   whether any of it survives into a record's system prompt is a per-seed choice
   (`Seed.render_grounding`), defaulting to no.
2. **The owner's own writing** (Zettelkasten and friends): the topics he actually
   goes deep on. Sorcha "lights up when someone explains what they love" — these
   note titles are the list of things he loves.

Radioactive-data rule: this module reads LOCAL paths named in lane_d.yaml. No
vault content may enter git, tests, fixtures, goldens, or docs; every test here
builds a synthetic vault in a tmp_path.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
BULLET = re.compile(r"^[-*]\s+(.*)$", re.MULTILINE)
# Timeline bullets lead with a bolded date: "- **Jan 1:** Gram died ..."
DATED_BULLET = re.compile(r"^[-*]\s+\*\*(?P<when>[^*]+?):?\*\*:?\s*(?P<what>.+)$", re.MULTILINE)
SECTION = re.compile(
    r"^##\s+(?P<name>.+?)\s*$\n(?P<body>.*?)(?=^##\s|\Z)", re.MULTILINE | re.DOTALL
)


class VaultConfig(BaseModel):
    """Local vault wiring. Paths only — never content (radioactive-data rule)."""

    path: Path  # vault root
    kb_dir: str = "Sorcha"  # the knowledge base built for this persona
    # Folders of the owner's OWN writing; their titles become "things he loves".
    notes_dirs: list[str] = Field(default_factory=lambda: ["Zettelkasten"])
    voice_note: str = "Voice/Voice & texting style.md"  # relative to kb_dir


@dataclass(frozen=True)
class Note:
    rel: str  # path relative to the vault root; NEVER logged into the repo
    title: str
    section: str  # Me | People | Timeline | Voice | notes
    frontmatter: dict
    body: str

    @property
    def family_key(self) -> str:
        """Stable, name-free family id.

        Person and note titles are real names and private topics. They must not
        become greppable strings in a committed config, and holdout is declared by
        family — so families are hashed. Stable across runs because it is a pure
        function of the relative path.
        """
        digest = hashlib.sha256(self.rel.encode()).hexdigest()[:8]
        return f"{self.section.lower()}-{digest}"

    def bullets(self) -> list[str]:
        return [b.strip() for b in BULLET.findall(self.body) if len(b.strip()) > 40]

    def dated_events(self) -> list[tuple[str, str]]:
        return [
            (m.group("when").strip(), m.group("what").strip())
            for m in DATED_BULLET.finditer(self.body)
        ]

    def sections(self) -> dict[str, str]:
        return {m.group("name"): m.group("body").strip() for m in SECTION.finditer(self.body)}

    def summary(self, limit: int = 1200) -> str:
        """Lead paragraph(s), trimmed — enough to ground a scene without pasting a
        whole dossier into a prompt (SOUL.md: 'callbacks are in-character memory,
        never a recited dossier')."""
        text = self.body.strip()
        return text if len(text) <= limit else text[:limit].rsplit("\n", 1)[0] + " …"


@dataclass
class Vault:
    root: Path
    kb: dict[str, list[Note]] = field(default_factory=dict)
    notes: list[Note] = field(default_factory=list)
    voice: str = ""

    @classmethod
    def load(cls, cfg: VaultConfig) -> Vault:
        root = cfg.path.expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"vault not found: {root}")
        vault = cls(root=root)

        kb_root = root / cfg.kb_dir
        for path in sorted(kb_root.rglob("*.md")) if kb_root.is_dir() else []:
            rel = path.relative_to(root).as_posix()
            parts = path.relative_to(kb_root).parts
            section = parts[0] if len(parts) > 1 else "Index"
            vault.kb.setdefault(section, []).append(_read(path, rel, section))

        for name in cfg.notes_dirs:
            folder = root / name
            for path in sorted(folder.rglob("*.md")) if folder.is_dir() else []:
                vault.notes.append(_read(path, path.relative_to(root).as_posix(), "notes"))

        voice_path = kb_root / cfg.voice_note
        vault.voice = voice_path.read_text(encoding="utf-8").strip() if voice_path.exists() else ""
        return vault

    def me(self) -> list[Note]:
        return self.kb.get("Me", [])

    def people(self) -> list[Note]:
        return self.kb.get("People", [])

    def timeline(self) -> list[Note]:
        return self.kb.get("Timeline", [])

    def stats(self) -> dict[str, int]:
        counts = {section: len(notes) for section, notes in sorted(self.kb.items())}
        return counts | {"notes": len(self.notes), "voice_chars": len(self.voice)}


def _read(path: Path, rel: str, section: str) -> Note:
    raw = path.read_text(encoding="utf-8")
    front: dict = {}
    match = FRONTMATTER.match(raw)
    if match:
        parsed = yaml.safe_load(match.group(1))
        front = parsed if isinstance(parsed, dict) else {}
        raw = raw[match.end() :]
    title = str(front.get("name") or "").strip() or path.stem
    return Note(rel=rel, title=title, section=section, frontmatter=front, body=raw.strip())
