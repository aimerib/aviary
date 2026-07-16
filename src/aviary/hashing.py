"""Stable hashes for provenance and cache discipline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def hash_tree(*roots: Path) -> str:
    """sha256 over every file under the given roots (sorted relative paths + bytes).

    Used for datagen_config_hash: covers datagen/configs + toolsets + persona.
    """
    h = hashlib.sha256()
    for root in sorted(roots, key=str):
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if path.name == ".DS_Store":
                continue
            h.update(str(path.relative_to(root.parent)).encode())
            h.update(b"\0")
            h.update(path.read_bytes())
            h.update(b"\0")
    return h.hexdigest()
