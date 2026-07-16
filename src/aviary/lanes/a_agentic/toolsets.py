"""Load per-family tool allowlists + schemas from datagen/toolsets/*.yaml.

The canonical <tools> JSON produced here is byte-stable within a run (cache
discipline): one json.dumps policy, key order as authored in the YAML.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def load_toolset_schemas(toolsets_dir: Path) -> dict[str, str]:
    """family -> canonical JSON array of tool schemas (the <tools> block content)."""
    out: dict[str, str] = {}
    for path in sorted(toolsets_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict) or "family" not in raw:
            continue
        out[raw["family"]] = json.dumps(raw.get("tools", []), ensure_ascii=False)
    return out


def allowed_tool_names(toolsets_dir: Path) -> dict[str, set[str]]:
    names: dict[str, set[str]] = {}
    for family, schema_json in load_toolset_schemas(toolsets_dir).items():
        names[family] = {t.get("function", {}).get("name", "") for t in json.loads(schema_json)} - {
            ""
        }
    return names
