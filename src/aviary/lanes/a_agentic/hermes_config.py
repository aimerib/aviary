"""Emit hermes-agent batch_runner inputs and configs.

Config keys mirror the pinned checkout's datagen-config-examples/ (v0.18.2:
environment, toolsets, num_workers, batch_size, max_items, model,
ephemeral_system_prompt, output_dir). `just install` verifies the pinned checkout;
verify_config_keys() cross-checks our emitted keys against its examples so we
never invent hermes config keys (CLAUDE.md rule).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from aviary.lanes.a_agentic.taskbank import TaskInstance

KNOWN_KEYS = {
    "environment",
    "toolsets",
    "num_workers",
    "batch_size",
    "max_items",
    "model",
    "ephemeral_system_prompt",
    "output_dir",
    "compression",
    "eval_every",
    "eval_size",
}


def emit_batch_inputs(instances: list[TaskInstance], out_path: Path) -> int:
    """One line per rollout: instances repeat n_rollouts times (rejection sampling).
    Line order is the prompt_index contract used by ingest."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for inst in instances:
            for _ in range(inst.n_rollouts):
                f.write(json.dumps({"prompt": inst.prompt}, ensure_ascii=False) + "\n")
                n += 1
    return n


def emit_batch_config(
    *,
    toolsets: list[str],
    wire_model: str,
    system_prompt: str,
    output_dir: Path,
    num_workers: int,
    batch_size: int,
    out_path: Path,
) -> Path:
    config = {
        "toolsets": toolsets,
        "num_workers": num_workers,
        "batch_size": batch_size,
        "model": wire_model,
        "ephemeral_system_prompt": system_prompt,
        "output_dir": str(output_dir),
    }
    unknown = set(config) - KNOWN_KEYS
    if unknown:
        raise ValueError(f"unknown hermes config keys (never invent them): {sorted(unknown)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    return out_path


def verify_config_keys(hermes_checkout: Path) -> set[str]:
    """Collect config keys used by the pinned checkout's datagen-config-examples/;
    raises if our KNOWN_KEYS contains anything the examples don't show."""
    examples = hermes_checkout / "datagen-config-examples"
    seen: set[str] = set()
    for path in examples.glob("*.yaml"):
        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict):
            seen.update(data.keys())
    if not seen:
        raise FileNotFoundError(f"no datagen config examples under {examples}")
    invented = KNOWN_KEYS - seen
    if invented:
        raise ValueError(
            f"aviary assumes hermes config keys not present in the pinned checkout's "
            f"examples: {sorted(invented)} — update KNOWN_KEYS/emit_batch_config to match "
            f"the checkout, not the other way around"
        )
    return seen
