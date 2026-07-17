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

from aviary.lanes.a_agentic.taskbank import TaskInstance, rollout_order

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
    """One line per rollout, in the prompt_index order defined by rollout_order().
    ingest maps prompt_index i back to the same expansion."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = rollout_order(instances)
    with out_path.open("w", encoding="utf-8") as f:
        for inst in lines:
            f.write(json.dumps({"prompt": inst.prompt}, ensure_ascii=False) + "\n")
    return len(lines)


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


def verify_hermes_pin(hermes_checkout: Path, expected_pin: str) -> str:
    """Return the checkout's `git describe`, raising if it doesn't match expected_pin.
    Called at run start (not just from `just install`) so a burn can never record a
    hermes pin the working checkout doesn't actually match."""
    import subprocess

    head = subprocess.run(
        ["git", "-C", str(hermes_checkout), "describe", "--tags", "--always"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if expected_pin and head != expected_pin:
        raise ValueError(
            f"hermes checkout at {head!r} but teachers.yaml pins {expected_pin!r} — "
            "align the checkout (`just install`) before generating"
        )
    return head


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
