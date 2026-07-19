"""Emit hermes-agent batch_runner inputs and CLI invocations.

The pinned batch_runner.py takes NO config file: it is a fire CLI driven entirely
by flags, samples toolsets per prompt from a named distribution in its hardcoded
registries (toolset_distributions.py / toolsets.py), and writes output to
<cwd>/data/<run_name>/. verify_hermes_interface() cross-checks every flag we pass
against the pinned checkout's batch_runner.py main() signature, and our
distribution/tool names against its registries, so we never invent the hermes
interface (CLAUDE.md rule).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from aviary.lanes.a_agentic.taskbank import TaskInstance, rollout_order

# Every flag build_batch_command() emits. Verified as a subset of the pinned
# batch_runner.py main() kwargs before any batch runs.
PASSED_FLAGS = {
    "dataset_file",
    "batch_size",
    "run_name",
    "distribution",
    "model",
    "base_url",
    "api_key",
    "num_workers",
    "max_turns",
    "ephemeral_system_prompt",
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


def build_batch_command(
    *,
    dataset_file: Path,
    run_name: str,
    distribution: str,
    wire_model: str,
    base_url: str,
    api_key: str,
    system_prompt: str,
    num_workers: int,
    batch_size: int,
    max_turns: int,
) -> list[str]:
    """The batch_runner argv (run with cwd=<hermes checkout>). Contains the API
    key — never log the returned command."""
    return [
        "python",
        "batch_runner.py",
        f"--dataset_file={dataset_file}",
        f"--batch_size={batch_size}",
        f"--run_name={run_name}",
        f"--distribution={distribution}",
        f"--model={wire_model}",
        f"--base_url={base_url}",
        f"--api_key={api_key}",
        f"--num_workers={num_workers}",
        f"--max_turns={max_turns}",
        f"--ephemeral_system_prompt={system_prompt}",
    ]


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


def _module_dict_literal(path: Path, name: str) -> dict:
    """Extract a module-level dict assignment entry-by-entry without importing
    hermes code. Entries whose values aren't pure literals (e.g. hermes-cli's
    `_HERMES_CORE_TOOLS` reference) are skipped — if a skipped entry is one we
    actually need, the required-tools check downstream fails loudly."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == name
                    and isinstance(node.value, ast.Dict)
                ):
                    out = {}
                    for key, value in zip(node.value.keys, node.value.values, strict=True):
                        try:
                            out[ast.literal_eval(key)] = ast.literal_eval(value)
                        except ValueError:
                            continue
                    return out
    raise ValueError(f"{name} dict not found in {path.name}")


def _main_kwargs(batch_runner: Path) -> set[str]:
    tree = ast.parse(batch_runner.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return {a.arg for a in node.args.args + node.args.kwonlyargs}
    raise ValueError(f"no main() found in {batch_runner.name}")


def _toolset_tools(toolsets: dict, name: str) -> set[str]:
    """A toolset's tools, with its `includes` expanded recursively."""
    entry = toolsets.get(name, {})
    tools = set(entry.get("tools", []))
    for included in entry.get("includes", []):
        tools |= _toolset_tools(toolsets, included)
    return tools


def verify_hermes_interface(
    hermes_checkout: Path,
    distribution: str | None = None,
    required_tools: list[str] | tuple[str, ...] = (),
) -> set[str]:
    """Cross-check aviary's assumptions against the pinned checkout's source.

    Always: every flag we pass exists in batch_runner.py main(). With a
    distribution: it exists in DISTRIBUTIONS, and every required tool (the task
    bank's union) is served by a toolset the distribution enables at 100% —
    probabilistic toolsets don't count, a task's tools must ALWAYS be present.
    Returns the set of always-present tools."""
    flags = _main_kwargs(hermes_checkout / "batch_runner.py")
    invented = PASSED_FLAGS - flags
    if invented:
        raise ValueError(
            f"aviary passes batch_runner flags the pinned checkout's main() does not "
            f"accept: {sorted(invented)} — update build_batch_command to match the "
            "checkout, not the other way around"
        )
    if distribution is None:
        return set()

    distributions = _module_dict_literal(
        hermes_checkout / "toolset_distributions.py", "DISTRIBUTIONS"
    )
    if distribution not in distributions:
        raise ValueError(
            f"unknown hermes distribution {distribution!r}; pinned checkout has: "
            f"{sorted(distributions)}"
        )
    toolsets = _module_dict_literal(hermes_checkout / "toolsets.py", "TOOLSETS")
    always_present: set[str] = set()
    for toolset_name, probability in distributions[distribution]["toolsets"].items():
        if probability >= 100:
            always_present |= _toolset_tools(toolsets, toolset_name)
    missing = set(required_tools) - always_present
    if missing:
        raise ValueError(
            f"task bank needs tools not guaranteed by distribution {distribution!r}: "
            f"{sorted(missing)} (always-present: {sorted(always_present)}) — pick a "
            "distribution whose 100% toolsets cover the task bank"
        )
    return always_present
