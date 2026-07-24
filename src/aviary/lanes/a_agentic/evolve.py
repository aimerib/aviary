"""Auto Evol-Instruct for lane A: grow a task template's instances with a teacher,
WITHOUT touching its verifier.

The move that makes this contract-safe: a lane A verifier judges the final state
against GROUND-TRUTH params carried on the instance (count.tagged_count checks the
precomputed `true_count`, never the teacher's own tally). So evolution splits into a
creative half and a trusted half:

  1. a teacher invents new SURFACE content — a fresh list, a new domain, a longer or
     trickier variant. This is the in-breadth / in-depth axis of Evol-Instruct, and
     it is the ONLY thing the model is trusted with.
  2. a per-template DERIVER recomputes the ground truth from that surface,
     deterministically. The model never supplies an answer the verifier will trust;
     if the surface is malformed or the task would be trivial, the deriver drops it.
  3. the existing verifier judges the evolved instance unchanged.

So there is no verifier co-evolution, no new fixtures, and no auto-generated verifier
code — the parts the outcome-gate and fixture rules (CLAUDE.md #4/#5) keep
human-audited stay human-audited. Evolved instances are APPENDED to the template YAML
as candidates; a human still owns the difficulty band and the merge. Only templates
with a registered deriver can be evolved; a web-comparison task whose success is not
mechanically checkable is deliberately out of scope.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable

# A deriver takes the teacher's surface params and returns the COMPLETE instance
# params (surface + recomputed ground truth), or None to reject the candidate.
Deriver = Callable[[dict], dict | None]

_SLUG = re.compile(r"[^a-z0-9]+")
_LIST_LINE = re.compile(r"^\s*\d+[.)]\s+(.*\S)\s*$")


def _slug(s: str, n: int = 32) -> str:
    return _SLUG.sub("-", s.lower()).strip("-")[:n].strip("-") or "item"


def _list_items(block: str) -> list[str]:
    """The text of each numbered line ('1. Foo #done' -> 'Foo #done'). Non-list lines
    are ignored, so a stray heading can't corrupt the count."""
    return [m.group(1) for line in block.splitlines() if (m := _LIST_LINE.match(line))]


def derive_count_tagged(surface: dict) -> dict | None:
    """Ground truth for count.tagged_count: `true_count` is RECOMPUTED here as the
    number of list items carrying `tag` — never taken from the teacher. Rejects a
    surface that isn't a real list, whose tag is absent, or where the tag is on all or
    none of the items (a degenerate count teaches nothing and drifts out of band)."""
    block = str(surface.get("entries_block", "")).strip()
    tag = str(surface.get("tag", "")).strip()
    name = str(surface.get("name", "")).strip()
    if not block or not tag or not name:
        return None
    items = _list_items(block)
    if len(items) < 3:
        return None
    tagged = sum(1 for it in items if tag in it)
    if tagged == 0 or tagged == len(items):  # must be a genuine partial count
        return None
    slug = _slug(name)
    return {
        "entries_block": block + "\n",  # match the authored trailing-newline style
        "tag": tag,
        "true_count": tagged,
        "destination": f"lists/{slug}-log.txt",
        "count_destination": f"lists/{slug}-count.txt",
    }


# template id -> (surface fields the teacher must supply, deriver). Add a template by
# registering its deriver here; the recompute is what keeps its verifier valid.
DERIVERS: dict[str, tuple[tuple[str, ...], Deriver]] = {
    "count.tagged_count": (("name", "entries_block", "tag"), derive_count_tagged),
}


def parse_surface_candidates(text: str) -> list[dict]:
    """Pull a JSON array of surface objects out of a teacher reply, tolerating prose
    or a ```json fence around it. A non-array or unparseable reply yields []."""
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    blob = fenced.group(1) if fenced else text[text.find("[") : text.rfind("]") + 1]
    try:
        data = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        return []
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


def derive_instances(template_id: str, candidates: list[dict]) -> list[dict]:
    """Recompute ground truth for each surface candidate and keep the valid ones.
    Deduplicates within the batch by the derived instance's identity."""
    if template_id not in DERIVERS:
        raise KeyError(f"no evolve deriver registered for {template_id!r}")
    _, deriver = DERIVERS[template_id]
    out: list[dict] = []
    seen: set[str] = set()
    for cand in candidates:
        inst = deriver(cand)
        if inst is None:
            continue
        key = json.dumps(inst, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(inst)
    return out


def existing_instance_keys(template: dict) -> set[str]:
    """Identity of each instance already in a zip-mode template, so evolution never
    re-adds one. Keyed by the derived-truth fields that make an instance unique."""
    params = template.get("params", {})
    keys = sorted(params)
    if not keys:
        return set()
    rows = zip(*(params[k] for k in keys), strict=False)
    return {json.dumps(dict(zip(keys, r, strict=False)), sort_keys=True) for r in rows}


def merge_instances(template: dict, instances: list[dict]) -> tuple[dict, int]:
    """Append new zip-aligned instances to a template's param lists, skipping any whose
    param tuple already exists. Returns (updated template, number added). Requires
    param_mode: zip — evolution grows unique-destination instances, not a product."""
    if template.get("param_mode") != "zip":
        raise ValueError("evolve only appends to zip-mode templates (unique destinations)")
    params = {k: list(v) for k, v in template.get("params", {}).items()}
    have = existing_instance_keys(template)
    added = 0
    for inst in instances:
        if not inst or set(inst) != set(params):
            continue  # a rejected (None) or column-mismatched candidate can't be zip-aligned
        key = json.dumps({k: inst[k] for k in sorted(params)}, sort_keys=True)
        if key in have:
            continue
        have.add(key)
        for k in params:
            params[k].append(inst[k])
        added += 1
    return {**template, "params": params}, added


# --- teacher-driven surface generation -------------------------------------------

EVOLVE_SYSTEM = (
    "You expand a task bank. Given a task and a few existing examples, you invent NEW "
    "example CONTENT along two axes: in-breadth (fresh domains and situations the task "
    "has not covered) and in-depth (longer, denser, more easily miscounted, more "
    "realistically messy). You never state the answer — only the raw material. Output a "
    "single JSON array and nothing else."
)


def _examples(template: dict, fields: Iterable[str], k: int = 3) -> list[dict]:
    """The first k existing instances, projected to the surface `fields`, as few-shot."""
    params = template.get("params", {})
    keys = [f for f in fields if f in params]
    rows = list(zip(*(params[k] for k in keys), strict=False))[:k]
    return [dict(zip(keys, r, strict=False)) for r in rows]


def build_prompt(template: dict, fields: tuple[str, ...], n: int) -> str:
    """The evolution instruction: what the task teaches, the surface fields to fill,
    the diversity/difficulty push, a few existing examples, and the output shape."""
    ex = _examples(template, fields, 3)
    return (
        f"TASK: {template.get('description', template.get('id', '')).strip()}\n\n"
        f"Invent {n} NEW examples. Each is a JSON object with exactly these fields: "
        f"{list(fields)}.\n"
        "- Make them genuinely varied: different domains, tones, lengths, and list "
        "sizes than the examples and than each other.\n"
        "- `name` is a short 1-3 word label for the list's subject (used to name files).\n"
        "- Put the tag on SOME but not all items, and vary how many — the point is that "
        "eyeballing the count is error-prone.\n"
        "- Do not compute or mention any count; only produce the list and the tag.\n\n"
        f"EXISTING EXAMPLES (do not repeat these):\n{json.dumps(ex, indent=2)}\n\n"
        f"Return ONLY a JSON array of {n} objects."
    )


def evolve_template(
    template: dict,
    generate: Callable[[str], str],
    *,
    n: int,
    rounds: int = 3,
    existing: set[str] | None = None,
) -> list[dict]:
    """Generate up to `n` evolved instances for `template`. `generate(prompt) -> reply`
    is the teacher call, injected so the loop is testable offline. Ground truth is
    recomputed by the registered deriver; candidates are deduped against `existing`
    (existing_instance_keys) and each other. Runs at most `rounds` batches — a teacher
    that keeps returning near-duplicates or junk terminates instead of looping."""
    tid = template["id"]
    if tid not in DERIVERS:
        raise KeyError(f"no evolve deriver registered for {tid!r}")
    fields, _ = DERIVERS[tid]
    have = set(existing or ())
    out: list[dict] = []
    for _ in range(rounds):
        if len(out) >= n:
            break
        candidates = parse_surface_candidates(
            generate(build_prompt(template, fields, n - len(out)))
        )
        for inst in derive_instances(tid, candidates):
            key = json.dumps({k: inst[k] for k in sorted(inst)}, sort_keys=True)
            if key in have:
                continue
            have.add(key)
            out.append(inst)
            if len(out) >= n:
                break
    return out
