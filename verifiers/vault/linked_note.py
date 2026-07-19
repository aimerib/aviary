"""Verifier for vault.linked_note: outcome gate on the FINAL vault state.

Passes iff both files end in successful writes — the note at `destination`
opening with a YAML frontmatter block ('---' fences) that mentions tags, and
the hub at `hub` whose FINAL content contains a wikilink to the note's stem
([[bird-logger]] and [[notes/bird-logger]] both count; [[bird-logger|alias]]
too) — and the final assistant turn confirms. Path is irrelevant (outcome-gate
rule): hub-first, note-first, or fix-it-after-a-read-back all pass.

Whether the note's prose is any good is judge territory; this gates the vault
conventions (frontmatter, non-orphaned note) being structurally respected.
"""

import json
import re
from pathlib import PurePosixPath

from aviary.schema.records import ConversationRecord
from aviary.schema.results import VerifierResult

VERIFIER_ID = "vault/linked_note"

_FRONTMATTER = re.compile(r"^---\s*\n.*?\n---\s*(\n|$)", re.DOTALL)


def _is_destination(path: str, destination: str) -> bool:
    """Outcome-equivalent path identity (see files/save_note)."""
    d = PurePosixPath(destination).parts
    p = PurePosixPath(path).parts
    return bool(d) and len(p) >= len(d) and p[-len(d) :] == d


def _final_write(rec: ConversationRecord, target: str | None) -> tuple[bool, str]:
    ok, content_out = False, ""
    calls: dict[str, tuple[str, str]] = {}
    for m in rec.messages:
        if m.role == "assistant":
            for tc in m.tool_calls:
                if tc.name == "write_file" and "path" in tc.arguments:
                    calls[tc.id] = (
                        str(tc.arguments["path"]),
                        str(tc.arguments.get("content", "")),
                    )
        elif m.role == "tool" and m.tool_call_id in calls:
            path, content = calls[m.tool_call_id]
            if not (target and _is_destination(path, target)):
                continue
            try:
                result = json.loads(m.content)
            except json.JSONDecodeError:
                result = None
            ok = isinstance(result, dict) and not result.get("error")
            content_out = content
    return ok, content_out


def verify(rec: ConversationRecord) -> VerifierResult:
    params = rec.provenance.source.detail.get("params", {})
    destination = params.get("destination")
    hub = params.get("hub")

    note_ok, note_content = _final_write(rec, destination)
    hub_ok, hub_content = _final_write(rec, hub)

    wrote_note = bool(destination) and note_ok
    fm = _FRONTMATTER.match(note_content)
    frontmatter_tags = bool(fm) and "tags" in fm.group(0)

    wrote_hub = bool(hub) and hub_ok
    # Wikilink to the note's stem: [[stem]], [[notes/stem]], [[stem|alias]] ...
    linked = False
    if destination:
        stem = PurePosixPath(destination).stem
        linked = bool(re.search(r"\[\[[^\]]*" + re.escape(stem) + r"(\|[^\]]*)?\]\]", hub_content))

    last = rec.messages[-1]
    confirmed = last.role == "assistant" and not last.tool_calls and bool(last.content.strip())

    checks = {
        "wrote_note": wrote_note,
        "note_has_frontmatter_tags": frontmatter_tags,
        "wrote_hub": wrote_hub,
        "hub_links_note": linked,
        "confirmed_to_user": confirmed,
    }
    return VerifierResult(
        passed=all(checks.values()),
        verifier_id=VERIFIER_ID,
        details={k: v for k, v in checks.items() if not v},
    )
