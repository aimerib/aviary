"""Render a cleaned external corpus into interleave-ready training text.

The point of this module is that it does NOT format anything itself. It adapts
rows into `ConversationRecord` and hands them to `render_conversation`, the same
choke point aviary's own corpus goes through, so interleaved data is
byte-identical in chat template to the corpus it will be mixed with. A second
formatter — even a "obviously equivalent" one — would risk teaching the model
two templates, which is precisely what the choke-point rule exists to prevent.

Decisions worth knowing
-----------------------
* **Thoughts are lifted but not rendered.** The source's `<think>` is
  out-of-character GM planning (`SCENE:/CHARACTERS:/PLAN:`); aviary's is
  in-character interiority. Both are valid, but mixed under one tag with no
  disambiguating signal the model picks a mode at random. The reasoning is
  preserved on the record (so `--with-thoughts` remains possible) and simply not
  emitted by default.
* **lane="c".** `Lane` is a closed literal (a/b/c/d), so foreign data cannot
  declare its own. Character RP is nearest to self-play. The give-away is
  `family="external:<name>"`, and these records are never written into a run
  store — they live under `$AVIARY_DATA_DIR/external/`.
* **Rows without a system prompt are skipped.** In this corpus the character
  card is then embedded as the FIRST assistant turn, so training on it would
  teach the model to emit character cards instead of dialogue.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from aviary.io.jsonl import read_jsonl, write_jsonl
from aviary.paths import data_dir, external_dir
from aviary.render.serializer import ThoughtMode, render_conversation
from aviary.schema.records import ConversationRecord, Message, Provenance, SourceRef

LEADING_THINK = re.compile(r"\A\s*<think>\s*(.*?)\s*</think>\s*", re.S | re.I)

_ROLE = {"human": "user", "gpt": "assistant"}


def split_think(body: str) -> tuple[str | None, str]:
    """Lift a leading `<think>` block off an assistant turn -> (thought, rest)."""
    m = LEADING_THINK.match(body)
    if not m:
        return None, body
    return (m.group(1) or None), body[m.end() :]


def to_record(row: dict, *, name: str, run_id: str) -> ConversationRecord | None:
    """ShareGPT `{from,value}` -> ConversationRecord, or None if unusable."""
    turns = row.get("conversations") or []
    system = next((t.get("value") or "" for t in turns if t.get("from") == "system"), "")
    if not system.strip():
        return None  # card would be embedded as an assistant turn; see module docstring

    messages: list[Message] = []
    for turn in turns:
        role = _ROLE.get(turn.get("from") or "")
        if role is None:
            continue  # the system turn, already lifted
        body = turn.get("value") or ""
        if role == "assistant":
            thought, content = split_think(body)
            messages.append(Message(role="assistant", content=content, thought=thought))
        else:
            messages.append(Message(role="user", content=body))

    if not any(m.role == "assistant" for m in messages):
        return None
    rid = str(row.get("id") or "")
    return ConversationRecord(
        system=system,
        messages=messages,
        provenance=Provenance(
            record_id=rid,
            lane="c",
            run_id=run_id,
            family=f"external:{name}",
            source=SourceRef(kind="selfplay_seed", detail={"external": name, "row_id": rid}),
        ),
    )


def _approx_tokens(chars: int) -> int:
    """~4 chars/token. Approximate on purpose — used only for mix ratios."""
    return chars // 4


def _corpus_size(path: Path) -> tuple[int, int]:
    """(records, approx tokens) of a rendered jsonl artifact."""
    if not path.is_file():
        return 0, 0
    rows = chars = 0
    with path.open() as f:
        for line in f:
            rows += 1
            chars += len(json.loads(line).get("text", ""))
    return rows, _approx_tokens(chars)


def prepare(name: str, *, with_thoughts: bool = False, against_run: str | None = None) -> dict:
    """Render `external/<name>/cleaned.jsonl` into `external/<name>/rendered/`."""
    root = external_dir(name)
    src = root / "cleaned.jsonl"
    if not src.is_file():
        raise FileNotFoundError(f"{src} not found — run the cleaner for {name!r} first")

    run_id = f"external-{name}"
    skipped: Counter[str] = Counter()
    records: list[ConversationRecord] = []
    total = 0
    with src.open() as f:
        for line in f:
            total += 1
            rec = to_record(json.loads(line), name=name, run_id=run_id)
            if rec is None:
                skipped["no_system_prompt_or_no_assistant_turn"] += 1
            else:
                records.append(rec)

    mode = ThoughtMode.WITH if with_thoughts else ThoughtMode.WITHOUT
    samples = [render_conversation(rec, mode) for rec in records]
    out_dir = root / "rendered"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / f"train_{mode.value}.jsonl"
    write_jsonl(artifact, samples)

    chars = sum(len(s.text) for s in samples)
    report = {
        "source_rows": total,
        "skipped": dict(skipped),
        "rendered": len(samples),
        "thought_mode": mode.value,
        "approx_tokens": _approx_tokens(chars),
        "artifact": str(artifact),
    }

    if against_run:
        base = data_dir() / against_run / "rendered" / "train_no_thoughts.jsonl"
        b_rows, b_tokens = _corpus_size(base)
        ext_tokens = report["approx_tokens"]
        denom = b_tokens + ext_tokens
        report["mix_against"] = {
            "run": against_run,
            "aviary_records": b_rows,
            "aviary_approx_tokens": b_tokens,
            "external_records": len(samples),
            "external_approx_tokens": ext_tokens,
            "aviary_token_share": round(b_tokens / denom, 4) if denom else None,
            "external_token_share": round(ext_tokens / denom, 4) if denom else None,
        }
    (out_dir / "interleave_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


__all__ = ["prepare", "split_think", "to_record", "read_jsonl"]
