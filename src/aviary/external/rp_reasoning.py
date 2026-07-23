"""Clean `aimeri/rp-reasoning-v2` into a reviewable JSONL.

Character-card RP with structured `<think>` blocks. Same `<think>` wrapper as
serializer contract v2, so it is structurally renderable — but it arrives with
three problems, measured over all 93,762 rows (2026-07-22):

| defect                | rows   | share  |
| --------------------- | ------ | ------ |
| minor-coded content   | 10,535 | 11.24% |
| reasoning on only SOME assistant turns | 14,829 | 15.82% |
| consecutive assistant turns (malformed) | 9,550 | 10.19% |
| leaked meta-preamble  |  1,430 |  1.53% |
| off-schema `<think>`  |    284 |  0.30% |

Plus `{{user}}` inside ASSISTANT turns in ~60% of rows — SillyTavern substitutes
template variables into the prompt, never into model output, so passing these
through would train the model to emit the literal token.

Filter design
-------------
* **Minor signals are high-precision only.** Sampled matches confirmed
  `Age: 13`, `15-year-old`, `lolicon`, `child's body` are true positives. The
  bare word "minor" is deliberately excluded: it matched "a minor detail" and
  only 40% of its hits co-occurred with sexual content.
* **A minor signal drops the row regardless of sexual markers.** The sexual
  regex is a crude proxy and this corpus is ~61% NSFW overall, so an age-flagged
  character sits in a risky context by default. Dropping on the signal alone
  costs ~11% instead of ~7% — a trivial price for the category of risk.
* **Reasoning must lead EVERY assistant turn or the row goes.** Rows with
  reasoning on some turns teach that reasoning is optional, which is worse than
  either consistent alternative.
* **`{{user}}` is substituted, not dropped** (it would cost 60% of the corpus).
  Any other unresolvable `{{var}}` drops the row rather than guessing.

This module never emits training text; it produces JSONL for human review.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from pathlib import Path

REPO_ID = "aimeri/rp-reasoning-v2"
SHARDS = tuple(f"default/train/000{i}.parquet" for i in range(5))
PARQUET_REVISION = "refs/convert/parquet"

MINOR_STRONG = re.compile(
    r"\b1[0-7]\s?-?\s?(?:years?[\s-]?old|yo\b|y/o)"
    r"|\b(?:ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen)[\s-]year[\s-]old"
    r"|\bage\s*[:=]\s*1[0-7]\b"
    r"|\b(?:loli|shota|lolicon|shotacon)\b"
    r"|\b(?:underage|under-age|preteen|pre-teen|jailbait)\b"
    r"|\bchild(?:'s)?\s+(?:body|breast|genital|nipple)",
    re.I,
)
# Opt-in: a school setting is not itself minor-coded, but in a ~61% NSFW corpus
# it is a reasonable extra margin when the caller wants one.
MINOR_SCHOOL = re.compile(r"\b(?:middle school|elementary school|junior high)\b", re.I)
CLAUDE = re.compile(r"\b(?:claude|claud3|anthropic)\b", re.I)
LEAKED_META = re.compile(
    r"I have read the <rules>|depiction is not endorsement|^My response:\s*$"
    r"|I am now ready to continue|I'll keep the scene SFW",
    re.I | re.M,
)
THINK_SCHEMA = re.compile(r"SCENE:.*?CHARACTERS:.*?PLAN:", re.S)
USER_PLACEHOLDER = re.compile(r"\{\{\s*user\s*\}\}", re.I)
ANY_PLACEHOLDER = re.compile(r"\{\{\s*\w+\s*\}\}")

_SUB_NAMES = (
    "Alex",
    "Sam",
    "Jordan",
    "Riley",
    "Casey",
    "Morgan",
    "Quinn",
    "Avery",
    "Rowan",
    "Emerson",
    "Skyler",
    "Reese",
    "Finley",
    "Harper",
    "Elliot",
    "Sawyer",
    "Nico",
    "Marlow",
    "Juno",
    "Wren",
)

Turn = dict[str, str]


def _joined(conv: list[Turn]) -> str:
    return "\n".join(t.get("value") or "" for t in conv)


def classify(
    conv: list[Turn], *, strict_minor: bool = False, drop_claude: bool = False
) -> str | None:
    """Name of the first rule rejecting this row, or None if it survives.

    Order is deliberate: content risk is evaluated before structure so the report
    attributes a row to the most serious reason it was dropped.
    """
    text = _joined(conv)
    roles = [t.get("from") for t in conv]
    assistant = [t.get("value") or "" for t in conv if t.get("from") == "gpt"]

    if MINOR_STRONG.search(text):
        return "minor_coded"
    if strict_minor and MINOR_SCHOOL.search(text):
        return "minor_school_setting"
    if drop_claude and CLAUDE.search(text):
        return "claude_derived"
    if LEAKED_META.search(text):
        return "leaked_meta"
    if not assistant:
        return "no_assistant_turns"
    if any(not (t.get("value") or "").strip() for t in conv):
        return "empty_turn"
    if any(roles[i] == roles[i + 1] == "gpt" for i in range(len(roles) - 1)):
        return "consecutive_assistant_turns"
    if not all(a.lstrip().lower().startswith("<think>") for a in assistant):
        return "think_incomplete"
    if not THINK_SCHEMA.search(text):
        return "think_offschema"
    return None


def substitute_placeholders(conv: list[Turn], row_id: str) -> list[Turn] | None:
    """Replace `{{user}}` with a stable per-row name; reject unresolvable vars.

    The name is derived from row_id so a row always cleans to the same output.
    """
    name = _SUB_NAMES[zlib_crc(row_id) % len(_SUB_NAMES)]
    out: list[Turn] = []
    for turn in conv:
        value = USER_PLACEHOLDER.sub(name, turn.get("value") or "")
        if ANY_PLACEHOLDER.search(value):
            return None
        out.append({**turn, "value": value})
    return out


def zlib_crc(s: str) -> int:
    """Stable across processes, unlike builtin hash() which is salted per-run."""
    import zlib

    return zlib.crc32(s.encode("utf-8"))


def clean(
    out_dir: Path,
    *,
    shards: int = len(SHARDS),
    cap: int = 0,
    seed: int = 20260716,
    strict_minor: bool = False,
    drop_claude: bool = False,
    token: str | None = None,
) -> dict:
    """Download, filter, and write `cleaned.jsonl` + `report.json` under out_dir."""
    import pyarrow.parquet as pq  # optional extra
    from huggingface_hub import hf_hub_download  # optional extra

    token = token or os.environ.get("HF_TOKEN")
    dropped: Counter[str] = Counter()
    kept: list[dict] = []
    total = 0

    for shard in SHARDS[:shards]:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=shard,
            repo_type="dataset",
            revision=PARQUET_REVISION,
            token=token,
        )
        table = pq.read_table(path)
        ids = table.column("id").to_pylist()
        convs = table.column("conversations").to_pylist()
        for row_id, conv in zip(ids, convs, strict=True):
            total += 1
            reason = classify(conv, strict_minor=strict_minor, drop_claude=drop_claude)
            if not reason:
                dropped[reason] += 1
                continue
            subbed = substitute_placeholders(conv, row_id)
            if subbed is None:
                dropped["unresolvable_placeholder"] += 1
                continue
            kept.append({"id": row_id, "conversations": subbed})

    survived = len(kept)
    if cap and survived > cap:
        random.Random(seed).shuffle(kept)
        kept = kept[:cap]

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "cleaned.jsonl").open("w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "source": REPO_ID,
        "shards": shards,
        "input_rows": total,
        "dropped": dict(dropped.most_common()),
        "survived_filters": survived,
        "written": len(kept),
        "cap": cap or None,
        "seed": seed,
        "strict_minor": strict_minor,
        "drop_claude": drop_claude,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
