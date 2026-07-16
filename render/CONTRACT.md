# Serializer contract — v2

The byte format emitted by `src/aviary/render/serializer.py`, the only code path
allowed to produce training-format text. `render/goldens/` locks it; goldens change
only with an explicit contract version bump decided by a human, recorded here.

## v2 (2026-07-16) — initial implemented contract

Supersedes the never-implemented v1 sketch. Human-approved as part of the
three-lane expansion plan. Additions over the v1 sketch: inner thoughts,
group-chat speaker prefixes, next-speaker-prediction samples, DPO pairs.

### Turn framing (ChatML, Qwen-style)

```
<|im_start|>{role}\n{body}<|im_end|>\n
```

Roles: `system`, `user`, `assistant`, `tool`. The record's `system` field renders
as the first turn; a `system` role inside `messages` is an error.

### Tools (Hermes format)

When `tools_schema_json` is set, the system turn appends the fixed `# Tools`
preamble with the schema inside `<tools>…</tools>` (see `TOOLS_PREAMBLE`).
Assistant tool calls render as, in order, after the prose body:

```
<tool_call>\n{"name": …, "arguments": …}\n</tool_call>
```

JSON policy: `json.dumps(..., ensure_ascii=False, separators=(", ", ": "))`,
argument key order preserved as recorded (`canonical_tool_call_json`).
Tool results render as `tool`-role turns wrapping content in
`<tool_response>\n…\n</tool_response>`.

### Inner thoughts

`ThoughtMode.WITH`: an assistant message's `thought` renders as
`<think>\n{thought}\n</think>` as the first block of the assistant body
(SillyTavern reasoning-parses this; GRPO can strip it).
`ThoughtMode.WITHOUT`: `thought` is omitted entirely. Both variants are rendered
from the same record; they are separate artifacts.

### Group chat

When a record has >1 distinct assistant `speaker`, every assistant body is
prefixed `"{speaker}: "`. Single-speaker records get no prefix.

Next-speaker-prediction samples (multi-speaker records only): conversation
prefix + `<|im_start|>assistant\n{speaker}:` with the train span covering only
`{speaker}:`.

### Train spans

Character offsets `[start, end)` over the rendered text covering each assistant
body from just after `<|im_start|>assistant\n` through the closing `<|im_end|>`
(exclusive of the trailing newline). Loss is computed only inside train spans.

### DPO pairs

`prompt_text` = render of system + messages through the first user turn.
`chosen_text` / `rejected_text` = the remainder of each sibling's full render.
Pairs require: same `sibling_group`, both verifier-passed, chosen judge-passed,
rejected judge-dropped, judge-overall margin ≥ configured minimum, and
byte-identical prompt prefixes.

## Changing this contract

1. A human decides and records the bump here (v3, date, rationale).
2. Update the serializer, regenerate goldens, review the golden diff line by line.
3. `serializer_contract` in run manifests must match; mixed-contract corpora are
   forbidden.
