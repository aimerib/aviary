# Serializer contract — v3

The byte format emitted by `src/aviary/render/serializer.py`, the only code path
allowed to produce training-format text. `render/goldens/` locks it; goldens change
only with an explicit contract version bump decided by a human, recorded here.

**The authority for this format is not this document.** It is the `chat_template`
in `Qwen/Qwen3.5-35B-A3B`'s `tokenizer_config.json`, vendored at
`tests/fixtures/chat_template/qwen3_5.jinja`. `tests/test_template_parity.py`
renders every fixture record through that jinja and asserts byte-equality with our
output. A golden can be regenerated from wrong code and still look self-consistent;
the parity test is what makes it *right*. Keep both.

## v3 (2026-07-22) — Qwen3.5-native chat template

Human-approved. v2 emitted the Qwen3/Hermes format — correct for the previous
generation, wrong for the base model we actually train. Six divergences from the
base model's own template, each of which is train/inference skew: at deployment the
runtime applies the stock template, so anything we render differently is a shape
the model was trained on but will never see.

1. **Tool calls were JSON, now XML.** The headline bug: a model trained to emit
   `<tool_call>{"name": …}</tool_call>` produces calls the Qwen3.5 tool parser
   will not parse. Lane A's entire purpose.
2. **Tool results were `tool`-role turns, now grouped `user` turns.** Consecutive
   results collapse into ONE user turn (842 of them in the 2026-07-22 burn).
3. **Tools block was appended after the system text, now precedes it.**
4. **Tools were one JSON array, now one JSON object per line**, each wrapped in the
   OpenAI `{"type": "function", "function": {…}}` envelope — see below.
5. **`</think>` was followed by one newline, now two.**
6. **Reasoning-off was "no block", now the empty block** `<think>\n\n</think>\n\n`,
   which is exactly what `enable_thinking=false` primes at inference.

Message content is trimmed, as the template trims it.

### Turn framing (ChatML, Qwen-style)

```
<|im_start|>{role}\n{body}<|im_end|>\n
```

Roles as emitted: `system`, `user`, `assistant`. The record's `system` field renders
as the first turn; a `system` role inside `messages` is an error. A record with no
system text and no tools emits no system turn. `tool`-role messages exist in the
schema but never render as a `tool` turn — see Tools.

### Tools (Qwen3.5 XML)

When `tools_schema_json` is set, the system turn opens with the fixed `# Tools`
preamble (`TOOLS_HEADER` … `TOOLS_FOOTER`), one tool per line, and the record's own
system text is appended after a blank line — in that order.

Each tool is wrapped as `{"type": "function", "function": {…}}` unless already
wrapped. **This is the one judgment call in v3**: the template dumps whatever the
caller passes in `tools`, and every OpenAI-compatible server (vLLM, SGLang,
llama.cpp) forwards the request's `tools` array verbatim, which is OpenAI-shaped.
Serving with bare function objects instead would make this line diverge.

Assistant tool calls render after the prose body:

```
<tool_call>\n<function={name}>\n<parameter={key}>\n{value}\n</parameter>\n</function>\n</tool_call>
```

Separator before the first call: `\n\n` if the body is non-empty, nothing if empty
(802 such turns in the burn); `\n` before each subsequent call. Argument key order
is preserved as recorded. Argument values follow the template's rule — `json.dumps`
for mappings and non-string sequences, Python `str()` otherwise (so bools render
`True`/`False`, matching Jinja's `string` filter, deliberately).

Tool results render as a `user` turn wrapping each result in
`<tool_response>\n…\n</tool_response>`, with consecutive results grouped into one
turn.

### Inner thoughts

An assistant turn carries a think block only where the template's
`last_query_index` rule puts one: after the last real user query. Earlier assistant
turns render bare, because that is what the model sees in its own context at
inference.

`ThoughtMode.WITH`: `<think>\n{thought}\n</think>\n\n` at the head of the body,
empty when the message has no thought. `ThoughtMode.WITHOUT`: the same block with
empty contents. Both variants are rendered from the same record; they are separate
artifacts.

**Records with no user turn at all** — lane B group-chat scenes — are a
training-only shape the stock template refuses outright (`No user query found in
messages`). There every assistant turn keeps its thought: the interiority is the
lane's payload and there is no inference-time context shape to be faithful to.
Measured on the 2026-07-22 burn, this rule costs nothing elsewhere: all 2,517 lane A
thoughts already fall after that lane's single opening user turn, and lane C
generates reasoning-off.

### Group chat

When a record has >1 distinct assistant `speaker`, every assistant body is
prefixed `"{speaker}: "` (after the think block). Single-speaker records get no
prefix.

Next-speaker-prediction samples (multi-speaker records only): conversation
prefix + `<|im_start|>assistant\n`, plus the empty think block wherever the main
artifact would carry one, plus `{speaker}:` — with the train span covering only
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

## v2 (2026-07-16) — superseded by v3

Historical record; do not render to it. Superseded the never-implemented v1 sketch. Human-approved as part of the
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
