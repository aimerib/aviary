# Lane D timing: initiation data — plumbing now, rendering later

Status: **design only**. The timestamp plumbing (source → adapter → `Message.ts`
→ silently dropped by the v2 serializer) is implemented and fixture-exercised;
nothing in this document is. Serializer format changes are a human decision
(golden-file rule); the choke point and `render/goldens/` are untouched by the
Sorcha prep. This doc exists so that decision can be made concretely, not from
scratch, when Sorcha's corpus assembly starts.

Why timing matters at all: a companion model that only ever *replies* cannot be
present in the way lane D's source material shows real presence — noticing a
lull, picking a thread back up, knowing when a conversation is over. That
behavior needs (1) training text that encodes silence and elapsed time, and
(2) self-play that generates initiation scenarios on purpose. Part (a) proposes
the format; part (b) proposes the generation.

---

## (a) Contract v3 proposal: time-gap conditioning + silence samples

### What v2 already gives us

- NSP samples (multi-speaker records): conversation prefix +
  `<|im_start|>assistant\n{speaker}:` with the train span covering only
  `{speaker}:` — the model learns *who* speaks next.
- Train spans: loss only inside assistant bodies.

v3 extends NSP from "who speaks next?" to "**what happens next?**" — where the
answer may be *nobody, for six hours*.

### Gap annotations (conditioning, never target)

For records whose messages carry `ts`, a bucketed gap marker renders **on the
prompt side** before a turn that follows silence ≥ 10 minutes:

```
<|im_start|>user
[t+3h] Rio: out. that was brutal but I think question 3 saved me<|im_end|>
```

- Buckets, not raw durations: `[t+2m]`, `[t+40m]`, `[t+3h]`, `[t+2d]` (minute /
  ten-minute / hour / day resolution by magnitude). Small vocabulary, no
  overfitting to exact clock arithmetic.
- Gap markers are **never inside a train span**. The model is conditioned on
  elapsed time; it is never asked to *generate* clock strings.
- Records without `ts` (lanes A/B/C today) render **byte-identically to v2** —
  gap logic keys on the presence of timestamps, not on lane. This is the load-
  bearing property: **every existing golden survives v3 unchanged.**

### Silence samples (the "when not to speak" target)

v2 NSP asks the model to emit `{speaker}:`. v3 adds a third possible
continuation after a gap annotation: a literal silence marker.

```
…prefix…<|im_start|>user
[t+30s] Rio: night. for real this time<|im_end|>
<|im_start|>assistant
<silence><|im_end|>
```

Train span covers only `<silence>`. Sampling policy (renderer, deterministic
seed): silence samples are cut at points where the source stream shows the
modeled speaker genuinely *not* responding before a long gap — the supervision
is real behavior, not synthesized restraint. Ratio capped (proposal: ≤1 silence
sample per record) so the corpus doesn't teach muteness.

### Initiation samples

The dual of silence: where the source shows the modeled speaker breaking a long
gap, an ordinary assistant-body sample is cut *after* the gap annotation — the
model practices opening, conditioned on elapsed time and dangling context:

```
…prefix…<|im_start|>assistant
[gap annotation on prompt side]  ← conditioning
Sam: how was the dinner thing in the end?<|im_end|>   ← train span
```

### Golden/train-span impact summary

| Artifact | v3 impact |
|---|---|
| Existing goldens (plain_chat, single_tool, recovered_error, scene_thoughts, nsp_0-2, dpo_pair) | **byte-identical** (no `ts` → no gap logic) |
| New goldens to add at v3 | `timed_stream.with_thoughts`, `timed_stream.nsp_gap`, `timed_stream.silence`, `timed_stream.initiation` |
| Train spans | unchanged for normal turns; `<silence>` spans cover the marker only; gap markers never in-span |
| `SERIALIZER_CONTRACT` | "v2" → "v3", human-approved, recorded in `render/CONTRACT.md` |

Open questions for the v3 decision (flagged, not resolved here): whether
`<silence>` is a plain string or a reserved token added to the tokenizer at
SFT time; whether gap buckets belong inside the user body (as sketched) or as a
separate meta-turn; ST/GRPO parser compatibility for both.

---

## (b) Lane C under the sorcha target: generating initiation

Lane D supplies *organic* timing behavior; it is finite and one household's
distribution. Sorcha-persona self-play (post-v2.2, post-co-authoring) densifies
it with *targeted* scenarios the organic data undersamples. Design only; nothing
here runs until Sorcha's persona exists.

- **Clock-driven seeds.** Lane C seeds gain an optional scenario clock: the
  driver advances simulated time between turns (`[t+4h]` injected into the
  character side's context, mirroring v3 rendering). Seed families: `lull`
  (conversation trails off — does she let it?), `re-engage` (dangling thread +
  hours of silence — does she pick it up, and does she recall the thread
  correctly?), `dont-speak` (user said goodnight / said "busy, later" — the
  correct continuation is `<silence>`), `morning` (cold-open check-in from
  standing context).
- **User-sim personas** gain timing knobs: reply-latency distribution,
  goes-quiet-without-warning probability, receptiveness to re-engagement
  (welcoming vs terse). The sim must sometimes *not* reward initiation — a
  companion that's always rewarded for pinging learns to nag.
- **Judging**: a `timing` axis set under the sorcha target's rubrics
  (co-authored, like everything Sorcha): initiation appropriateness, thread
  recall fidelity, restraint quality. The existing RP-transparency machinery is
  untouched — these are persona-voiced records (speaker == Sorcha), harmonizable
  and judged on the persona rubric by the same voice-gating that handles Olivia
  simple-chats today.
- **Verifier sketch**: structural checks stay outcome-only — a `dont-speak` seed
  whose transcript contains a character turn after the goodbye fails; a
  `re-engage` seed that never re-engages within the scenario window fails.

The two halves meet in the corpus: lane D provides the prior (how this one
household actually paces), lane C under sorcha provides coverage (the long tail
of timing situations, at controllable density), and both render through the same
v3 format so the model sees one consistent language of time.
