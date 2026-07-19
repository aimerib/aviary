# aviary

Where you raise a parrot. This repo generates the bootstrap SFT corpus for
**Swift Parrot** (`aimeri/spoomplesmaxx-flash-35B-A3`, Qwen3.5-35B-A3B base) by
running three data lanes through shared gates and one serializer, rendering
survivors into Swift Parrot's exact chat template. The same task bank +
verifiers later become the GRPO reward suite (Atropos).

Pipeline:

```
lane A: tasks ── hermes batch_runner ─────────────────────────┐
lane B: books ── extract profiles/scenes/dialogue+thoughts ───┼─ verify → judge → scrub/dedupe → harmonize → render → ship
lane C: seeds ── user-sim ⇄ character self-play ──────────────┘
```

Three lanes, one spine. Every lane's adapter normalizes into ONE record schema
(`src/aviary/schema/records.py`); everything downstream is lane-agnostic.

- **Lane A (agentic)** — task templates rolled out in-character as Olivia via
  hermes-agent batch datagen. Tool-use competence.
- **Lane B (extraction)** — CoSER-style decomposition of organic fiction (local
  ebooks, AO3, Gutenberg) into character profiles, scene setups, and dialogue
  **with inner thoughts**. The only non-LLM prose-entropy source; the ToM data.
- **Lane C (self-play)** — user-simulator ⇄ character multi-turn RP targeting
  SillyTavern dynamics: short/lazy/typo'd user turns (deterministic seeded typo
  injector), response-length control, impersonation avoidance.
- **Lane D (personal)** — the owner's private streams (chat exports, journals)
  normalized straight into records by `lanes/d_personal/` — no teacher calls;
  human-origin records judged by the primary judge (cross-vendor is vacuous).
  Sorcha builds only; run data ships NOWHERE (manifest records the exemption).
- Magpie-style null-prompt extraction: deliberately out of scope.

Volumes are a funnel by design; rejection is a feature. Verify-rejected rollouts
feed difficulty tuning. **Judge-rejected, verifier-passed siblings additionally
feed the DPO artifact** (chosen = kept sibling, rejected = judge-failed sibling,
margin-gated) — see `src/aviary/render/dpo.py`. Scope: DPO pairs are **lane A
only** (only lane A emits multiple rollout siblings of one instance sharing a
byte-identical prompt — `sibling_group`; lanes B/C leave it unset) and are
emitted **with-thoughts only** (`ThoughtMode.WITH`), unlike the SFT path which
emits both thought modes.

## Build targets

`datagen/configs/targets/<name>.yaml` binds everything persona- or model-specific:
the persona attached at lane A ingest and generated-as in lanes A/C, the
harmonizer's canonical voice + paraphrase prompt, the judge's voice rubric,
per-lane harmonize policy, and which lanes a render may include. Selected via
`$AVIARY_TARGET`; the default `flash-v2_2` (persona: Olivia) reproduces the
pre-target pipeline byte-for-byte. `sorcha-v1` (persona: Sorcha, lanes b+d) is a
private single-user companion trained from the same base — **Olivia does not
exist in Sorcha's corpus** and renders are target-bounded to enforce it. Sorcha's
persona content is CO-AUTHORED with the owner in a working session — the files in
`datagen/persona/sorcha/` are placeholders until then; never invent it.

## Vocabulary

- **trajectory / record** — one complete conversation: messages, tool calls,
  tool results, inner thoughts, provenance. In-repo type: `ConversationRecord`.
- **template vs instance** — a template is a parameterized task family in
  `tasks/`; an instance is one concrete parameterization.
- **family** — the split unit: task family (lane A), work_id i.e. whole book
  (lane B), scenario family (lane C). Lane-C records seeded from lane-B works
  INHERIT the work's family so holdout can't leak across lanes.
- **teacher** — a model generating data. Mixed roster (DeepSeek/Kimi/GLM) so no
  single idiolect dominates; **no Claude-class model anywhere** (ToS risk,
  decided 2026-07-16). Swift Parrot never generates in this repo.
- **choke point** — the single serializer emitting training-format text:
  `src/aviary/render/serializer.py`, locked by `render/goldens/`.

## Non-negotiable contracts

Violating any of these silently poisons the corpus. When a contract blocks a
task you were asked to do, stop and say so instead of working around it.

1. **Choke-point rule.** Exactly one code path
   (`src/aviary/render/serializer.py`) converts records into Swift Parrot
   ChatML + Hermes-format `<tool_call>` text (byte format: `render/CONTRACT.md`,
   contract **v2**: `<think>` blocks, group-chat speaker prefixes, NSP samples,
   DPO pairs). Byte-identical across SFT, eval, deployment, GRPO. No inline
   formatting anywhere else — not in tests, not in scripts, not "just for
   debugging output that got committed."

2. **Golden-file rule.** `render/goldens/` locks serializer output. Goldens
   change only with an explicit contract version bump decided by a human,
   recorded in `render/CONTRACT.md`. Never edit a golden to make a test pass.
   If output and golden disagree, the default assumption is the code is wrong.

3. **Span-protection rule.** The harmonizer (`src/aviary/gates/harmonize.py`)
   may rewrite conversational spans only — structurally, that is
   `Message.content`/`Message.thought` on user/assistant turns, nothing else,
   and per policy only **persona-voiced turns** of the build target: lane A (the
   persona by construction), and lane C **only where the character IS the
   persona** (all assistant turns spoken by the target's `persona_speaker`).
   The persona is RP-transparent: lane B character voices **and lane-C
   character-RP voices** (lane-B-seeded, "You play X and only X") are the
   entropy source and are never paraphrased — the assistant persona must never
   leak into a roleplay character. Immutable: tool call arguments, tool results,
   system prompts, tool schemas, and (via mask/restore placeholders) code,
   paths, quoted strings, identifiers, numbers inside prose. If restore isn't
   byte-perfect, drop the record — never bend the span boundary. (Scar tissue:
   Gemma-era paraphraser leakage.)

4. **Outcome-gate rule.** Verifiers judge final state, never the path taken.
   Error-then-recovery trajectories are keepers — prime data. Never fix a red
   verifier by loosening it; read the trajectory first and decide whether the
   verifier or the trajectory is wrong.

5. **Fixture rule.** Every verifier ships fixtures in `verifiers/fixtures/<id>/`
   covering clean pass, clean fail, and recovered-failure pass. Enforced
   mechanically by `tests/test_verifier_fixtures.py` — a verifier without
   correct fixtures does not merge.

6. **Split rule.** Holdout is family-level (`holdout: true` in task YAML /
   lane_b.yaml books / lane_c.yaml seeds). The renderer hard-fails — writes
   nothing — if a holdout family appears in a train render
   (`HoldoutViolation`). Eval also gets unseen parameterizations of seen
   templates, selected at render time with a manifest-recorded seed.

7. **Data-in-git rule.** No trajectories, session records, book texts, or
   rendered corpora in git, ever. Lane D is stricter (radioactive): personal
   data never appears in git, tests, fixtures, goldens, or docs — not even
   single-message samples; every lane D test runs on synthetic data. Run data lives under `$AVIARY_DATA_DIR/<run_id>/`
   and ships to a private HF dataset repo referenced by its manifest. `runs/`
   holds manifests only. Exception: fixtures and goldens are small, curated,
   and tracked on purpose — they are the contract.

8. **Provenance rule.** Every pilot/burn writes a manifest (v2, see
   `runs/TEMPLATE.manifest.yaml`): hermes pin, dated teacher snapshot IDs
   (never `latest` — enforced by `Roster`), task-bank commit, datagen config
   hash, PromptSet hash, counts, keep rates, spend by teacher and lane.

9. **Source-of-truth rule.** Lane A's serializer input is the raw hermes batch
   record (`trajectories.jsonl`: `conversations` + `toolsets_used`/`tool_stats`
   metadata), normalized by `lanes/a_agentic/ingest.py`. Stripped ShareGPT
   views are a human-readable preview, never an input. hermes does NOT persist
   the ephemeral system prompt — the frozen Olivia prompt is re-attached at
   ingest.

10. **Cache-discipline rule.** System prompts and tool schemas are byte-stable
    across a run: the `PromptSet` is frozen and hashed at run start, recorded
    in the manifest, and asserted before every LLM stage. A mid-run prompt edit
    changes the hash and the run refuses to continue — treat it as a new run.
    (~5x on the API bill.)

11. **Judge-independence rule.** Judging is cross-vendor: a record is never
    judged by the vendor that generated it (`Roster.judge_for`). Judge text
    never enters the corpus.

## Map

| Path                  | What lives here                                          |
| --------------------- | -------------------------------------------------------- |
| `src/aviary/`         | All engine code (installable package; `aviary` CLI)      |
| `src/aviary/schema/`  | The unified record schema + manifest models              |
| `src/aviary/lanes/`   | a_agentic (hermes), b_fiction (extraction), c_selfplay, d_personal |
| `src/aviary/gates/`   | verify/judge/scrub/dedupe/spans/harmonize + pipeline     |
| `src/aviary/render/`  | THE serializer, split, DPO pairing, render orchestration |
| `tasks/<family>/`     | Lane A task templates (YAML; schema: `tasks/TEMPLATE.task.yaml`) |
| `verifiers/`          | Verifier plugins (pure functions) + `fixtures/`; laneb/, lanec/ structural verifiers |
| `datagen/configs/`    | teachers roster, pricing, pilot/burn, lane configs; `targets/` build targets |
| `datagen/toolsets/`   | Per-family tool allowlists + JSON schemas                |
| `datagen/persona/`    | per-persona dirs (`olivia/` canonical + voice rubric + paraphrase prompt; `sorcha/` placeholders); `user_sims/` |
| `datagen/prompts/`    | Lane B extraction + lane C user-sim prompt templates (part of the hashed PromptSet) |
| `gates/judge/`        | Rubric YAMLs + judge prompt                              |
| `gates/scrub/`        | Denylist + PII patterns                                  |
| `render/`             | `CONTRACT.md` (byte format, versioned) + `goldens/`      |
| `runs/`               | Manifests only                                           |
| `tests/`              | Pytest; offline only (socket-ban fixture), fixtures in `tests/fixtures/` |

## Workflow

`just` recipes (implemented; names are contract — don't rename):

- `just install`      — verify pinned hermes checkout (`$AVIARY_HERMES_DIR`) + its batch_runner interface
- `just pilot`        — calibration run across lanes enabled in `pilot.yaml`; writes a manifest
- `just burn`         — full run; refuses without a recent healthy pilot manifest
- `just gate <run>`   — verify → judge → scrub/dedupe → harmonize
- `just render <run>` — choke-point serialization + split; hard-fails on split-rule violations
- `just stats <run>`  — keep rates, pass rates per template, spend, difficulty-band violations
- `just test` / `just lint`

Task-design invariant: templates target a 30–80% teacher pass rate. Outside
that band, revise the template (difficulty), don't touch the verifier.

## Stack and conventions

- Python 3.12, `uv`, `ruff`, `pytest`. Runtime deps: pydantic, pyyaml, httpx —
  keep it that way.
- Verifiers and the serializer are pure functions over data — no network, no
  model calls inside them. Tests are fully offline (autouse socket ban);
  LLM-dependent stages test against `FakeTeacherClient` + recorded fixtures.
- Teacher API: one httpx client, OpenAI-compatible wire. DeepSeek + GLM direct,
  Kimi via OpenRouter. Keys: `DEEPSEEK_API_KEY`, `GLM_API_KEY`/`ZAI_API_KEY`,
  `OPENROUTER_API_KEY`. Response cache under `$AVIARY_DATA_DIR/cache/` doubles
  as the test-fixture format.
- hermes-agent is a **pinned external dependency we call** (tag in
  `teachers.yaml`) — never vendored, never forked, never patched from here.
- Don't invent the hermes interface: `batch_runner.py` takes CLI flags only (no
  config file). `verify_hermes_interface` cross-checks every flag we pass
  against the pinned checkout's `main()` signature, and our distribution/tool
  names against its registries (`toolsets.py`, `toolset_distributions.py`).
  Runs at `just install` and again before every batch.
- hermes writes batch output to `<checkout>/data/<run_name>/` (hardcoded);
  `run_hermes_batch` collects `trajectories.jsonl` into
  `$AVIARY_DATA_DIR/<run_id>/hermes/` immediately — the run store stays the
  single source of truth. Nothing is ever written into the checkout by hand.

## Things to never do

- Never edit goldens or fixtures to make tests pass.
- Never write trajectory data, session DBs, book texts, or rendered corpora
  into the repo.
- Never widen a family's tool allowlist without updating that family's
  verifier fixtures in the same change.
- Never point datagen at a non-dated teacher model ID (Roster refuses them).
- Never emit training-format text outside `src/aviary/render/serializer.py`.
- Never let a train render proceed past a holdout violation.
- Never use a Claude-class model as teacher or judge.
- Never paraphrase a non-persona voice into the persona's — lane B character
  dialogue or lane-C character-RP turns. The persona is present only in tasks
  (lane A) and simple-chats (inline lane-C seeds); it is transparent in roleplay.
- Never let personal (lane D) data into git, tests, fixtures, goldens, docs, or
  HF — synthetic only, always; lane D run data ships nowhere.
- Never author Sorcha persona content unilaterally — it is co-authored with the
  owner, and nothing Olivia-voiced ever enters a sorcha render (target-bounded).
