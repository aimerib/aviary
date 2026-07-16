# aviary

Where you raise a parrot. This repo generates the bootstrap SFT corpus for
**Swift Parrot** (`aimeri/spoomplesmaxx-flash-35B-A3`, Qwen3.5-35B-A3B base) by
running teacher models through **hermes-agent** batch datagen, gating the
trajectories, and rendering survivors into Swift Parrot's exact chat template.
The same task bank + verifiers later become the GRPO reward suite (Atropos).

Pipeline (see `docs/pipeline.svg`):

```
tasks → rollouts (hermes batch_runner) → verify → judge → harmonize → render → ship
```

Volumes are a funnel by design: ~10,000 rollouts in, ~3,200 kept, split into
train + held-out eval. Rejection is a feature; the reject pile feeds difficulty
tuning, not the corpus.

## Vocabulary

- **trajectory** — one complete recorded job: every message, tool call, and
  tool result from task prompt to final state.
- **template vs instance** — a template is a parameterized task family in
  `tasks/`; an instance is one concrete parameterization of it.
- **teacher** — the model generating trajectories (DeepSeek V4 Flash for
  easy/mid families, V4 Pro for hard/long-horizon). Swift Parrot never
  generates in this repo; it only consumes the output.
- **choke point** — the single serializer in `render/` that emits
  training-format text. Nothing else is allowed to.

## Non-negotiable contracts

Violating any of these silently poisons the corpus. When a contract blocks a
task you were asked to do, stop and say so instead of working around it.

1. **Choke-point rule.** Exactly one code path (`render/`) converts raw hermes
   session records into Swift Parrot ChatML + Hermes-format `<tool_call>`
   text. The rendered template must be byte-identical across SFT, eval,
   deployment, and future GRPO. No inline formatting anywhere else — not in
   tests, not in scripts, not "just for debugging output that got committed."

2. **Golden-file rule.** `render/goldens/` locks serializer output.
   Goldens change only with an explicit contract version bump decided by a
   human. Never edit a golden to make a test pass. If output and golden
   disagree, the default assumption is the code is wrong.

3. **Span-protection rule.** The harmonizer (`gates/harmonize/`) may rewrite
   conversational spans only. Immutable: tool call arguments, tool results,
   file paths, quoted strings, identifiers, numbers, code. If an Olivia-voice
   paraphrase would touch protected text, drop the trajectory — do not bend
   the span boundary. (Scar tissue: Gemma-era paraphraser leakage.)

4. **Outcome-gate rule.** Verifiers judge final state, never the path taken.
   Trajectories where the teacher hit an error and recovered are keepers —
   prime data, do not filter on "an error occurred." Never fix a red verifier
   by loosening it; read the trajectory first and decide whether the verifier
   or the trajectory is wrong.

5. **Fixture rule.** Every verifier ships with fixture trajectories in
   `verifiers/fixtures/` covering at minimum: clean pass, clean fail, and
   recovered-failure pass. A verifier without fixtures does not merge.

6. **Split rule.** Holdout is family-level, declared in task YAML
   (`holdout: true`). The renderer must hard-fail — not warn — if a holdout
   template appears in a train render. Eval also gets unseen parameterizations
   of seen templates; that selection happens at render time, recorded in the
   run manifest.

7. **Data-in-git rule.** No trajectories, session records, or rendered corpora
   in git, ever. Per-run data goes to a private HF dataset repo referenced by
   its manifest. `runs/` holds manifests only. Exception: fixtures and goldens
   are small, curated, and tracked on purpose — they are the contract.

8. **Provenance rule.** Every pilot/burn writes a manifest
   (`runs/TEMPLATE.manifest.yaml`): hermes commit, dated teacher snapshot IDs,
   task-bank commit, datagen config hash, counts, keep rates, spend. Teacher
   model IDs are pinned dated snapshots, never `latest` — an upstream bump
   mid-burn shifts the data distribution invisibly.

9. **Source-of-truth rule.** The serializer consumes raw hermes session
   records. ShareGPT export is a human-readable preview, not an input.

10. **Cache-discipline rule.** System prompt and tool schemas must be
    byte-stable across all steps and rollouts of a run. This is a cost
    contract (~5x on the API bill), so treat prompt edits mid-run as a new run.

## Map

| Path                | What lives here                                            |
| ------------------- | ---------------------------------------------------------- |
| `tasks/<family>/`   | Task templates (YAML). Families: web, files, nightrunner, chains, social. Schema: `tasks/TEMPLATE.task.yaml` |
| `verifiers/`        | One verifier per template, pure functions. Fixtures in `verifiers/fixtures/` |
| `datagen/configs/`  | hermes batch_runner configs (workers, batch size, N)       |
| `datagen/toolsets/` | Toolset distributions — the per-family tool allowlists     |
| `datagen/persona/`  | Teacher-side Olivia persona (SOUL.md) and system prompt    |
| `gates/judge/`      | rp_judge hookup: Olivia-voice axis + quality rubric        |
| `gates/scrub/`      | PII + teacher-identity scrub, dedupe                       |
| `gates/harmonize/`  | Span-protected Olivia paraphrase                           |
| `render/`           | THE serializer + `goldens/`                                |
| `runs/`             | Manifests only                                             |
| `tests/`            | Pytest for verifiers, gates, renderer                      |
| `docs/pipeline.svg` | The pipeline, one picture                                  |

## Workflow

`just` recipes (stubs until implemented — implement, don't rename):

- `just pilot`  — ~200-rollout calibration run; writes a manifest
- `just burn`   — full run; must refuse to start without a recent pilot manifest
- `just gate`   — verify → judge → scrub/harmonize on a run's raw data
- `just render` — choke-point serialization + split; hard-fails on split-rule violations
- `just stats`  — keep rates, pass@N per template, spend, difficulty-band violations

Task-design invariant: templates target a 30–80% teacher pass rate. Outside
that band, revise the template (difficulty), don't touch the verifier.

## Stack and conventions

- Python 3.12, `uv` for env/deps, `ruff` for lint+format, `pytest`.
- Task templates and manifests are YAML; trajectories are JSONL (in HF repos,
  not here).
- hermes-agent is a **pinned external dependency we call** — never vendored,
  never forked, never patched from this repo. If hermes behavior needs
  changing, that's an upstream issue, not a monkeypatch.
- Don't invent hermes config keys. Read the pinned checkout's
  `datagen-config-examples/` and source before writing configs.
- hermes wants configs in its own directories at runtime: `just install`
  symlinks from this repo outward. The repo is the source of truth; never
  edit the symlink targets in place.
- Keep dependencies minimal. Verifiers and the serializer must stay pure
  functions over data — no network, no model calls inside them.

## Things to never do

- Never edit goldens or fixtures to make tests pass.
- Never write trajectory data, session DBs, or rendered corpora into the repo.
- Never widen a family's tool allowlist without updating that family's
  verifier fixtures in the same change.
- Never point datagen at a non-dated teacher model ID.
- Never emit training-format text outside `render/`.
- Never let a train render proceed past a holdout violation.
