# aviary

Where you raise a parrot.

Bootstrap SFT corpus generation for **Swift Parrot**
(`spoomplesmaxx-flash-35B-A3`). Three data lanes feed one gated funnel and one
serializer:

- **Lane A — agentic**: teacher models run task-bank errands in-character as
  Olivia through [hermes-agent](https://github.com/NousResearch/hermes-agent)
  batch datagen (pinned checkout, called, never forked).
- **Lane B — fiction extraction**: CoSER-style decomposition of organic fiction
  (your ebook library, AO3, Gutenberg) into character profiles, scenes, and
  dialogue with inner thoughts — the non-LLM entropy anchor and the
  theory-of-mind data.
- **Lane C — self-play**: user-simulator ⇄ character conversations tuned for
  SillyTavern turn dynamics (lazy typo'd user turns, length control,
  impersonation avoidance).

Everything is outcome-verified, rubric-judged cross-vendor, scrubbed, deduped,
span-safely harmonized, and rendered through a single serializer into Swift
Parrot's ChatML + Hermes `<tool_call>` template — with-thoughts and no-thoughts
variants, next-speaker-prediction samples, and a DPO artifact from judge-rejected
rollout siblings. The task bank and verifiers become the GRPO reward suite later.

- **The contracts (start here, especially if you are Claude Code):**
  [`CLAUDE.md`](CLAUDE.md)
- **The serializer byte format:** [`render/CONTRACT.md`](render/CONTRACT.md)

## Quickstart

```sh
uv sync
just test                  # offline suite: goldens, gates, lanes, guards
cp .env.example .env       # keys: DEEPSEEK_API_KEY, GLM_API_KEY, OPENROUTER_API_KEY

# 1. verify teacher snapshot ids + prices (datagen/configs/teachers.yaml, pricing.yaml)
# 2. list books in datagen/configs/lane_b.yaml (local paths, never committed)
just pilot                 # small calibration run -> runs/<date>-pilot.manifest.yaml
just gate <run_id>
just render <run_id>
just stats <run_id>
just burn                  # refuses without a recent healthy pilot
```

Lane A additionally needs a pinned hermes-agent checkout: set
`AVIARY_HERMES_DIR`, run `just install`, add `a` to `lanes:` in
`datagen/configs/pilot.yaml`, and fill the task bank (`tasks/`).

## Status

Implemented and offline-tested end-to-end (fake-teacher fixtures; no live API
calls in tests). Before the first paid run: verify dated teacher snapshot IDs
and pricing, add books, and run a spend-capped lane B smoke on one book.

## Non-goals

Training code (spoomplesmaxx repo), hermes forks/patches, trajectory data in
git (private HF dataset repos, referenced from `runs/` manifests).
