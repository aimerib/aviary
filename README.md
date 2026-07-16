# aviary

Where you raise a parrot.

Bootstrap SFT trace generation for **Swift Parrot**
(`spoomplesmaxx-flash-35B-A3`): teacher models run agentic tasks in-character
as Olivia through [hermes-agent](https://github.com/NousResearch/hermes-agent)
batch datagen; trajectories are outcome-verified, voice-judged, span-safely
harmonized, and rendered through a single serializer into Swift Parrot's chat
template. The task bank and verifiers built here become the GRPO reward suite
later.

- **The pipeline in one picture:** [`docs/pipeline.svg`](docs/pipeline.svg)
- **The contracts (start here, especially if you are Claude Code):**
  [`CLAUDE.md`](CLAUDE.md)

## Status

Pre-pilot scaffold. Nothing runs yet. First milestones:

1. `just install` — pin a hermes-agent checkout, wire symlinks
2. Three task templates in one family, with verifiers + fixtures
3. `just pilot` — ~200 rollouts to calibrate keep rates
4. Renderer + goldens against pilot data

## Non-goals

Training code (lives in the spoomplesmaxx repo), hermes forks/patches
(pinned dependency only), and any trajectory data in git (private HF
dataset repos, referenced from `runs/` manifests).
