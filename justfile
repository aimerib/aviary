# aviary — recipe names are part of the workflow contract (see CLAUDE.md).
# Bodies delegate 1:1 to the aviary CLI (src/aviary/cli.py).
#
# Build-target selection: every recipe honors $AVIARY_TARGET
# (datagen/configs/targets/<name>.yaml; default flash-v2_2 — the v2.2 pipeline
# runs unchanged unless you explicitly select another target, e.g.
# `AVIARY_TARGET=sorcha-v1 just pilot`).

set dotenv-load := true

default:
    @just --list

# Verify the pinned hermes-agent checkout ($AVIARY_HERMES_DIR) and symlink
# datagen configs/toolsets/persona into it. Repo is source of truth; never
# edit symlink targets.
install:
    uv run aviary install

# ~200-rollout calibration run across lanes enabled in datagen/configs/pilot.yaml.
# Writes runs/<date>-pilot.manifest.yaml.
pilot:
    uv run aviary pilot

# Full burn. Refuses to start without a recent pilot manifest whose keep_rates
# are within the bands in burn.yaml and whose datagen config hash matches.
burn:
    uv run aviary burn

# verify -> judge -> scrub -> dedupe -> harmonize over a run's raw records.
gate run_id:
    uv run aviary gate {{run_id}}

# Choke-point serialization + train/eval split for a gated run.
# Hard-fails on any holdout family in the train split.
render run_id *args:
    uv run aviary render {{run_id}} {{args}}

# Keep rates, pass@N per template, spend, difficulty-band violations.
stats run_id:
    uv run aviary stats {{run_id}}

# Upload a rendered run to a PRIVATE HF dataset repo; records it in the manifest.
# Always private (no flag widens it); refuses runs containing ship-exempt lanes.
# Use --dry-run to validate and list files without contacting HF.
ship run_id *args:
    uv run aviary ship {{run_id}} {{args}}

test:
    uv run pytest

lint:
    uv run ruff check . && uv run ruff format --check .
