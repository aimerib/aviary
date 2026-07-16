# aviary — recipe names are part of the workflow contract (see CLAUDE.md).
# Implement bodies; don't rename recipes.

default:
    @just --list

# Symlink datagen configs/toolsets/persona into hermes-agent's expected
# directories. Repo is source of truth; never edit symlink targets.
install:
    @echo "TODO: symlink datagen/* into the pinned hermes-agent checkout"

# ~200-rollout calibration run. Writes runs/<date>-pilot.manifest.yaml.
pilot:
    @echo "TODO: hermes batch_runner with datagen/configs/pilot.yaml"

# Full burn. MUST refuse to start without a recent pilot manifest
# whose keep_rates are within expected bands.
burn:
    @echo "TODO: guard on pilot manifest, then full batch_runner run"

# verify -> judge -> scrub/harmonize over a run's raw session records.
gate run_id:
    @echo "TODO: gate pipeline for {{run_id}}"

# Choke-point serialization + train/eval split for a gated run.
# Hard-fails on any holdout template in the train split.
render run_id:
    @echo "TODO: render {{run_id}} through render/ serializer"

# Keep rates, pass@N per template, spend, difficulty-band violations.
stats run_id:
    @echo "TODO: stats for {{run_id}}"

test:
    uv run pytest

lint:
    uv run ruff check . && uv run ruff format --check .
