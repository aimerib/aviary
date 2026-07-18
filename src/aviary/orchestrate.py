"""Run orchestration behind the just recipes: pilot/burn -> gate -> render -> stats.

Each command is manifest-scoped: pilot/burn create runs/<id>.manifest.yaml, gate and
render update it. Generation data lives under $AVIARY_DATA_DIR/<run_id>/ only.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import random
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from aviary.hashing import hash_tree
from aviary.io.jsonl import read_jsonl, write_jsonl
from aviary.io.store import RunStore
from aviary.paths import REPO_ROOT, configs_dir, data_dir, runs_dir
from aviary.schema.manifest import (
    Artifacts,
    Counts,
    KeepRates,
    Provenance,
    RunManifest,
    Teachers,
)
from aviary.schema.records import ConversationRecord
from aviary.teacher.cache import ResponseCache
from aviary.teacher.client import HttpTeacherClient
from aviary.teacher.cost import CostLedger
from aviary.teacher.prompts import PromptSet
from aviary.teacher.roster import Roster

log = logging.getLogger(__name__)

# Fixed default reproducibility seed. Deliberate constant (not a runtime date) so a
# run is reproducible regardless of when it executes; override per run in the config.
DEFAULT_SEED = 20260716

PROMPT_FILES = {
    "olivia_system": REPO_ROOT / "datagen" / "persona" / "system.md",
    "laneb_profiles": REPO_ROOT / "datagen" / "prompts" / "laneb_profiles.md",
    "laneb_scenes": REPO_ROOT / "datagen" / "prompts" / "laneb_scenes.md",
    "laneb_dialogue": REPO_ROOT / "datagen" / "prompts" / "laneb_dialogue.md",
    "lanec_user_sim": REPO_ROOT / "datagen" / "prompts" / "lanec_user_sim.md",
    "judge_prompt": REPO_ROOT / "gates" / "judge" / "judge_prompt.md",
    "harmonize_prompt": REPO_ROOT / "gates" / "harmonize" / "paraphrase_prompt.md",
}


class LaneCRunConfig(BaseModel):
    conversations_per_seed: int = 1
    personas: list[str] = Field(default_factory=lambda: ["lazy_texter"])
    rng_seed: int = DEFAULT_SEED


class RunConfig(BaseModel):
    kind: str
    lanes: list[str]
    eval_param_fraction: float = 0.15
    split_seed: int = DEFAULT_SEED
    dpo_min_margin: float = 1.0
    dedupe_threshold: float = 0.8
    lane_a_num_workers: int = 4
    lane_a_batch_size: int = 20
    lane_a_timeout_s: int = 14400  # wall-clock cap on the hermes batch subprocess (4h)
    lane_c: LaneCRunConfig = Field(default_factory=LaneCRunConfig)
    burn_bands_verify: tuple[float, float] = (0.25, 0.65)
    burn_bands_judge: tuple[float, float] = (0.6, 0.95)
    burn_max_pilot_age_days: int = 14

    @classmethod
    def load(cls, path: Path) -> RunConfig:
        return cls.model_validate(yaml.safe_load(path.read_text()))


def load_prompt_set() -> PromptSet:
    return PromptSet.load(PROMPT_FILES)


def datagen_config_hash() -> str:
    dg = REPO_ROOT / "datagen"
    return hash_tree(dg / "configs", dg / "toolsets", dg / "persona", dg / "prompts")


def gate_inputs_hash() -> str:
    """Everything that decides gate/render outcomes but lives OUTSIDE datagen/:
    judge rubrics + scrub patterns (gates/), verifier code (verifiers/), and the
    task bank (tasks/). Frozen at generation, re-asserted before gate/render."""
    return hash_tree(REPO_ROOT / "gates", REPO_ROOT / "verifiers", REPO_ROOT / "tasks")


def _assert_frozen_inputs(manifest: RunManifest) -> None:
    """Refuse to gate/render if the inputs drifted since generation. `assert_hash`
    already freezes prompt texts; this covers roster/config and the gate/render
    inputs, closing the 'changed a rubric between generate and gate' hole."""
    checks = {
        "datagen_config_hash": (manifest.provenance.datagen_config_hash, datagen_config_hash()),
        "gate_inputs_hash": (manifest.provenance.gate_inputs_hash, gate_inputs_hash()),
    }
    drifted = [name for name, (recorded, now) in checks.items() if recorded and recorded != now]
    if drifted:
        raise ProvenanceError(
            f"inputs changed since generation ({', '.join(drifted)} differ). Gating/rendering "
            "under changed config would silently alter results — restore the inputs or "
            "regenerate the run."
        )


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class ProvenanceError(RuntimeError):
    """Provenance could not be captured. Refuse rather than ship an empty field —
    an unverifiable run is worthless for reproducibility (provenance rule)."""


def _git_commit() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError) as e:
        raise ProvenanceError(f"could not read task-bank git commit: {e}") from e
    if not out:
        raise ProvenanceError("git rev-parse HEAD returned empty task-bank commit")
    return out


def _lane_workers(roster: Roster, lane_key: str, roles: tuple[str, ...]) -> int:
    """Concurrency for coarse-grained lane parallelism (whole books/conversations):
    the smallest per-provider call cap among the lane's models, so N concurrent
    units keep each provider within max_concurrency (each unit issues <=1 call per
    provider at a time)."""
    caps = [roster.assigned(lane_key, role).max_concurrency for role in roles]
    return max(1, min(caps)) if caps else 1


def make_clients(roster: Roster, run_id: str) -> tuple[HttpTeacherClient, CostLedger]:
    ledger = CostLedger(configs_dir() / "pricing.yaml")
    cache = ResponseCache(data_dir() / "cache")
    return HttpTeacherClient(roster, cache=cache, ledger=ledger), ledger


class RunExistsError(RuntimeError):
    """A run with this id already exists. Overwriting it would rewrite the manifest
    and reuse the data dir — silently blending two generations (and reusing stale
    lane-B checkpoints) under one manifest. Refuse (data-integrity)."""


def manifest_path(run_id: str) -> Path:
    return runs_dir() / f"{run_id}.manifest.yaml"


def _assert_fresh_run(run_id: str, store: RunStore) -> None:
    """run_id is <date>-<kind>, so a second same-day run of the same kind collides.
    Refuse rather than overwrite; AVIARY_ALLOW_RERUN=1 opts into deliberate resume."""
    if os.environ.get("AVIARY_ALLOW_RERUN") == "1":
        return
    existing = [p for p in (manifest_path(run_id), store.root) if p.exists()]
    if existing:
        raise RunExistsError(
            f"run {run_id!r} already exists ({', '.join(str(p) for p in existing)}). "
            "A same-day rerun overwrites the manifest and reuses the data dir, silently "
            "mixing old and new records. Remove them, or set AVIARY_ALLOW_RERUN=1 to "
            "resume deliberately (unchanged config only)."
        )


def find_manifest(run_id: str) -> RunManifest:
    return RunManifest.load(manifest_path(run_id))


def cmd_generate(kind: str) -> str:
    """pilot/burn: run every enabled lane's generation, write the manifest."""
    cfg = RunConfig.load(configs_dir() / f"{kind}.yaml")
    roster = Roster.load(configs_dir() / "teachers.yaml")
    prompts = load_prompt_set()
    # Optional label distinguishes same-day runs of the same kind — e.g. A/B teacher
    # experiments (AVIARY_RUN_LABEL=glm -> 2026-07-17-pilot-glm). Sanitized to keep
    # run_id a safe path/filename segment.
    label = re.sub(r"[^A-Za-z0-9._-]", "-", os.environ.get("AVIARY_RUN_LABEL", "")).strip("-")
    run_id = f"{_dt.date.today().isoformat()}-{kind}" + (f"-{label}" if label else "")
    store = RunStore(run_id)
    _assert_fresh_run(run_id, store)
    client, ledger = make_clients(roster, run_id)

    if kind == "burn":
        from aviary.lanes.a_agentic.run import BurnBands, guard_burn

        guard_burn(
            runs_dir(),
            datagen_config_hash(),
            BurnBands(
                verify=cfg.burn_bands_verify,
                judge=cfg.burn_bands_judge,
                max_age_days=cfg.burn_max_pilot_age_days,
            ),
            _dt.date.today().isoformat(),
        )

    # Verify the hermes checkout actually matches the pinned tag at run time (not
    # just in the opt-in `just install`) and record the VERIFIED checkout, never a
    # value merely trusted from YAML. Only when lane A (hermes) is enabled.
    hermes_commit = roster.hermes_pin
    if "a" in cfg.lanes:
        from aviary.lanes.a_agentic.hermes_config import verify_hermes_pin
        from aviary.paths import hermes_dir

        hermes_commit = verify_hermes_pin(hermes_dir(), roster.hermes_pin)

    manifest = RunManifest(
        run_id=run_id,
        kind=kind,  # type: ignore[arg-type]
        lanes=cfg.lanes,
        started=_now_iso(),
        provenance=Provenance(
            hermes_commit=hermes_commit,
            task_bank_commit=_git_commit(),
            datagen_config_hash=datagen_config_hash(),
            gate_inputs_hash=gate_inputs_hash(),
        ),
        teachers=Teachers(
            roster=[
                {"id": t.id, "provider": t.provider, "route": t.route}  # type: ignore[list-item]
                for t in roster.teachers
            ],
            assignments=roster.assignments,
        ),
        prompt_set_hash=prompts.hash,
    )
    manifest.save(manifest_path(run_id))

    try:
        rollouts = 0
        if "b" in cfg.lanes:
            from aviary.lanes.b_fiction.pipeline import LaneBConfig, run_lane_b

            lane_b_cfg = LaneBConfig.load(configs_dir() / "lane_b.yaml")
            roles = ("profiles", "scenes", "dialogue")
            models = {role: roster.assigned("lane_b", role).id for role in roles}
            rollouts += run_lane_b(
                lane_b_cfg,
                store,
                client,
                prompts,
                models,
                max_workers=_lane_workers(roster, "lane_b", roles),
            )

        if "c" in cfg.lanes:
            rollouts += _generate_lane_c(cfg, store, roster, client, prompts)

        if "a" in cfg.lanes:
            rollouts += _generate_lane_a(cfg, store, roster, prompts, run_id)
    except Exception:
        # Leave a durable failure marker instead of a manifest that looks complete.
        manifest.status = "failed"
        manifest.finished = _now_iso()
        manifest.spend_usd = ledger.to_spend()
        manifest.save(manifest_path(run_id))
        raise

    manifest.counts = Counts(rollouts=rollouts)
    manifest.spend_usd = ledger.to_spend()
    manifest.status = "complete"
    manifest.finished = _now_iso()
    manifest.save(manifest_path(run_id))
    log.info("run %s: %d rollouts, $%.2f", run_id, rollouts, ledger.total)
    return run_id


def _generate_lane_c(cfg, store, roster, client, prompts) -> int:
    from aviary.lanes.c_selfplay.driver import LaneCConfig, run_selfplay
    from aviary.lanes.c_selfplay.seeds import load_inline_seeds, seeds_from_lane_b
    from aviary.lanes.c_selfplay.usersim import load_personas
    from aviary.teacher.pool import TeacherPool

    raw = yaml.safe_load((configs_dir() / "lane_c.yaml").read_text()) or {}
    seeds = load_inline_seeds(configs_dir() / "lane_c.yaml", prompts["olivia_system"])
    # RP characters seed from a DESIGNATED lane B run (an RP-appropriate corpus, e.g.
    # AO3), not necessarily this run's lane B. Published-fiction characters are
    # off-distribution for RP, so lane C never seeds from the prose corpus. Falls back
    # to this run's own lane B output when seed_from_run is unset (combined [b,c] run).
    seed_run = raw.get("seed_from_run")
    seed_store = RunStore(seed_run) if seed_run else store
    seeds += seeds_from_lane_b(seed_store, max_seeds=raw.get("max_lane_b_seeds"))
    personas = load_personas(REPO_ROOT / "datagen" / "persona" / "user_sims")
    models = {
        "user_sim": roster.assigned("lane_c", "user_sim").id,
        "character": roster.assigned("lane_c", "character").id,
    }
    lane_cfg = LaneCConfig(**raw.get("driver", {}))

    # Draw one seed per conversation up front, in a fixed order: each conversation is
    # then fully determined by its own seed, so running them concurrently below yields
    # the same corpus as serial execution (reproducibility survives parallelism).
    rng = random.Random(cfg.lane_c.rng_seed)
    jobs = [
        (seed, personas[persona_id], rng.randrange(2**31))
        for seed in seeds
        for persona_id in cfg.lane_c.personas
        for _ in range(cfg.lane_c.conversations_per_seed)
    ]

    def _one(job):
        seed, persona, conv_seed = job
        return run_selfplay(
            seed,
            persona,
            lane_cfg,
            client,
            models,
            prompts,
            run_id=store.run_id,
            prompt_set_hash=prompts.hash,
            rng=random.Random(conv_seed),
        )

    pool = TeacherPool(
        client, roster, max_workers=_lane_workers(roster, "lane_c", ("user_sim", "character"))
    )
    results = pool.run([lambda j=job: _one(j) for job in jobs])
    records = []
    for job, res in zip(jobs, results, strict=True):
        if isinstance(res, Exception):
            log.warning("lane C conversation (seed %s) failed: %s", job[0].seed_id, res)
        else:
            records.append(res)
    return write_jsonl(store.raw("c"), records)


def _generate_lane_a(cfg, store, roster, prompts, run_id) -> int:
    from aviary.lanes.a_agentic.ingest import IngestContext
    from aviary.lanes.a_agentic.run import ingest_trajectories, run_hermes_batch
    from aviary.lanes.a_agentic.taskbank import expand_all, load_taskbank
    from aviary.lanes.a_agentic.toolsets import load_toolset_schemas
    from aviary.paths import hermes_dir

    templates = load_taskbank(REPO_ROOT / "tasks")
    instances = expand_all(templates)
    if not instances:
        log.warning("lane A enabled but the task bank is empty; skipping")
        return 0
    teacher = roster.assigned("lane_a", "easy_mid")
    toolsets = sorted({tool for t in templates for tool in t.tools})
    trajectories = run_hermes_batch(
        hermes_dir(),
        instances,
        toolsets=toolsets,
        wire_model=teacher.wire_model,
        system_prompt=prompts["olivia_system"],
        store=store,
        num_workers=cfg.lane_a_num_workers,
        batch_size=cfg.lane_a_batch_size,
        run_name=run_id,
        timeout_s=cfg.lane_a_timeout_s,
    )
    ctx = IngestContext(
        run_id=run_id,
        instances=instances,
        system_prompt=prompts["olivia_system"],
        tools_schema_by_family=load_toolset_schemas(REPO_ROOT / "datagen" / "toolsets"),
        teacher_id=teacher.id,
        hermes_commit=roster.hermes_pin,
        prompt_set_hash=prompts.hash,
    )
    return ingest_trajectories(trajectories, ctx, store)


def cmd_gate(run_id: str) -> None:
    from aviary.gates.judge import Rubric
    from aviary.gates.pipeline import default_resolver, run_gates
    from aviary.gates.scrub import load_patterns
    from aviary.lanes.a_agentic.taskbank import load_taskbank

    manifest = find_manifest(run_id)
    _assert_frozen_inputs(manifest)
    cfg = RunConfig.load(configs_dir() / f"{manifest.kind}.yaml")
    roster = Roster.load(configs_dir() / "teachers.yaml")
    prompts = load_prompt_set()
    prompts.assert_hash(manifest.prompt_set_hash)
    store = RunStore(run_id)
    client, ledger = make_clients(roster, run_id)

    quality = Rubric.load(REPO_ROOT / "gates" / "judge" / "quality.rubric.yaml")
    rubrics = {
        "a": quality,
        "c": quality,  # Olivia simple-chats; character-RP records use c_character
        "b": Rubric.load(REPO_ROOT / "gates" / "judge" / "laneb.rubric.yaml"),
        "c_character": Rubric.load(REPO_ROOT / "gates" / "judge" / "character_rp.rubric.yaml"),
    }
    patterns = load_patterns(
        REPO_ROOT / "gates" / "scrub" / "denylist.yaml",
        REPO_ROOT / "gates" / "scrub" / "pii_patterns.yaml",
    )
    template_verifiers = {t.id: t.verifier for t in load_taskbank(REPO_ROOT / "tasks")}
    stats = run_gates(
        store,
        default_resolver(template_verifiers),
        rubrics,
        patterns,
        roster,
        client,
        prompts,
        dedupe_threshold=cfg.dedupe_threshold,
    )
    manifest.counts.verified = stats.verified
    manifest.counts.judged = stats.judged
    manifest.keep_rates = KeepRates(
        verify=round(stats.verify_rate, 4), judge=round(stats.judge_rate, 4)
    )
    for teacher_id, usd in ledger.to_spend().by_teacher.items():
        manifest.spend_usd.by_teacher[teacher_id] = (
            manifest.spend_usd.by_teacher.get(teacher_id, 0.0) + usd
        )
    manifest.spend_usd.total = round(sum(manifest.spend_usd.by_teacher.values()), 4)
    manifest.save(manifest_path(run_id))


def cmd_render(run_id: str, prior_run_ids: list[str] | None = None) -> None:
    from aviary.render.run import render_run

    manifest = find_manifest(run_id)
    _assert_frozen_inputs(manifest)
    cfg = RunConfig.load(configs_dir() / f"{manifest.kind}.yaml")
    store = RunStore(run_id)
    extra: list[ConversationRecord] = []
    for prior in prior_run_ids or []:
        extra.extend(read_jsonl(RunStore(prior).gated_kept(), ConversationRecord))

    counts = render_run(
        store,
        eval_param_fraction=cfg.eval_param_fraction,
        split_seed=cfg.split_seed,
        dpo_min_margin=cfg.dpo_min_margin,
        extra_corpus=extra,
    )
    manifest.counts.rendered_train = counts.train
    manifest.counts.rendered_eval = counts.eval
    manifest.counts.rendered_dpo_pairs = counts.dpo_pairs
    manifest.counts.rendered_nsp = counts.nsp
    kept = list(read_jsonl(store.gated_kept(), ConversationRecord))
    manifest.artifacts = Artifacts(
        hf_dataset=manifest.artifacts.hf_dataset,
        eval_families_held_out=sorted({r.provenance.family for r in kept if r.provenance.holdout}),
        eval_param_seed=cfg.split_seed,
    )
    manifest.save(manifest_path(run_id))


def cmd_stats(run_id: str) -> str:
    from aviary.lanes.a_agentic.taskbank import load_taskbank
    from aviary.stats import compute_stats, format_stats

    manifest = find_manifest(run_id)
    stats = compute_stats(RunStore(run_id), load_taskbank(REPO_ROOT / "tasks"))
    return format_stats(stats, manifest)
