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
from aviary.paths import REPO_ROOT, configs_dir, data_dir, manifest_path, runs_dir
from aviary.schema.manifest import (
    Artifacts,
    Counts,
    KeepRates,
    LaneKeepRate,
    Provenance,
    RunManifest,
    Teachers,
)
from aviary.schema.records import ConversationRecord
from aviary.targets import Target
from aviary.teacher.cache import ResponseCache
from aviary.teacher.client import HttpTeacherClient
from aviary.teacher.cost import CostLedger
from aviary.teacher.prompts import PromptSet
from aviary.teacher.roster import Roster

log = logging.getLogger(__name__)

# Fixed default reproducibility seed. Deliberate constant (not a runtime date) so a
# run is reproducible regardless of when it executes; override per run in the config.
DEFAULT_SEED = 20260716


def prompt_files(target: Target) -> dict[str, Path]:
    """The frozen PromptSet's contents for a build target. Persona-owned prompts
    (system + paraphrase) resolve through the target; the key for the persona
    system is constructed from the persona name, so the default target's keys —
    and therefore its PromptSet hash — are byte-identical to the pre-target era."""
    return {
        target.persona_system_key: target.persona_dir / "system.md",
        "laneb_profiles": REPO_ROOT / "datagen" / "prompts" / "laneb_profiles.md",
        "laneb_scenes": REPO_ROOT / "datagen" / "prompts" / "laneb_scenes.md",
        "laneb_dialogue": REPO_ROOT / "datagen" / "prompts" / "laneb_dialogue.md",
        "lanec_user_sim": REPO_ROOT / "datagen" / "prompts" / "lanec_user_sim.md",
        "judge_prompt": REPO_ROOT / "gates" / "judge" / "judge_prompt.md",
        "harmonize_prompt": REPO_ROOT / target.harmonize_prompt,
    }


class LaneCRunConfig(BaseModel):
    conversations_per_seed: int = 1
    personas: list[str] = Field(default_factory=lambda: ["lazy_texter"])
    rng_seed: int = DEFAULT_SEED
    # Companion seeds are generated from the vault, so their count is a property of
    # how much the owner has written — 527 today — not of the run's size. A pilot
    # must sample them or it is a burn: every seed is a full multi-turn self-play
    # conversation, two teacher calls per turn. 0 = all (burn).
    # Sampled on an even stride so the sample keeps the event/person/topic/obsession
    # mix rather than becoming whichever kind sorts first.
    max_companion_seeds: int = 0
    # Override lane_c.yaml's max_lane_b_seeds for this run kind (0 = use the file).
    # lane_c.yaml is shared with flash, so a pilot trims RP seeds here instead.
    max_lane_b_seeds: int = 0


class LaneBand(BaseModel):
    """Acceptable [min, max] verify/judge keep rates for ONE lane at burn time.
    Bands are guardrails against surprise regressions (a task that drifted too
    easy/hard, a rubric that broke), not precise targets — set them wide enough
    to pass a healthy pilot, tight enough to catch a collapse. Lanes differ by
    design: lane B judges low on the thought_quality floor, so its judge band
    sits far below lane A's."""

    verify: tuple[float, float] = (0.25, 0.95)
    judge: tuple[float, float] = (0.2, 0.95)


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
    # Hermes toolset distribution for the batch. Must enable, at 100%, toolsets
    # covering the task bank's tool union (verified before the batch runs).
    # terminal_web = terminal+file+web all at 100%: deterministic, covers the
    # files/web families, and the extra tools ride along as distractors.
    lane_a_distribution: str = "terminal_web"
    lane_a_max_turns: int = 10
    lane_c: LaneCRunConfig = Field(default_factory=LaneCRunConfig)
    # Lane D generation is free (no teacher calls) but JUDGING it is not, and the
    # export's size is fixed by history rather than by the run. 0 = all (burn).
    lane_d_max_records: int = 0
    # Cap on lane B books. 0 = the whole corpus (burn).
    #
    # This knob did not exist until 2026-07-23, and its absence was masked: lane B
    # replayed from the response cache, so "all 482 books" cost nothing and nobody
    # noticed the pilot had no cap. Changing the cache key format invalidated that
    # replay and the next pilot went to 27h / ~$111 of live extraction — to then
    # subsample it down to 20 lane C seeds. A pilot must be able to bound the one
    # lane whose volume comes from the corpus rather than from this file.
    lane_b_max_books: int = 0
    # Per-lane keep-rate bands the burn guard enforces against the blessing pilot.
    # Every lane the burn runs MUST have a band here and MUST have been measured by
    # the pilot, or the guard refuses (never burn an unpiloted lane — esp. lane A).
    burn_bands: dict[str, LaneBand] = Field(default_factory=dict)
    burn_max_pilot_age_days: int = 14

    @classmethod
    def load(cls, path: Path) -> RunConfig:
        return cls.model_validate(yaml.safe_load(path.read_text()))

    @classmethod
    def for_target(cls, kind: str, target: Target) -> RunConfig:
        """Run config for `kind` under `target`: `<kind>.<target>.yaml` if present,
        else the shared `<kind>.yaml`.

        Targets do not share a lane set — sorcha-v1 runs lane D, flash-v2_2 must
        never touch it — so they cannot share one run config either. Without this
        split, running Sorcha would mean editing pilot.yaml, which changes the
        datagen config hash and de-authorizes the flash burn as a side effect.
        """
        scoped = configs_dir() / f"{kind}.{target.name}.yaml"
        return cls.load(scoped if scoped.exists() else configs_dir() / f"{kind}.yaml")


def load_prompt_set(target: Target) -> PromptSet:
    return PromptSet.load(prompt_files(target))


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
    provider at a time).

    Counts the whole ROTATION POOL per role, not just the primary. A role that
    rotates can route any given unit to any model in its pool, so the cap has to be
    the smallest of them: with lane C's character side rotating deepseek-pro (12)
    and glm (6), reading only the primary gave 12 workers and drove glm at twice
    its own limit."""
    caps = [
        route.max_concurrency
        for role in roles
        for route in roster.assigned_pool(lane_key, role)
    ]
    return max(1, min(caps)) if caps else 1


def make_clients(roster: Roster, run_id: str) -> tuple[HttpTeacherClient, CostLedger]:
    ledger = CostLedger(configs_dir() / "pricing.yaml")
    cache = ResponseCache(data_dir() / "cache")
    return HttpTeacherClient(roster, cache=cache, ledger=ledger), ledger


class RunExistsError(RuntimeError):
    """A run with this id already exists. Overwriting it would rewrite the manifest
    and reuse the data dir — silently blending two generations (and reusing stale
    lane-B checkpoints) under one manifest. Refuse (data-integrity)."""


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
    target = Target.resolve()
    cfg = RunConfig.for_target(kind, target)
    roster = Roster.load(configs_dir() / "teachers.yaml")
    prompts = load_prompt_set(target)
    # Optional label distinguishes same-day runs of the same kind — e.g. A/B teacher
    # experiments (AVIARY_RUN_LABEL=glm -> 2026-07-17-pilot-glm). Sanitized to keep
    # run_id a safe path/filename segment.
    label = re.sub(r"[^A-Za-z0-9._-]", "-", os.environ.get("AVIARY_RUN_LABEL", "")).strip("-")
    run_id = f"{_dt.date.today().isoformat()}-{kind}" + (f"-{label}" if label else "")
    store = RunStore(run_id)
    _assert_fresh_run(run_id, store)
    client, ledger = make_clients(roster, run_id)

    if kind == "burn":
        from aviary.lanes.a_agentic.run import guard_burn

        guard_burn(
            runs_dir(),
            datagen_config_hash(),
            cfg.burn_bands,
            cfg.lanes,
            cfg.burn_max_pilot_age_days,
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
            target=target.name,
            base_model=target.base_model,
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
            if cfg.lane_b_max_books:
                from dataclasses import replace as _replace

                from aviary.lanes.b_fiction.pipeline import sample_books

                kept = sample_books(lane_b_cfg.books, cfg.lane_b_max_books)
                log.info(
                    "lane B: %d of %d books (cap=%d, %d holdout)",
                    len(kept),
                    len(lane_b_cfg.books),
                    cfg.lane_b_max_books,
                    sum(1 for b in kept if b.holdout),
                )
                lane_b_cfg = _replace(lane_b_cfg, books=kept)
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
            rollouts += _generate_lane_c(cfg, store, roster, client, prompts, target)

        if "a" in cfg.lanes:
            rollouts += _generate_lane_a(cfg, store, roster, prompts, run_id, target)

        if "d" in cfg.lanes:
            rollouts += _generate_lane_d(store, run_id, cfg.lane_d_max_records)
            # Lane D ships nowhere: run data stays under $AVIARY_DATA_DIR only,
            # exempt from the ship-to-HF step. The manifest records the exemption.
            manifest.artifacts.ship_exempt_lanes = sorted(
                {*manifest.artifacts.ship_exempt_lanes, "d"}
            )
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


def _companion_seeds(target: Target, prompts: PromptSet, max_seeds: int = 0) -> list:
    """Vault-grounded companion seeds, when lane_d.yaml declares a vault.

    Off unless configured, so flash-v2_2 is unaffected: no vault, no seeds. The
    vault lives in lane_d.yaml because it IS a personal stream (CLAUDE.md scopes
    lane D to 'chat exports, journals, notes'); lane C reading it mirrors lane C
    already seeding RP from a designated lane B run.
    """
    from aviary.lanes.c_selfplay.companion_seeds import companion_seeds, seed_mix
    from aviary.lanes.d_personal.adapter import LaneDConfig
    from aviary.lanes.d_personal.vault import Vault

    lane_d = configs_dir() / "lane_d.yaml"
    if not lane_d.exists():
        return []
    raw = yaml.safe_load(lane_d.read_text()) or {}
    vault_cfg = raw.get("vault")
    if not vault_cfg:
        return []

    from aviary.lanes.d_personal.vault import VaultConfig

    LaneDConfig.load(lane_d)  # fail early on a malformed lane D config
    vault = Vault.load(VaultConfig.model_validate(vault_cfg))
    seeds = companion_seeds(vault, prompts[target.persona_system_key], target.persona_speaker)
    if max_seeds and len(seeds) > max_seeds:
        step = len(seeds) / max_seeds
        seeds = [seeds[int(i * step)] for i in range(max_seeds)]
    log.info("lane C: %d companion seeds from vault %s", len(seeds), seed_mix(seeds))
    return seeds


def _generate_lane_c(cfg, store, roster, client, prompts, target: Target) -> int:
    from aviary.lanes.c_selfplay.driver import LaneCConfig, run_selfplay
    from aviary.lanes.c_selfplay.seeds import load_inline_seeds, seeds_from_lane_b
    from aviary.lanes.c_selfplay.usersim import load_personas
    from aviary.teacher.pool import TeacherPool

    raw = yaml.safe_load((configs_dir() / "lane_c.yaml").read_text()) or {}
    seeds = load_inline_seeds(
        configs_dir() / "lane_c.yaml", prompts[target.persona_system_key], target.persona_speaker
    )
    # RP characters seed from a DESIGNATED lane B run (an RP-appropriate corpus, e.g.
    # AO3), not necessarily this run's lane B. Published-fiction characters are
    # off-distribution for RP, so lane C never seeds from the prose corpus. Falls back
    # to this run's own lane B output when seed_from_run is unset (combined [b,c] run).
    seed_run = raw.get("seed_from_run")
    seed_store = RunStore(seed_run) if seed_run else store
    max_b = cfg.lane_c.max_lane_b_seeds or raw.get("max_lane_b_seeds")
    seeds += seeds_from_lane_b(seed_store, max_seeds=max_b)
    seeds += _companion_seeds(target, prompts, cfg.lane_c.max_companion_seeds)
    personas = load_personas(REPO_ROOT / "datagen" / "persona" / "user_sims")
    user_sim = roster.assigned("lane_c", "user_sim")
    character_pool = roster.assigned_pool("lane_c", "character")
    lane_cfg = LaneCConfig(**raw.get("driver", {}))

    def _character_for(seed_id: str) -> str:
        """Which teacher plays the character in THIS conversation. Keyed by seed so
        it is stable across reruns (cache) and so a seed's two conversations at
        conversations_per_seed>1 can still differ by rng, not by model churn."""
        return roster.rotate(character_pool, seed_id, avoid=user_sim.provider).id

    if len(character_pool) > 1:
        log.info(
            "lane C: character side rotates across %s (user-sim %s)",
            [r.id for r in character_pool],
            user_sim.id,
        )

    # Draw one seed per conversation up front, in a fixed order: each conversation is
    # then fully determined by its own seed, so running them concurrently below yields
    # the same corpus as serial execution (reproducibility survives parallelism).
    rng = random.Random(cfg.lane_c.rng_seed)
    jobs = [
        (seed, personas[persona_id], rng.randrange(2**31))
        for seed in seeds
        # A seed may restrict which user-sim personas suit it: companion scenes
        # only make sense in the owner's voice, RP scenes only in a roleplayer's.
        # Crossing every seed with every persona produced incoherent records and
        # multiplied the run by 4-5x for no gain.
        for persona_id in (seed.user_sim_personas or cfg.lane_c.personas)
        if persona_id in personas
        for _ in range(cfg.lane_c.conversations_per_seed)
    ]

    def _one(job):
        seed, persona, conv_seed = job
        return run_selfplay(
            seed,
            persona,
            lane_cfg,
            client,
            {"user_sim": user_sim.id, "character": _character_for(seed.seed_id)},
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
    empty = 0
    for job, res in zip(jobs, results, strict=True):
        if isinstance(res, Exception):
            log.warning("lane C conversation (seed %s) failed: %s", job[0].seed_id, res)
        elif not res.messages:
            # A conversation that produced no turns is not a record. This is almost
            # always a teacher misconfiguration rather than a hard failure — a
            # reasoning model with no thinking-off control spends its whole budget
            # reasoning and returns empty content, so the loop breaks on turn one.
            empty += 1
        else:
            records.append(res)

    if empty:
        share = empty / len(jobs)
        # Loud on purpose. 74% of lane C came out empty once and the run still
        # exited 0 with a full-looking jsonl: the same shape as the keyless-gate
        # corruption. A silent majority-empty lane must never look like success.
        log.log(
            logging.ERROR if share > 0.2 else logging.WARNING,
            "lane C: %d/%d conversations produced NO turns (%.0f%%). Above ~20%% this "
            "is a teacher config problem, not bad luck — check that the user_sim "
            "model has a thinking-off control in teachers.yaml.",
            empty,
            len(jobs),
            share * 100,
        )
    return write_jsonl(store.raw("c"), records)


def _generate_lane_a(cfg, store, roster, prompts, run_id, target: Target) -> int:
    from aviary.lanes.a_agentic.ingest import IngestContext
    from aviary.lanes.a_agentic.run import ingest_trajectories, run_hermes_batch
    from aviary.lanes.a_agentic.taskbank import expand_all, load_taskbank
    from aviary.paths import hermes_dir

    templates = load_taskbank(REPO_ROOT / "tasks")
    instances = expand_all(templates, target.address)
    if not instances:
        log.warning("lane A enabled but the task bank is empty; skipping")
        return 0
    teacher = roster.assigned("lane_a", "easy_mid")
    api_key = os.environ.get(teacher.api_key_env) or (
        os.environ.get(teacher.api_key_env_fallback) if teacher.api_key_env_fallback else None
    )
    if not api_key:
        raise RuntimeError(f"missing API key for lane A teacher: set {teacher.api_key_env}")
    required_tools = sorted({tool for t in templates for tool in t.tools})
    trajectories = run_hermes_batch(
        hermes_dir(),
        instances,
        distribution=cfg.lane_a_distribution,
        required_tools=required_tools,
        wire_model=teacher.wire_model,
        base_url=teacher.base_url,
        api_key=api_key,
        system_prompt=prompts[target.persona_system_key],
        store=store,
        num_workers=cfg.lane_a_num_workers,
        batch_size=cfg.lane_a_batch_size,
        max_turns=cfg.lane_a_max_turns,
        run_name=run_id,
        timeout_s=cfg.lane_a_timeout_s,
    )
    ctx = IngestContext(
        run_id=run_id,
        instances=instances,
        system_prompt=prompts[target.persona_system_key],
        persona_speaker=target.persona_speaker,
        teacher_id=teacher.id,
        hermes_commit=roster.hermes_pin,
        prompt_set_hash=prompts.hash,
    )
    return ingest_trajectories(trajectories, ctx, store)


def _generate_lane_d(store: RunStore, run_id: str, max_records: int = 0) -> int:
    """Pure normalization — no teacher calls. Sources are local paths in
    lane_d.yaml (never committed); output stays in the run store."""
    from aviary.lanes.d_personal.adapter import LaneDConfig, parse_all

    lane_cfg = LaneDConfig.load(configs_dir() / "lane_d.yaml")
    if not lane_cfg.sources:
        log.warning("lane D enabled but lane_d.yaml declares no sources; skipping")
        return 0
    records = list(parse_all(lane_cfg, run_id))
    if max_records and len(records) > max_records:
        # Even stride, not head: a pilot sample must span the whole history, or its
        # measured keep rate describes one era and then authorizes a burn over six.
        step = len(records) / max_records
        records = [records[int(i * step)] for i in range(max_records)]
        log.info("lane D: sampled %d records (cap)", len(records))
    return write_jsonl(store.raw("d"), records)


def cmd_gate(run_id: str) -> None:
    from aviary.gates.judge import Rubric
    from aviary.gates.pipeline import default_resolver, run_gates
    from aviary.gates.scrub import load_lane_policies, load_patterns
    from aviary.lanes.a_agentic.taskbank import load_taskbank

    manifest = find_manifest(run_id)
    _assert_frozen_inputs(manifest)
    target = Target.resolve()
    cfg = RunConfig.for_target(manifest.kind, target)
    roster = Roster.load(configs_dir() / "teachers.yaml")
    prompts = load_prompt_set(target)
    prompts.assert_hash(manifest.prompt_set_hash)
    store = RunStore(run_id)
    client, ledger = make_clients(roster, run_id)

    quality = Rubric.load(REPO_ROOT / target.quality_rubric)
    rubrics = {
        "a": quality,
        "c": quality,  # persona simple-chats; character-RP records use c_character
        "b": Rubric.load(REPO_ROOT / "gates" / "judge" / "laneb.rubric.yaml"),
    }
    for lane, rubric_path in target.lane_rubrics.items():
        rubrics[lane] = Rubric.load(REPO_ROOT / rubric_path)
    # Gating an older run means restoring its generation-time gates/ tree (frozen-
    # inputs rule), which may predate later-added rubrics. Load those only if
    # present; a record that actually routes to a missing rubric fails loudly
    # (KeyError names it) instead of blocking runs that never needed it.
    character_rp = REPO_ROOT / target.character_rubric
    if character_rp.exists():
        rubrics["c_character"] = Rubric.load(character_rp)
    patterns = load_patterns(
        REPO_ROOT / "gates" / "scrub" / "denylist.yaml",
        REPO_ROOT / "gates" / "scrub" / "pii_patterns.yaml",
    )
    scrub_policies = load_lane_policies(REPO_ROOT / "gates" / "scrub" / "lane_policy.yaml")
    if "d" in scrub_policies:
        # Third-party protection for lane D: stable pseudonyms, mapping persisted
        # under the run dir. Rules are personal -> a LOCAL file lane_d.yaml points
        # at (unset = no-op rules, synthetic smoke only).
        from aviary.gates.pseudonym import Pseudonymizer, PseudonymRules

        lane_d_cfg_path = configs_dir() / "lane_d.yaml"
        rules_path = None
        if lane_d_cfg_path.exists():
            raw_d = yaml.safe_load(lane_d_cfg_path.read_text()) or {}
            if raw_d.get("pseudonym_rules"):
                rules_path = Path(raw_d["pseudonym_rules"]).expanduser()
        pseudo = Pseudonymizer(
            PseudonymRules.load(rules_path), store.root / "scrub" / "pseudonyms.json"
        )
        scrub_policies["d"].transform = pseudo.apply
    template_verifiers = {t.id: t.verifier for t in load_taskbank(REPO_ROOT / "tasks")}
    stats = run_gates(
        store,
        default_resolver(template_verifiers),
        rubrics,
        patterns,
        roster,
        client,
        prompts,
        persona_speaker=target.persona_speaker,
        harmonize_policy={k: tuple(v) for k, v in target.harmonize.items()},
        scrub_policies=scrub_policies,
        dedupe_threshold=cfg.dedupe_threshold,
        # Judge fan-out capped by the tightest judge provider so any per-record
        # routing stays within limits.
        judge_workers=_lane_workers(roster, "judge", ("primary", "secondary", "tertiary")),
        # Harmonize fans out the same way, capped by the harmonizer's provider.
        harmonize_workers=_lane_workers(roster, "harmonizer", ("primary",)),
    )
    manifest.counts.verified = stats.verified
    manifest.counts.judged = stats.judged
    manifest.keep_rates = KeepRates(
        verify=round(stats.verify_rate, 4),
        judge=round(stats.judge_rate, 4),
        by_lane={
            lane: LaneKeepRate(verify=round(s.verify_rate, 4), judge=round(s.judge_rate, 4))
            for lane, s in stats.by_lane.items()
        },
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
    target = Target.resolve()
    cfg = RunConfig.for_target(manifest.kind, target)
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
        include_lanes=set(target.lanes),
    )
    manifest.counts.rendered_train = counts.train
    manifest.counts.rendered_eval = counts.eval
    manifest.counts.rendered_dpo_pairs = counts.dpo_pairs
    manifest.counts.rendered_nsp = counts.nsp
    kept = list(read_jsonl(store.gated_kept(), ConversationRecord))
    manifest.artifacts = Artifacts(
        hf_dataset=manifest.artifacts.hf_dataset,
        ship_exempt_lanes=manifest.artifacts.ship_exempt_lanes,
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
