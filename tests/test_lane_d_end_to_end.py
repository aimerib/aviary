"""Sorcha prep acceptance: a synthetic lane D corpus flows end to end OFFLINE
under the sorcha-v1 target — adapter → gates → render → stats — with time-blocked
holdout enforced, the manifest-relevant counts produced, and zero Olivia anywhere
in the rendered output. All people and messages are synthetic (radioactive rule).
"""

from __future__ import annotations

import json
from pathlib import Path

from aviary.gates.pipeline import default_resolver, run_gates
from aviary.gates.pseudonym import Pseudonymizer, PseudonymRules
from aviary.gates.scrub import load_lane_policies, load_patterns
from aviary.io.jsonl import read_jsonl, write_jsonl
from aviary.io.store import RunStore
from aviary.lanes.d_personal.adapter import LaneDConfig, parse_all
from aviary.render.run import render_run
from aviary.schema.records import ConversationRecord
from aviary.targets import Target
from aviary.teacher.fake import FakeTeacherClient
from aviary.teacher.prompts import PromptSet
from aviary.teacher.roster import Roster

REPO = Path(__file__).resolve().parents[1]
OWNER, FRIEND = "Sam", "Rio"


def _line(conv, ts, sender, text):
    return json.dumps({"conversation_id": conv, "ts": ts, "sender": sender, "text": text})


def _roster() -> Roster:
    return Roster(
        teachers=[
            {
                "id": "glm-5-20260430",
                "provider": "zhipu",
                "route": "direct",
                "base_url": "x",
                "wire_model": "glm",
                "api_key_env": "K",
            },
            {
                "id": "deepseek-v4-flash-20260610",
                "provider": "deepseek",
                "route": "direct",
                "base_url": "x",
                "wire_model": "ds",
                "api_key_env": "K",
            },
        ],
        assignments={
            "judge": {"primary": "glm-5-20260430", "secondary": "deepseek-v4-flash-20260610"},
            "harmonizer": {"primary": "deepseek-v4-flash-20260610"},
        },
    )


def test_sorcha_target_synthetic_run_end_to_end(tmp_path):
    target = Target.load("sorcha-v1")
    # Lanes a/c joined 2026-07-22; the invariant was never "only b+d", it is that
    # nothing Olivia-voiced may enter this render.
    assert target.lanes == ["a", "b", "c", "d"]
    assert target.persona == "sorcha" and target.persona_speaker == "Sorcha"
    assert target.harmonize == {}  # her voice comes from generation, not paraphrase

    # --- adapter: synthetic export with a march (train) and may (holdout) month ---
    export = tmp_path / "export.jsonl"
    export.write_text(
        "\n".join(
            [
                _line("c1", "2026-03-14T21:07:03Z", FRIEND, "the trail was pure mud today"),
                _line("c1", "2026-03-14T21:09:41Z", OWNER, "and yet you sound delighted"),
                _line("c1", "2026-03-14T21:10:02Z", FRIEND, "I lost a shoe. delighted is one word"),
                _line("c1", "2026-03-14T21:12:19Z", OWNER, "the swamp keeps what it wants"),
                _line("c2", "2026-05-02T10:00:00Z", FRIEND, "ok may plans. cabin or coast?"),
                _line(
                    "c2", "2026-05-02T10:03:00Z", OWNER, "coast. the cabin has a raccoon regime now"
                ),
            ]
        )
        + "\n"
    )
    cfg = LaneDConfig.model_validate(
        {
            "sources": [
                {
                    "id": "chat",
                    "parser": "generic_jsonl",
                    "path": str(export),
                    "assistant_sender": OWNER,
                }
            ],
            "holdout_families": ["chat/2026-05"],
        }
    )
    records = list(parse_all(cfg, run_id="sorcha-accept"))
    assert {r.provenance.family for r in records} == {"chat/2026-03", "chat/2026-05"}

    store = RunStore("sorcha-accept", root=tmp_path / "data")
    write_jsonl(store.raw("d"), records)

    # --- gates: real verifiers/rubrics/policies, fake teacher (offline) ---
    rubrics = {
        "d": __import__("aviary.gates.judge", fromlist=["Rubric"]).Rubric.load(
            REPO / target.lane_rubrics["d"]
        )
    }
    policies = load_lane_policies(REPO / "gates" / "scrub" / "lane_policy.yaml")
    pseudo = Pseudonymizer(
        PseudonymRules(preserve=[OWNER], name_patterns=[FRIEND]),
        store.root / "scrub" / "pseudonyms.json",
    )
    policies["d"].transform = pseudo.apply
    stats = run_gates(
        store,
        default_resolver({}),  # lane d -> verifiers/laned/*.py
        rubrics,
        load_patterns(
            REPO / "gates" / "scrub" / "denylist.yaml",
            REPO / "gates" / "scrub" / "pii_patterns.yaml",
        ),
        _roster(),
        FakeTeacherClient(
            script=lambda r: json.dumps(
                {
                    "scores": {
                        "substance": 5,
                        "relational_texture": 4,
                        "coherence": 5,
                        "no_impersonation": 5,
                        "safe_to_train": 5,
                    }
                }
            )
        ),
        PromptSet({"judge_prompt": "judge {axes}", "harmonize_prompt": "unused"}),
        persona_speaker=target.persona_speaker,
        harmonize_policy={k: tuple(v) for k, v in target.harmonize.items()},
        scrub_policies=policies,
    )
    assert stats.kept == len(records)
    kept = list(read_jsonl(store.gated_kept(), ConversationRecord))
    assert all(
        m.speaker in (OWNER, "Person-1") for r in kept for m in r.messages
    )  # third party pseudonymized, owner preserved
    assert not any(r.gate_state.harmonized for r in kept)  # sorcha harmonizes nothing yet

    # --- render: target-bounded, holdout enforced, zero Olivia ---
    counts = render_run(
        store,
        eval_param_fraction=0.0,
        split_seed=1,
        dpo_min_margin=1.0,
        include_lanes=set(target.lanes),
    )
    assert counts.train > 0
    rendered_dir = store.root / "rendered"
    all_text = "".join(p.read_text() for p in rendered_dir.glob("*.jsonl"))
    assert "olivia" not in all_text.lower()  # the whole point
    train_text = (rendered_dir / "train_with_thoughts.jsonl").read_text()
    assert "raccoon regime" not in train_text  # holdout month never trains
    assert "swamp" in train_text  # train month renders

    # --- stats: lane D flows through without a task bank ---
    from aviary.stats import compute_stats

    s = compute_stats(store, [])
    assert s is not None
