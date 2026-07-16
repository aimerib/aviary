"""Canonical on-disk layout for a run's data under $AVIARY_DATA_DIR/<run_id>/.

raw/       lane_a.jsonl lane_b.jsonl lane_c.jsonl   (ConversationRecord, pre-gate)
laneb/     profiles.jsonl scenes.jsonl ...          (stage checkpoints)
gated/     kept.jsonl rejected.jsonl                (post-gate; rejected retained for DPO)
rendered/  train_with_thoughts.jsonl train_no_thoughts.jsonl eval_*.jsonl nsp.jsonl dpo.jsonl
"""

from __future__ import annotations

from pathlib import Path

from aviary.paths import data_dir


class RunStore:
    def __init__(self, run_id: str, root: Path | None = None):
        self.run_id = run_id
        self.root = (root or data_dir()) / run_id

    def raw(self, lane: str) -> Path:
        return self.root / "raw" / f"lane_{lane}.jsonl"

    def raw_files(self) -> list[Path]:
        d = self.root / "raw"
        return sorted(d.glob("lane_*.jsonl")) if d.exists() else []

    def stage(self, lane: str, name: str) -> Path:
        return self.root / f"lane{lane}" / f"{name}.jsonl"

    def gated_kept(self) -> Path:
        return self.root / "gated" / "kept.jsonl"

    def gated_rejected(self) -> Path:
        return self.root / "gated" / "rejected.jsonl"

    def rendered(self, name: str) -> Path:
        return self.root / "rendered" / f"{name}.jsonl"

    def hermes_out(self) -> Path:
        return self.root / "hermes"
