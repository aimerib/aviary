"""Cost tracking: usage x pricing.yaml -> manifest spend_usd."""

from __future__ import annotations

import threading
from pathlib import Path

import yaml

from aviary.schema.manifest import Spend


class CostLedger:
    def __init__(self, pricing_path: Path | None = None):
        # pricing.yaml: {<teacher_id>: {input_per_mtok: float, output_per_mtok: float}}
        self.pricing: dict[str, dict[str, float]] = (
            yaml.safe_load(pricing_path.read_text()) if pricing_path else {}
        ) or {}
        self._lock = threading.Lock()
        self.by_teacher: dict[str, float] = {}
        self.by_lane: dict[str, float] = {}

    def add(self, teacher_id: str, lane: str, input_tokens: int, output_tokens: int) -> None:
        price = self.pricing.get(teacher_id, {})
        usd = (
            input_tokens * price.get("input_per_mtok", 0.0)
            + output_tokens * price.get("output_per_mtok", 0.0)
        ) / 1_000_000
        with self._lock:
            self.by_teacher[teacher_id] = self.by_teacher.get(teacher_id, 0.0) + usd
            if lane:
                self.by_lane[lane] = self.by_lane.get(lane, 0.0) + usd

    @property
    def total(self) -> float:
        return sum(self.by_teacher.values())

    def to_spend(self) -> Spend:
        return Spend(
            total=round(self.total, 4),
            by_teacher={k: round(v, 4) for k, v in self.by_teacher.items()},
            by_lane={k: round(v, 4) for k, v in self.by_lane.items()},
        )
