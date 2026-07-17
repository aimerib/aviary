"""Cost tracking: usage x pricing.yaml -> manifest spend_usd."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import yaml

from aviary.schema.manifest import Spend

log = logging.getLogger(__name__)

_REQUIRED_PRICE_FIELDS = ("input_per_mtok", "output_per_mtok")


class CostLedger:
    def __init__(self, pricing_path: Path | None = None):
        # pricing.yaml: {<teacher_id>: {input_per_mtok: float, output_per_mtok: float}}
        self.pricing: dict[str, dict[str, float]] = (
            yaml.safe_load(pricing_path.read_text()) if pricing_path else {}
        ) or {}
        self._lock = threading.Lock()
        self.by_teacher: dict[str, float] = {}
        self.by_lane: dict[str, float] = {}
        self._warned: set[str] = set()

    def _warn_uncosted(self, teacher_id: str, price: dict | None) -> None:
        # Warn ONCE per teacher rather than raise: an in-flight paid run must not
        # abort over telemetry, but a missing entry / misspelled field silently
        # reads $0 and corrupts the spend_usd provenance — never let it be silent.
        with self._lock:
            if teacher_id in self._warned:
                return
            self._warned.add(teacher_id)
        if price is None:
            log.warning(
                "cost: no pricing entry for teacher %r — its spend reads $0 and "
                "understates manifest provenance (check pricing.yaml)",
                teacher_id,
            )
        else:
            missing = [f for f in _REQUIRED_PRICE_FIELDS if f not in price]
            log.warning(
                "cost: pricing entry for teacher %r is missing %s — those default to 0 "
                "and understate spend (typo in pricing.yaml?)",
                teacher_id,
                missing,
            )

    def add(self, teacher_id: str, lane: str, input_tokens: int, output_tokens: int) -> None:
        price = self.pricing.get(teacher_id)
        if price is None or any(f not in price for f in _REQUIRED_PRICE_FIELDS):
            self._warn_uncosted(teacher_id, price)
            price = price or {}
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
