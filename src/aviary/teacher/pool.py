"""Bounded parallel execution of teacher work.

Two entry points:
- `map` — a batch of independent single requests, capped per provider (needs a roster).
- `run` — a batch of independent callables (a whole lane-B book or lane-C
  conversation, each an internally-sequential unit of work), capped by max_workers.

Both collect exceptions as results instead of raising, so one bad item never kills
the batch (paid pipeline). Results are returned in input order.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from aviary.teacher.client import ChatRequest, ChatResponse, TeacherClient
from aviary.teacher.roster import Roster

T = TypeVar("T")


class TeacherPool:
    def __init__(self, client: TeacherClient, roster: Roster | None = None, max_workers: int = 16):
        self.client = client
        self.roster = roster
        self.max_workers = max_workers
        self._sems: dict[str, threading.Semaphore] = {}
        if roster is not None:
            for t in roster.teachers:
                self._sems.setdefault(t.provider, threading.Semaphore(t.max_concurrency))

    @staticmethod
    def _guard(fn: Callable[[], T]) -> T | Exception:
        try:
            return fn()
        except Exception as e:  # collected, not raised: one bad item must not kill a batch
            return e

    def run(self, fns: Sequence[Callable[[], T]]) -> list[T | Exception]:
        """Run independent callables concurrently (global max_workers cap), returning
        results/exceptions in input order. Use for coarse units of work (a book, a
        conversation) that are internally sequential; set max_workers so the fan-out
        stays within providers' per-call limits."""
        results: list[T | Exception] = [None] * len(fns)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._guard, fn): i for i, fn in enumerate(fns)}
            for fut, i in futures.items():
                results[i] = fut.result()
        return results

    def _one(self, req: ChatRequest, lane: str) -> ChatResponse | Exception:
        sem = self._sems.get(self.roster.route_for(req.model).provider)
        try:
            if sem:
                with sem:
                    return self.client.complete(req, lane)
            return self.client.complete(req, lane)
        except Exception as e:  # collected, not raised: one bad item must not kill a batch
            return e

    def map(
        self,
        requests: Sequence[ChatRequest],
        lane: str,
        on_result: Callable[[int, ChatResponse | Exception], None] | None = None,
    ) -> list[ChatResponse | Exception]:
        results: list[ChatResponse | Exception] = [None] * len(requests)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._one, r, lane): i for i, r in enumerate(requests)}
            for fut, i in futures.items():
                results[i] = fut.result()
                if on_result:
                    on_result(i, results[i])
        return results
