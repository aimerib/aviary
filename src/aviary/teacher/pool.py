"""Bounded parallel execution of teacher calls, capped per provider."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

from aviary.teacher.client import ChatRequest, ChatResponse, TeacherClient
from aviary.teacher.roster import Roster


class TeacherPool:
    def __init__(self, client: TeacherClient, roster: Roster, max_workers: int = 16):
        self.client = client
        self.roster = roster
        self.max_workers = max_workers
        self._sems: dict[str, threading.Semaphore] = {}
        for t in roster.teachers:
            self._sems.setdefault(t.provider, threading.Semaphore(t.max_concurrency))

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
