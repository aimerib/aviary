"""Test/replay client: no network, ever. Replays recorded cache files or scripted
responses. Scripted mode maps a matcher over the request to a canned reply."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from aviary.teacher.cache import ResponseCache
from aviary.teacher.client import ChatRequest, ChatResponse, Usage


class FakeTeacherClient:
    def __init__(
        self,
        fixtures_dir: Path | None = None,
        script: Callable[[ChatRequest], str] | None = None,
    ):
        self.cache = ResponseCache(fixtures_dir) if fixtures_dir else None
        self.script = script
        self.requests: list[ChatRequest] = []

    def complete(self, req: ChatRequest, lane: str = "") -> ChatResponse:
        self.requests.append(req)
        if self.cache is not None:
            hit = self.cache.get(req)
            if hit is not None:
                return hit.model_copy(update={"cached": True})
        if self.script is not None:
            return ChatResponse(
                text=self.script(req), usage=Usage(input_tokens=1, output_tokens=1), model=req.model
            )
        raise AssertionError(f"no fixture or script for request to {req.model}")
