"""Teacher API client. httpx-only, sync, OpenAI-compatible wire (all current routes:
DeepSeek direct, GLM/Z.ai direct, Kimi via OpenRouter).

Cache discipline: the frozen PromptSet hash rides along on every request; the
on-disk ResponseCache makes runs resumable and doubles as the fixture format
replayed by FakeTeacherClient in tests.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field

from aviary.teacher.cache import ResponseCache
from aviary.teacher.cost import CostLedger
from aviary.teacher.roster import Roster, TeacherRoute

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class ChatRequest(BaseModel):
    model: str  # dated snapshot id from the roster
    system: str
    messages: list[dict[str, Any]]  # [{role, content}]
    temperature: float = 0.8
    max_tokens: int = 2048
    response_json: bool = False  # ask for a JSON object response
    reasoning_off: bool = False  # disable the model's thinking mode (direct output)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ChatResponse(BaseModel):
    text: str
    usage: Usage = Field(default_factory=Usage)
    model: str
    cached: bool = False
    # OpenAI-compatible stop reason. "length" means the reply was CUT OFF at
    # max_tokens, which is a different failure from a malformed reply and needs a
    # different remedy (more budget, not a repair round). Defaulted so cached
    # responses written before this field existed still load.
    finish_reason: str = ""


class TeacherClient(Protocol):
    def complete(self, req: ChatRequest, lane: str = "") -> ChatResponse: ...


class TeacherError(RuntimeError):
    pass


class HttpTeacherClient:
    def __init__(
        self,
        roster: Roster,
        cache: ResponseCache | None = None,
        ledger: CostLedger | None = None,
        max_attempts: int = 5,
        timeout: float = 300.0,
    ):
        self.roster = roster
        self.cache = cache
        self.ledger = ledger
        self.max_attempts = max_attempts
        self._http = httpx.Client(timeout=timeout)

    def complete(self, req: ChatRequest, lane: str = "") -> ChatResponse:
        if self.cache is not None:
            hit = self.cache.get(req, self.roster.route_for(req.model))
            if hit is not None:
                return hit.model_copy(update={"cached": True})

        route = self.roster.route_for(req.model)
        resp = self._call_with_retry(route, req)
        if self.ledger is not None:
            self.ledger.add(req.model, lane, resp.usage.input_tokens, resp.usage.output_tokens)
        if self.cache is not None:
            self.cache.put(req, resp, self.roster.route_for(req.model))
        return resp

    def _call_with_retry(self, route: TeacherRoute, req: ChatRequest) -> ChatResponse:
        last: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                return self._openai_compat(route, req)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status not in RETRYABLE_STATUS:
                    raise TeacherError(f"{route.provider} {status}: {e.response.text[:500]}") from e
                last = e
                retry_after = e.response.headers.get("retry-after")
                delay = float(retry_after) if retry_after else min(2**attempt, 30)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last = e
                delay = min(2**attempt, 30)
            time.sleep(delay + random.random())
        raise TeacherError(f"exhausted {self.max_attempts} attempts for {req.model}") from last

    def _openai_compat(self, route: TeacherRoute, req: ChatRequest) -> ChatResponse:
        key = os.environ.get(route.api_key_env) or (
            os.environ.get(route.api_key_env_fallback) if route.api_key_env_fallback else None
        )
        if not key:
            raise TeacherError(f"missing API key: set {route.api_key_env}")
        payload: dict[str, Any] = {
            "model": route.wire_model,
            "messages": [{"role": "system", "content": req.system}, *req.messages],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }
        if req.response_json:
            payload["response_format"] = {"type": "json_object"}
        # route.json_extra_body carries the provider's thinking-off control. Apply it
        # on JSON extraction (structured output wants no reasoning) OR when the caller
        # explicitly asks for direct output (reasoning_off) — e.g. lane C roleplay
        # turns, which must respond in-the-moment, not reason then reply.
        if req.response_json or req.reasoning_off:
            payload.update(route.json_extra_body)
        r = self._http.post(
            f"{route.base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {key}"},
        )
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage") or {}
        return ChatResponse(
            text=data["choices"][0]["message"]["content"] or "",
            usage=Usage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            ),
            model=req.model,
            finish_reason=data["choices"][0].get("finish_reason") or "",
        )
