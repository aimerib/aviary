"""Offline tests for HttpTeacherClient retry/backoff/status-mapping and TeacherPool.
Uses httpx.MockTransport (no sockets) and monkeypatches sleep, so the money-spending
retry path is exercised in CI without a real provider."""

from __future__ import annotations

import httpx
import pytest

from aviary.teacher.client import ChatRequest, ChatResponse, HttpTeacherClient, TeacherError
from aviary.teacher.pool import TeacherPool
from aviary.teacher.roster import Roster

ROSTER = Roster(
    teachers=[
        {
            "id": "deepseek-v4-flash-20260610",
            "provider": "deepseek",
            "route": "direct",
            "base_url": "https://api.test/v1",
            "wire_model": "deepseek-v4-flash-20260610",
            "api_key_env": "DEEPSEEK_API_KEY",
        }
    ],
    assignments={},
)


def _req(text: str = "x") -> ChatRequest:
    return ChatRequest(
        model="deepseek-v4-flash-20260610", system="s", messages=[{"role": "user", "content": text}]
    )


def _client(handler, monkeypatch, **kw) -> HttpTeacherClient:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setattr("aviary.teacher.client.time.sleep", lambda s: _client.sleeps.append(s))
    client = HttpTeacherClient(ROSTER, **kw)
    client._http = httpx.Client(transport=httpx.MockTransport(handler))
    return client


_client.sleeps = []  # type: ignore[attr-defined]


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
    )


def test_retries_on_429_then_succeeds(monkeypatch):
    _client.sleeps = []  # type: ignore[attr-defined]
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "2"}, text="slow down")
        return _ok_response()

    client = _client(handler, monkeypatch)
    resp = client.complete(_req())
    assert resp.text == "hi"
    assert calls["n"] == 2
    assert _client.sleeps and _client.sleeps[0] >= 2  # Retry-After honored


def test_non_retryable_status_raises_without_retry(monkeypatch):
    _client.sleeps = []  # type: ignore[attr-defined]
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    client = _client(handler, monkeypatch)
    with pytest.raises(TeacherError, match="400"):
        client.complete(_req())
    assert calls["n"] == 1  # 400 is not retried


def test_exhausts_attempts_on_persistent_503(monkeypatch):
    _client.sleeps = []  # type: ignore[attr-defined]
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="unavailable")

    client = _client(handler, monkeypatch, max_attempts=3)
    with pytest.raises(TeacherError, match="exhausted"):
        client.complete(_req())
    assert calls["n"] == 3


def test_transport_error_is_retried(monkeypatch):
    _client.sleeps = []  # type: ignore[attr-defined]
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom")
        return _ok_response()

    client = _client(handler, monkeypatch)
    assert client.complete(_req()).text == "hi"
    assert calls["n"] == 2


def test_pool_maps_and_collects_errors():
    class FlakyClient:
        def complete(self, req: ChatRequest, lane: str = "") -> ChatResponse:
            if req.messages[0]["content"] == "boom":
                raise TeacherError("nope")
            return ChatResponse(text=req.messages[0]["content"], model=req.model)

    pool = TeacherPool(FlakyClient(), ROSTER, max_workers=4)
    results = pool.map([_req("a"), _req("boom"), _req("b")], lane="x")
    assert results[0].text == "a"  # type: ignore[union-attr]
    assert isinstance(results[1], TeacherError)  # one bad item doesn't kill the batch
    assert results[2].text == "b"  # type: ignore[union-attr]


def test_pool_run_preserves_order_and_isolates_errors():
    # run() executes coarse callables concurrently; results stay in input order and a
    # raising unit becomes an Exception result, not an abort.
    pool = TeacherPool(client=None, max_workers=4)  # type: ignore[arg-type]

    def make(i):
        def fn():
            if i == 2:
                raise ValueError(f"unit {i} failed")
            return i * 10

        return fn

    results = pool.run([make(i) for i in range(5)])
    assert results[0] == 0
    assert results[1] == 10
    assert isinstance(results[2], ValueError)
    assert results[3] == 30
    assert results[4] == 40
