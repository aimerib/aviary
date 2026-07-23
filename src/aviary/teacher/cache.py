"""On-disk response cache. Key = sha256(model + canonical request). Provides run
resumability, and its files are the recorded-fixture format for FakeTeacherClient."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from aviary.hashing import canonical_json, sha256_text

if TYPE_CHECKING:
    from aviary.teacher.client import ChatRequest, ChatResponse
    from aviary.teacher.roster import TeacherRoute


def route_fingerprint(route: TeacherRoute | None) -> dict:
    """The parts of a route that change the bytes actually sent to the provider.

    Not cosmetic: `json_extra_body` carries the thinking-off controls, and it lives
    on the ROUTE, not the request. Without it in the key, turning Kimi's reasoning
    off changed nothing — the 100 empty responses recorded before the fix were
    replayed verbatim, so the run failed identically and looked like the fix had
    not worked. A cache keyed on less than the request it replays is a cache that
    lies.
    """
    if route is None:
        return {}
    return {"wire_model": route.wire_model, "json_extra_body": route.json_extra_body}


def request_key(req: ChatRequest, route: TeacherRoute | None = None) -> str:
    payload = req.model_dump(mode="json")
    fingerprint = route_fingerprint(route)
    if fingerprint:
        payload = {**payload, "_route": fingerprint}
    return sha256_text(canonical_json(payload))


class ResponseCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, req: ChatRequest, route: TeacherRoute | None = None) -> ChatResponse | None:
        from aviary.teacher.client import ChatResponse

        path = self._path(request_key(req, route))
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
            return ChatResponse.model_validate(payload["response"])
        except (json.JSONDecodeError, KeyError, ValidationError):
            # A truncated/corrupt file (e.g. crash mid-write before the atomic put
            # below existed, or a partial disk) is a cache MISS, not a fatal error.
            return None

    def put(self, req: ChatRequest, resp: ChatResponse, route: TeacherRoute | None = None) -> None:
        # An empty completion is a FAILURE, not a result. Caching one freezes the
        # failure permanently and invisibly: 3,991 empty responses (6.2% of the
        # cache) were being replayed on every rerun, 3,598 of them lane B scenes
        # that could therefore never succeed no matter how often the run repeated.
        if not (resp.text or "").strip():
            return
        key = request_key(req, route)
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(
            {"request": req.model_dump(mode="json"), "response": resp.model_dump(mode="json")},
            ensure_ascii=False,
            indent=2,
        )
        # Write-temp-then-rename so an interrupted write can never leave a corrupt
        # file at the key path that poisons the next run's resume. mkstemp keeps the
        # temp unique so concurrent writers of the same key don't clobber each other.
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
