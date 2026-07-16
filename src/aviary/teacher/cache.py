"""On-disk response cache. Key = sha256(model + canonical request). Provides run
resumability, and its files are the recorded-fixture format for FakeTeacherClient."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from aviary.hashing import canonical_json, sha256_text

if TYPE_CHECKING:
    from aviary.teacher.client import ChatRequest, ChatResponse


def request_key(req: ChatRequest) -> str:
    return sha256_text(canonical_json(req.model_dump(mode="json")))


class ResponseCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, req: ChatRequest) -> ChatResponse | None:
        from aviary.teacher.client import ChatResponse

        path = self._path(request_key(req))
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        return ChatResponse.model_validate(payload["response"])

    def put(self, req: ChatRequest, resp: ChatResponse) -> None:
        key = request_key(req)
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"request": req.model_dump(mode="json"), "response": resp.model_dump(mode="json")},
                ensure_ascii=False,
                indent=2,
            )
        )
