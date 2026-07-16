"""Shared lane utilities: JSON-constrained teacher calls with repair-retry."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

from aviary.teacher.client import ChatRequest, TeacherClient

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


class ExtractionError(RuntimeError):
    pass


def strip_fences(text: str) -> str:
    return _FENCE.sub("", text).strip()


def call_json[M: BaseModel](
    client: TeacherClient,
    req: ChatRequest,
    model_cls: type[M],
    lane: str,
    max_repairs: int = 2,
) -> M:
    """Call the teacher expecting a JSON object; on parse/validation failure, feed the
    error back for repair up to max_repairs times (CoSER-style repair-and-retry)."""
    request = req.model_copy(update={"response_json": True})
    for attempt in range(max_repairs + 1):
        resp = client.complete(request, lane)
        raw = strip_fences(resp.text)
        try:
            return model_cls.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt == max_repairs:
                raise ExtractionError(f"unparseable JSON after {max_repairs} repairs: {e}") from e
            request = request.model_copy(
                update={
                    "messages": [
                        *req.messages,
                        {"role": "assistant", "content": resp.text},
                        {
                            "role": "user",
                            "content": (
                                "Your previous reply was not valid for the required JSON schema: "
                                f"{e}. Reply again with ONLY the corrected JSON object."
                            ),
                        },
                    ]
                }
            )
    raise AssertionError("unreachable")
