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
            # Echo the failed reply back for context — but ONLY when it's non-empty.
            # Reasoning teachers (e.g. Kimi) sometimes return empty content; echoing
            # `{"role": "assistant", "content": ""}` is rejected by strict providers
            # (Moonshot 400 "assistant must not be empty"), which would turn a
            # droppable bad scene into a run-killing TeacherError. Omitting the echo
            # keeps the repair request valid so the loop can exhaust its retries and
            # raise a *skippable* ExtractionError instead.
            empty = not resp.text.strip()
            echo = [] if empty else [{"role": "assistant", "content": resp.text}]
            note = "was empty" if empty else f"was not valid for the required JSON schema: {e}"
            request = request.model_copy(
                update={
                    "messages": [
                        # Accumulate on the running request so a 2nd repair still
                        # sees the 1st failed attempt (not just the most recent one).
                        *request.messages,
                        *echo,
                        {
                            "role": "user",
                            "content": (
                                f"Your previous reply {note}. "
                                "Reply again with ONLY the corrected JSON object."
                            ),
                        },
                    ]
                }
            )
    raise AssertionError("unreachable")
