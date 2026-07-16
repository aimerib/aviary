from __future__ import annotations

from pathlib import Path

import pytest

from aviary.teacher.cache import ResponseCache, request_key
from aviary.teacher.client import ChatRequest, ChatResponse, Usage
from aviary.teacher.cost import CostLedger
from aviary.teacher.fake import FakeTeacherClient
from aviary.teacher.prompts import PromptSet, PromptSetError
from aviary.teacher.roster import Roster

ROSTER = Roster(
    teachers=[
        {
            "id": "deepseek-v4-flash-20260610",
            "provider": "deepseek",
            "route": "direct",
            "base_url": "https://api.deepseek.com/v1",
            "wire_model": "deepseek-chat-20260610",
            "api_key_env": "DEEPSEEK_API_KEY",
        },
        {
            "id": "glm-5-20260430",
            "provider": "zhipu",
            "route": "direct",
            "base_url": "https://api.z.ai/api/paas/v4",
            "wire_model": "glm-5-20260430",
            "api_key_env": "GLM_API_KEY",
            "api_key_env_fallback": "ZAI_API_KEY",
        },
    ],
    assignments={
        "lane_b": {"dialogue": "deepseek-v4-flash-20260610"},
        "judge": {"primary": "glm-5-20260430", "secondary": "deepseek-v4-flash-20260610"},
    },
)


def req(text: str = "hi") -> ChatRequest:
    return ChatRequest(
        model="deepseek-v4-flash-20260610",
        system="sys",
        messages=[{"role": "user", "content": text}],
    )


def test_undated_model_id_rejected():
    with pytest.raises(ValueError, match="dated snapshot"):
        Roster(
            teachers=[
                {
                    "id": "deepseek-latest",
                    "provider": "deepseek",
                    "route": "direct",
                    "base_url": "x",
                    "wire_model": "deepseek-latest",
                    "api_key_env": "K",
                }
            ],
            assignments={},
        )


def test_cross_vendor_judge():
    judge = ROSTER.judge_for("deepseek-v4-flash-20260610")
    assert judge.provider == "zhipu"
    judge2 = ROSTER.judge_for("glm-5-20260430")
    assert judge2.provider == "deepseek"


def test_cache_roundtrip(tmp_path: Path):
    cache = ResponseCache(tmp_path)
    r = req()
    assert cache.get(r) is None
    resp = ChatResponse(text="hello", usage=Usage(input_tokens=10, output_tokens=5), model=r.model)
    cache.put(r, resp)
    hit = cache.get(r)
    assert hit is not None and hit.text == "hello"
    assert request_key(r) == request_key(req())
    assert request_key(r) != request_key(req("other"))


def test_fake_client_replays_fixture(tmp_path: Path):
    cache = ResponseCache(tmp_path)
    r = req()
    cache.put(r, ChatResponse(text="recorded", model=r.model))
    fake = FakeTeacherClient(fixtures_dir=tmp_path)
    assert fake.complete(r).text == "recorded"
    with pytest.raises(AssertionError):
        fake.complete(req("unrecorded"))


def test_cost_ledger(tmp_path: Path):
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(
        "deepseek-v4-flash-20260610:\n  input_per_mtok: 0.30\n  output_per_mtok: 1.20\n"
    )
    ledger = CostLedger(pricing)
    ledger.add("deepseek-v4-flash-20260610", "b", 1_000_000, 500_000)
    spend = ledger.to_spend()
    assert spend.total == pytest.approx(0.9)
    assert spend.by_lane["b"] == pytest.approx(0.9)


def test_prompt_set_hash_discipline():
    ps1 = PromptSet({"judge": "You are a judge.", "sys": "Olivia."})
    ps2 = PromptSet({"judge": "You are a judge.", "sys": "Olivia."})
    assert ps1.hash == ps2.hash
    ps1.assert_hash(ps2.hash)
    edited = PromptSet({"judge": "You are a JUDGE.", "sys": "Olivia."})
    with pytest.raises(PromptSetError, match="new run"):
        edited.assert_hash(ps1.hash)
    with pytest.raises(PromptSetError):
        ps1["missing"]
