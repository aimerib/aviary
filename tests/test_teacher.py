from __future__ import annotations

from pathlib import Path

import pytest

from aviary.teacher.cache import ResponseCache, request_key
from aviary.teacher.client import ChatRequest, ChatResponse, TeacherError, Usage
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


def _roster_with_id(model_id: str) -> Roster:
    return Roster(
        teachers=[
            {
                "id": model_id,
                "provider": "deepseek",
                "route": "direct",
                "base_url": "x",
                "wire_model": model_id,
                "api_key_env": "K",
            }
        ],
        assignments={},
    )


def test_undated_model_id_rejected():
    with pytest.raises(ValueError, match="dated snapshot"):
        _roster_with_id("deepseek-latest")


def test_truncated_and_bare_model_ids_rejected():
    # The old blacklist accepted these; the positive dated-suffix check must not.
    for bad in ("deepseek-v4-flash-2026061", "deepseek-v4-flash", "deepseek-v4-flash-19991231x"):
        with pytest.raises(ValueError, match="dated snapshot"):
            _roster_with_id(bad)


def test_dated_snapshot_ids_accepted():
    for good in ("deepseek-v4-flash-20260610", "glm-5-20260430"):
        assert _roster_with_id(good).teachers[0].id == good


def test_wire_model_is_not_dated_validated():
    # id/wire_model are decoupled: `id` is the dated provenance pin, `wire_model`
    # carries the provider's real (rolling, undated) name. A dated `id` with a bare
    # rolling wire name must validate.
    roster = Roster(
        teachers=[
            {
                "id": "deepseek-v4-flash-20260717",
                "provider": "deepseek",
                "route": "direct",
                "base_url": "x",
                "wire_model": "deepseek-v4-flash",  # undated on purpose
                "api_key_env": "K",
            }
        ],
        assignments={},
    )
    assert roster.teachers[0].wire_model == "deepseek-v4-flash"


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


def test_cache_corrupt_file_is_miss(tmp_path: Path):
    # A truncated/corrupt cache file (crash mid-write) must degrade to a miss, not
    # crash the next run on JSONDecodeError.
    cache = ResponseCache(tmp_path)
    r = req()
    path = cache._path(request_key(r))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"response": {"text": "half')  # truncated JSON
    assert cache.get(r) is None
    # And it can be overwritten cleanly afterwards.
    cache.put(r, ChatResponse(text="ok", model=r.model))
    hit = cache.get(r)
    assert hit is not None and hit.text == "ok"


def test_fake_client_replays_fixture(tmp_path: Path):
    cache = ResponseCache(tmp_path)
    r = req()
    cache.put(r, ChatResponse(text="recorded", model=r.model))
    fake = FakeTeacherClient(fixtures_dir=tmp_path)
    assert fake.complete(r).text == "recorded"
    with pytest.raises(TeacherError):
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


# --- judge load balancing ----------------------------------------------------


def _balanced_roster() -> Roster:
    def t(tid, provider):
        return {
            "id": tid,
            "provider": provider,
            "route": "direct",
            "base_url": "x",
            "wire_model": "w",
            "api_key_env": "K",
        }

    return Roster(
        teachers=[
            t("deepseek-v4-pro-20260717", "deepseek"),
            t("glm-5.2-20260717", "zhipu"),
            t("kimi-k3-20260717", "moonshot"),
        ],
        assignments={
            "judge": {
                "primary": "glm-5.2-20260717",
                "secondary": "kimi-k3-20260717",
                "tertiary": "deepseek-v4-pro-20260717",
            }
        },
    )


def test_judge_never_shares_a_vendor_with_the_generator():
    roster = _balanced_roster()
    for key in (f"rec-{i}" for i in range(50)):
        judge = roster.judge_for("deepseek-v4-pro-20260717", key)
        assert judge.provider != "deepseek"


def test_judge_load_is_spread_not_first_match():
    """Declaration order was acting as a priority list: DeepSeek generates most
    records, so GLM won every time and Kimi was structurally unreachable — one
    vendor's rate limit became the pipeline's throughput ceiling."""
    roster = _balanced_roster()
    picked = {roster.judge_for("deepseek-v4-pro-20260717", f"rec-{i}").provider for i in range(100)}
    assert picked == {"zhipu", "moonshot"}  # both eligible vendors actually used

    counts: dict[str, int] = {}
    for i in range(400):
        p = roster.judge_for("deepseek-v4-pro-20260717", f"rec-{i}").provider
        counts[p] = counts.get(p, 0) + 1
    assert min(counts.values()) / max(counts.values()) > 0.7  # roughly even


def test_sibling_groups_share_one_judge():
    """DPO gates on chosen.overall - rejected.overall. Two vendors do not share a
    scoring scale, so siblings split across judges would create and destroy pairs
    on vendor offset rather than on quality."""
    roster = _balanced_roster()
    for group in ("grp-a", "grp-b", "grp-c", "grp-d"):
        chosen = roster.judge_for("deepseek-v4-pro-20260717", group)
        for _ in range(5):
            assert roster.judge_for("deepseek-v4-pro-20260717", group).id == chosen.id


def test_judge_routing_is_reproducible_across_instances():
    # A rerun must route identically or every cached judge response misses.
    a, b = _balanced_roster(), _balanced_roster()
    for i in range(30):
        key = f"rec-{i}"
        assert (
            a.judge_for("deepseek-v4-pro-20260717", key).id
            == b.judge_for("deepseek-v4-pro-20260717", key).id
        )


def test_empty_key_keeps_deterministic_first_match():
    roster = _balanced_roster()
    assert roster.judge_for("deepseek-v4-pro-20260717").id == "glm-5.2-20260717"


def test_no_cross_vendor_judge_is_an_error_not_a_fallback():
    roster = _balanced_roster()
    roster.assignments["judge"] = {"primary": "deepseek-v4-pro-20260717"}
    with pytest.raises(ValueError, match="no cross-vendor judge"):
        roster.judge_for("deepseek-v4-pro-20260717", "k")
