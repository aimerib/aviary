import pytest


def test_lane_subset_defaults_to_all(monkeypatch):
    from aviary.orchestrate import _lane_subset

    monkeypatch.delenv("AVIARY_LANES", raising=False)
    assert _lane_subset(["a", "b", "c", "d"]) == ["a", "b", "c", "d"]


def test_lane_subset_narrows_and_preserves_config_order(monkeypatch):
    from aviary.orchestrate import _lane_subset

    monkeypatch.setenv("AVIARY_LANES", "d,c")  # deliberately out of order
    assert _lane_subset(["a", "b", "c", "d"]) == ["c", "d"]


def test_lane_subset_refuses_to_widen_past_the_target(monkeypatch):
    # The target's lane set is a contract (target-bounded renders, lane D
    # exemption). An env override that could ADD a lane would breach it.
    from aviary.orchestrate import _lane_subset

    monkeypatch.setenv("AVIARY_LANES", "a")
    with pytest.raises(ValueError, match="never widen"):
        _lane_subset(["b", "d"])
