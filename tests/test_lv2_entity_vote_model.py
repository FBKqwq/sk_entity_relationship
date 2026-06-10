"""Tests for Lv2 entity vote models."""

from src.weak_supervision_lv2.entity_vote_model import softmax


def test_softmax_returns_probabilities() -> None:
    probs = softmax({"a": 1.0, "b": 2.0})
    assert set(probs) == {"a", "b"}
    assert abs(sum(probs.values()) - 1.0) < 1e-9
