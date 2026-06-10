"""Shared voting helpers for weak-supervision fusion."""

from math import exp


def sigmoid(value: float) -> float:
    """Return a numerically stable sigmoid score."""

    if value >= 0:
        z = exp(-value)
        return 1 / (1 + z)
    z = exp(value)
    return z / (1 + z)


def weighted_vote_score(votes: list[int], confidences: list[float], weights: list[float]) -> float:
    """Compute a normalized weighted vote score in [-1, 1]."""

    raw_score = 0.0
    norm = 0.0
    for vote, confidence, weight in zip(votes, confidences, weights, strict=False):
        if vote == 0:
            continue
        mass = confidence * weight
        raw_score += vote * mass
        norm += abs(mass)
    if norm == 0:
        return 0.0
    return raw_score / norm
