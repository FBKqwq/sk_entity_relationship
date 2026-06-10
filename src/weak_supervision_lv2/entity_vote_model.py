"""Lv2 entity type vote models."""

from math import exp


def softmax(scores: dict[str, float]) -> dict[str, float]:
    """Compute softmax probabilities for label scores."""

    if not scores:
        return {}
    max_score = max(scores.values())
    exp_scores = {label: exp(score - max_score) for label, score in scores.items()}
    total = sum(exp_scores.values())
    return {label: value / total for label, value in exp_scores.items()}
