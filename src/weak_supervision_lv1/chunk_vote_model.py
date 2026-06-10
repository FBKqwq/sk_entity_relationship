"""Lv1 chunk-label existence vote models."""

from dataclasses import dataclass, field
from typing import Any

from src.weak_supervision.common.lf_output import LFOutput
from src.weak_supervision.common.voting import sigmoid, weighted_vote_score


DEFAULT_LV1_VOTE_CONFIG: dict[str, Any] = {
    "method": "manual_weighted_vote",
    "alpha": 5.0,
    "threshold": {
        "sub_diseases": 0.35,
        "symptoms": 0.30,
        "tests": 0.30,
        "treatments": 0.35,
        "plans": 0.35,
    },
    "lf_weights": {
        "lv1_chunk_medical_pattern": 1.2,
        "lv1_chunk_prompted_llm": 1.1,
        "lv1_chunk_prompted_llm_semantic_presence": 0.9,
        "lv1_chunk_prompted_llm_evidence_anchor": 0.9,
        "lv1_chunk_prompted_llm_boundary_count": 0.8,
        "lv1_chunk_dictionary": 0.9,
        "lv1_chunk_regex_indicator": 0.6,
        "lv1_chunk_section_prior": 0.4,
    },
    "primary_lfs": [
        "lv1_chunk_medical_pattern",
        "lv1_chunk_prompted_llm",
        "lv1_chunk_prompted_llm_semantic_presence",
        "lv1_chunk_prompted_llm_evidence_anchor",
        "lv1_chunk_prompted_llm_boundary_count",
        "lv1_chunk_dictionary",
    ],
    "auxiliary_lfs": [
        "lv1_chunk_regex_indicator",
        "lv1_chunk_section_prior",
    ],
}


@dataclass(frozen=True)
class Lv1VoteDecision:
    """Interpretable Lv1 decision for one chunk-label pair."""

    label: str
    present: bool
    confidence: float
    vote_score: float
    threshold: float
    status: str
    count: int = 0
    supporting_lfs: list[str] = field(default_factory=list)
    opposing_lfs: list[str] = field(default_factory=list)
    abstained_lfs: list[str] = field(default_factory=list)
    evidence_texts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serializable decision record."""

        return {
            "label": self.label,
            "present": self.present,
            "confidence": self.confidence,
            "vote_score": self.vote_score,
            "threshold": self.threshold,
            "status": self.status,
            "count": self.count,
            "supporting_lfs": self.supporting_lfs,
            "opposing_lfs": self.opposing_lfs,
            "abstained_lfs": self.abstained_lfs,
            "evidence_texts": self.evidence_texts,
            "metadata": self.metadata,
        }


def majority_vote_binary(votes: list[int]) -> tuple[bool, float]:
    """Return `present` and confidence for binary majority voting."""

    positive = sum(1 for vote in votes if vote == 1)
    negative = sum(1 for vote in votes if vote == -1)
    total = positive + negative
    if total == 0:
        return False, 0.0
    return positive > negative, positive / total


def manual_weighted_vote_binary(
    votes: list[int],
    confidences: list[float],
    lf_weights: list[float],
    threshold: float = 0.25,
    alpha: float = 3.0,
) -> dict[str, float | bool]:
    """Return an interpretable weighted-vote decision for one chunk-label."""

    score = weighted_vote_score(votes, confidences, lf_weights)
    return {
        "present": score >= threshold,
        "confidence": sigmoid(alpha * score),
        "vote_score": score,
    }


def lv1_vote_config_from_project_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge project YAML config with default Lv1 vote settings."""

    merged = {
        "method": DEFAULT_LV1_VOTE_CONFIG["method"],
        "alpha": DEFAULT_LV1_VOTE_CONFIG["alpha"],
        "threshold": dict(DEFAULT_LV1_VOTE_CONFIG["threshold"]),
        "lf_weights": dict(DEFAULT_LV1_VOTE_CONFIG["lf_weights"]),
        "primary_lfs": list(DEFAULT_LV1_VOTE_CONFIG["primary_lfs"]),
        "auxiliary_lfs": list(DEFAULT_LV1_VOTE_CONFIG["auxiliary_lfs"]),
    }
    if not config:
        return merged
    raw = config.get("lv1_vote_model", {})
    if not isinstance(raw, dict):
        return merged
    if raw.get("method"):
        merged["method"] = str(raw["method"])
    if raw.get("alpha") is not None:
        merged["alpha"] = float(raw["alpha"])
    for key in ("threshold", "lf_weights"):
        value = raw.get(key)
        if isinstance(value, dict):
            merged[key].update({str(name): float(score) for name, score in value.items()})
    for key in ("primary_lfs", "auxiliary_lfs"):
        value = raw.get(key)
        if isinstance(value, list):
            merged[key] = [str(name) for name in value]
    return merged


def manual_weighted_vote_label(
    label: str,
    lf_outputs: list[LFOutput],
    *,
    vote_config: dict[str, Any] | None = None,
) -> Lv1VoteDecision:
    """Fuse all LF outputs for one chunk-label pair using configured weights.

    Abstaining LFs remain in the denominator when they are configured. This
    keeps a single weak auxiliary vote from becoming a high-confidence result.
    """

    config = lv1_vote_config_from_project_config({"lv1_vote_model": vote_config or {}})
    weights: dict[str, float] = config["lf_weights"]
    threshold = float(config["threshold"].get(label, 0.35))
    alpha = float(config["alpha"])
    primary_lfs = set(config["primary_lfs"])
    auxiliary_lfs = set(config["auxiliary_lfs"])

    by_name = {output.lf_name: output for output in lf_outputs if output.label == label}
    configured_names = [name for name in weights if name in by_name]
    if not configured_names:
        configured_names = list(weights)
    denominator = sum(abs(weights[name]) for name in configured_names)
    if denominator <= 0:
        denominator = 1.0

    raw_score = 0.0
    supporting_lfs: list[str] = []
    opposing_lfs: list[str] = []
    abstained_lfs: list[str] = []
    evidence_texts: list[str] = []
    positive_counts: list[int] = []

    for lf_name in configured_names:
        weight = weights[lf_name]
        output = by_name.get(lf_name)
        if output is None or output.vote == 0:
            abstained_lfs.append(lf_name)
            continue
        bounded_confidence = max(0.0, min(1.0, output.confidence))
        raw_score += output.vote * bounded_confidence * weight
        if output.vote > 0:
            supporting_lfs.append(lf_name)
            positive_counts.append(output.count)
            evidence_texts.extend(span.text for span in output.evidence)
        elif output.vote < 0:
            opposing_lfs.append(lf_name)

    vote_score = raw_score / denominator
    present = vote_score >= threshold
    confidence = sigmoid(alpha * (vote_score - threshold)) if supporting_lfs else 0.0

    has_primary_support = any(lf_name in primary_lfs for lf_name in supporting_lfs)
    only_auxiliary_support = bool(supporting_lfs) and all(
        lf_name in auxiliary_lfs for lf_name in supporting_lfs
    )
    if not present:
        status = "rejected"
    elif only_auxiliary_support or not has_primary_support:
        status = "weak"
    else:
        status = "accepted"

    return Lv1VoteDecision(
        label=label,
        present=present,
        confidence=confidence,
        vote_score=vote_score,
        threshold=threshold,
        status=status,
        count=max(positive_counts) if positive_counts else 0,
        supporting_lfs=supporting_lfs,
        opposing_lfs=opposing_lfs,
        abstained_lfs=abstained_lfs,
        evidence_texts=evidence_texts[:20],
        metadata={
            "method": config["method"],
            "has_primary_support": has_primary_support,
            "only_auxiliary_support": only_auxiliary_support,
        },
    )
