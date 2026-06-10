"""Lv1 count prediction features, regression, and fallback rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.weak_supervision.common.lf_output import LFOutput
from src.weak_supervision_lv1.chunk_vote_model import Lv1VoteDecision


def fallback_count(*counts: int, max_count: int = 20) -> int:
    """Use the strongest rule count when no accepted count model is available."""

    return max(0, min(max_count, max(counts, default=0)))


DEFAULT_LV1_COUNT_CONFIG: dict[str, Any] = {
    "method": "manual_count_fusion",
    "min_count": 1,
    "max_count": {
        "sub_diseases": 5,
        "symptoms": 30,
        "tests": 30,
        "treatments": 10,
        "plans": 20,
    },
    "lf_weights": {
        "lv1_chunk_medical_pattern": 1.2,
        "lv1_chunk_prompted_llm": 0.9,
        "lv1_chunk_prompted_llm_semantic_presence": 0.7,
        "lv1_chunk_prompted_llm_evidence_anchor": 0.8,
        "lv1_chunk_prompted_llm_boundary_count": 1.0,
        "lv1_chunk_dictionary": 0.8,
        "lv1_chunk_regex_indicator": 0.8,
        "lv1_chunk_section_prior": 0.0,
    },
    "evidence_floor_lfs": [
        "lv1_chunk_medical_pattern",
        "lv1_chunk_dictionary",
        "lv1_chunk_regex_indicator",
    ],
}


@dataclass(frozen=True)
class Lv1CountDecision:
    """Interpretable count prediction for one chunk-label pair."""

    predicted_count: int
    count_confidence: float
    count_sources: list[str] = field(default_factory=list)
    evidence_floor: int = 0
    weighted_count: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serializable count record."""

        return {
            "predicted_count": self.predicted_count,
            "count_confidence": self.count_confidence,
            "count_sources": self.count_sources,
            "evidence_floor": self.evidence_floor,
            "weighted_count": self.weighted_count,
            "count_metadata": self.metadata,
        }


def lv1_count_config_from_project_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge project YAML config with default Lv1 count settings."""

    merged = {
        "method": DEFAULT_LV1_COUNT_CONFIG["method"],
        "min_count": DEFAULT_LV1_COUNT_CONFIG["min_count"],
        "max_count": dict(DEFAULT_LV1_COUNT_CONFIG["max_count"]),
        "lf_weights": dict(DEFAULT_LV1_COUNT_CONFIG["lf_weights"]),
        "evidence_floor_lfs": list(DEFAULT_LV1_COUNT_CONFIG["evidence_floor_lfs"]),
    }
    if not config:
        return merged
    raw = config.get("lv1_count_model", {})
    if not isinstance(raw, dict):
        return merged
    if raw.get("method"):
        merged["method"] = str(raw["method"])
    if raw.get("min_count") is not None:
        merged["min_count"] = int(raw["min_count"])
    for key in ("max_count", "lf_weights"):
        value = raw.get(key)
        if isinstance(value, dict):
            caster = int if key == "max_count" else float
            merged[key].update({str(name): caster(score) for name, score in value.items()})
    evidence_floor_lfs = raw.get("evidence_floor_lfs")
    if isinstance(evidence_floor_lfs, list):
        merged["evidence_floor_lfs"] = [str(name) for name in evidence_floor_lfs]
    return merged


def _dedupe_evidence_texts(outputs: list[LFOutput], source_lfs: set[str]) -> list[str]:
    seen: set[str] = set()
    texts: list[str] = []
    for output in outputs:
        if output.lf_name not in source_lfs or output.vote <= 0:
            continue
        for span in output.evidence:
            clean = span.text.strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            texts.append(clean)
    return texts


def manual_count_fusion(
    label: str,
    lf_outputs: list[LFOutput],
    vote_decision: Lv1VoteDecision,
    *,
    count_config: dict[str, Any] | None = None,
) -> Lv1CountDecision:
    """Predict a count for one accepted/weak chunk-label decision."""

    config = lv1_count_config_from_project_config({"lv1_count_model": count_config or {}})
    if not vote_decision.present:
        return Lv1CountDecision(
            predicted_count=0,
            count_confidence=0.0,
            metadata={"method": config["method"], "reason": "label_not_present"},
        )

    weights: dict[str, float] = config["lf_weights"]
    positive_outputs = [
        output
        for output in lf_outputs
        if output.label == label and output.vote > 0 and output.lf_name in weights
    ]
    weighted_total = 0.0
    weight_total = 0.0
    count_sources: list[str] = []
    for output in positive_outputs:
        weight = weights[output.lf_name]
        if weight <= 0 or output.count <= 0:
            continue
        mass = weight * max(0.0, min(1.0, output.confidence))
        weighted_total += output.count * mass
        weight_total += mass
        count_sources.append(output.lf_name)

    weighted_count = weighted_total / weight_total if weight_total > 0 else 0.0
    evidence_floor_texts = _dedupe_evidence_texts(
        positive_outputs,
        set(config["evidence_floor_lfs"]),
    )
    evidence_floor = len(evidence_floor_texts)
    min_count = int(config["min_count"])
    max_count = int(config["max_count"].get(label, 20))
    raw_count = round(max(weighted_count, evidence_floor, min_count))
    predicted_count = max(0, min(max_count, raw_count))
    count_confidence = min(
        1.0,
        max(0.0, vote_decision.confidence * (0.6 + 0.4 * min(1.0, weight_total))),
    )

    return Lv1CountDecision(
        predicted_count=predicted_count,
        count_confidence=count_confidence,
        count_sources=count_sources,
        evidence_floor=evidence_floor,
        weighted_count=weighted_count,
        metadata={
            "method": config["method"],
            "max_count": max_count,
            "evidence_floor_texts": evidence_floor_texts[:20],
        },
    )
