"""Tests for Lv1 count regression and fallback rules."""

from src.weak_supervision.common.lf_output import EvidenceSpan, LFOutput
from src.weak_supervision_lv1.chunk_count_regression import fallback_count, manual_count_fusion
from src.weak_supervision_lv1.chunk_vote_model import Lv1VoteDecision


def test_fallback_count_uses_max_count() -> None:
    assert fallback_count(1, 3, 2) == 3


def test_manual_count_fusion_returns_zero_when_label_absent() -> None:
    decision = Lv1VoteDecision(
        label="symptoms",
        present=False,
        confidence=0.0,
        vote_score=0.0,
        threshold=0.3,
        status="rejected",
    )

    count = manual_count_fusion("symptoms", [], decision)

    assert count.predicted_count == 0
    assert count.count_confidence == 0.0


def test_manual_count_fusion_combines_weighted_count_and_evidence_floor() -> None:
    decision = Lv1VoteDecision(
        label="symptoms",
        present=True,
        confidence=0.8,
        vote_score=0.6,
        threshold=0.3,
        status="accepted",
    )
    outputs = [
        LFOutput(
            "lv1_chunk_medical_pattern",
            "symptoms",
            vote=1,
            confidence=0.8,
            count=2,
            evidence=[EvidenceSpan(0, 2, "口干"), EvidenceSpan(3, 5, "眼干")],
        ),
        LFOutput(
            "lv1_chunk_dictionary",
            "symptoms",
            vote=1,
            confidence=0.7,
            count=1,
            evidence=[EvidenceSpan(0, 2, "口干")],
        ),
        LFOutput("lv1_chunk_prompted_llm", "symptoms", vote=1, confidence=0.9, count=5),
    ]

    count = manual_count_fusion("symptoms", outputs, decision)

    assert count.predicted_count >= 2
    assert count.count_confidence > 0
    assert count.count_sources == [
        "lv1_chunk_medical_pattern",
        "lv1_chunk_dictionary",
        "lv1_chunk_prompted_llm",
    ]
    assert count.evidence_floor == 2
