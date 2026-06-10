"""Tests for Lv1 chunk vote models."""

from src.weak_supervision.common.lf_output import EvidenceSpan, LFOutput
from src.weak_supervision_lv1.chunk_vote_model import majority_vote_binary
from src.weak_supervision_lv1.chunk_vote_model import manual_weighted_vote_label


def test_majority_vote_binary_abstains_to_false() -> None:
    present, confidence = majority_vote_binary([0, 0])
    assert present is False
    assert confidence == 0.0


def test_manual_weighted_vote_accepts_primary_supported_label() -> None:
    outputs = [
        LFOutput(
            "lv1_chunk_medical_pattern",
            "symptoms",
            vote=1,
            confidence=0.8,
            count=2,
            evidence=[EvidenceSpan(0, 2, "口干")],
        ),
        LFOutput("lv1_chunk_dictionary", "symptoms", vote=1, confidence=0.7, count=1),
        LFOutput("lv1_chunk_section_prior", "symptoms", vote=0, confidence=0.0),
    ]

    decision = manual_weighted_vote_label("symptoms", outputs)

    assert decision.present is True
    assert decision.status == "accepted"
    assert decision.count == 2
    assert decision.supporting_lfs == ["lv1_chunk_medical_pattern", "lv1_chunk_dictionary"]
    assert decision.evidence_texts == ["口干"]


def test_manual_weighted_vote_marks_auxiliary_only_support_as_weak() -> None:
    outputs = [
        LFOutput("lv1_chunk_medical_pattern", "tests", vote=0, confidence=0.0),
        LFOutput("lv1_chunk_prompted_llm", "tests", vote=0, confidence=0.0),
        LFOutput("lv1_chunk_dictionary", "tests", vote=0, confidence=0.0),
        LFOutput("lv1_chunk_regex_indicator", "tests", vote=1, confidence=0.64, count=1),
        LFOutput("lv1_chunk_section_prior", "tests", vote=0, confidence=0.0),
    ]

    decision = manual_weighted_vote_label(
        "tests",
        outputs,
        vote_config={
            "threshold": {"tests": 0.08},
        },
    )

    assert decision.present is True
    assert decision.status == "weak"
    assert decision.metadata["only_auxiliary_support"] is True


def test_manual_weighted_vote_rejects_below_threshold() -> None:
    outputs = [
        LFOutput("lv1_chunk_section_prior", "symptoms", vote=1, confidence=0.45, count=1),
        LFOutput("lv1_chunk_medical_pattern", "symptoms", vote=0, confidence=0.0),
        LFOutput("lv1_chunk_dictionary", "symptoms", vote=0, confidence=0.0),
        LFOutput("lv1_chunk_prompted_llm", "symptoms", vote=0, confidence=0.0),
    ]

    decision = manual_weighted_vote_label("symptoms", outputs)

    assert decision.present is False
    assert decision.status == "rejected"
