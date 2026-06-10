"""Tests for Lv1 chunk signal building."""

from src.weak_supervision.common.lf_output import LFOutput
from src.weak_supervision_lv1.chunk_signal_builder import build_chunk_label_results


def test_build_chunk_label_results_adds_chunk_context() -> None:
    chunk = {
        "chunk_id": "CH0001",
        "document_id": "doc1",
        "section_title": "临床表现",
        "section_path": ["正文", "临床表现"],
    }
    outputs = [
        LFOutput("lv1_chunk_medical_pattern", "symptoms", vote=1, confidence=0.8, count=3),
        LFOutput("lv1_chunk_dictionary", "symptoms", vote=1, confidence=0.7, count=2),
        LFOutput("lv1_chunk_medical_pattern", "tests", vote=0, confidence=0.0),
    ]

    records = build_chunk_label_results(chunk, ["symptoms", "tests"], outputs)

    symptoms = records[0]
    tests = records[1]
    assert symptoms["chunk_id"] == "CH0001"
    assert symptoms["label"] == "symptoms"
    assert symptoms["present"] is True
    assert symptoms["status"] == "accepted"
    assert symptoms["predicted_count"] >= 2
    assert symptoms["count_confidence"] > 0
    assert symptoms["section_path"] == ["正文", "临床表现"]
    assert tests["label"] == "tests"
    assert tests["present"] is False
    assert tests["predicted_count"] == 0
