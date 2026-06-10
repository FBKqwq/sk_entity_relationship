"""Tests for entity_base record building."""

from src.entity_extraction.entity_base_builder import build_entity_base_records


def test_build_entity_base_records_deduplicates_and_adds_trace_fields() -> None:
    chunk = {
        "chunk_id": "CH1",
        "document_id": "DOC1",
        "section_title": "检查",
        "section_path": ["正文", "检查"],
    }
    entities = [
        {
            "entity_type": "tests",
            "name": "针刺反应",
            "properties": {"test_name": "针刺反应"},
            "evidence": "针刺反应阳性",
            "confidence": 0.9,
        },
        {
            "entity_type": "tests",
            "name": "针刺反应",
            "properties": {"test_name": "针刺反应"},
            "evidence": "重复",
            "confidence": 0.8,
        },
    ]

    records = build_entity_base_records(chunk, entities)

    assert len(records) == 1
    assert records[0]["chunk_id"] == "CH1"
    assert records[0]["document_id"] == "DOC1"
    assert records[0]["entity_type"] == "tests"
    assert records[0]["name"] == "针刺反应"
    assert records[0]["properties"] == {}
    assert records[0]["candidate_properties"]["test_name"] == "针刺反应"
    assert records[0]["evidence_text"] == "针刺反应阳性"
    assert records[0]["status"] == "candidate"


def test_build_entity_base_records_fills_disease_id() -> None:
    chunk = {"chunk_id": "__DOC__", "document_id": "DOC1", "section_title": "文档标题"}
    entities = [
        {
            "entity_type": "diseases",
            "name": "急性胰腺炎",
            "properties": {"disease_name": "急性胰腺炎", "ICD_10": ""},
            "evidence": "中国急性胰腺炎诊治指南（2021）",
            "confidence": 0.98,
        }
    ]

    records = build_entity_base_records(chunk, entities, source="document_title_rule")

    assert len(records) == 1
    assert records[0]["entity_type"] == "diseases"
    assert records[0]["properties"] == {}
    assert records[0]["candidate_properties"]["disease_name"] == "急性胰腺炎"
    assert records[0]["source"] == "document_title_rule"
