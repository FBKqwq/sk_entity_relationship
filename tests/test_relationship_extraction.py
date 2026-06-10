"""Tests for schema-constrained relationship extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.relationship_extraction.relationship_base_builder import (
    CONTEXT_LEVELS,
    build_candidate_relationships,
    extract_relationship_base_for_file,
)


def _entity(
    entity_id: str,
    entity_type: str,
    name: str,
    chunk_id: str,
    *,
    document_id: str = "DOC1",
) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "entity_type": entity_type,
        "name": name,
        "evidence_text": name,
        "confidence": 0.9,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_build_candidate_relationships_only_generates_legal_schema_edges() -> None:
    chunks = [
        {"chunk_id": "CH1", "text": "胆源性急性胰腺炎需要血清淀粉酶检查。"},
    ]
    entities = [
        _entity("E1", "sub_diseases", "胆源性急性胰腺炎", "CH1"),
        _entity("E2", "tests", "血清淀粉酶", "CH1"),
        _entity("E3", "plans", "抗菌药物", "CH1"),
    ]

    candidates = build_candidate_relationships(entities, chunks, document_id="DOC1", source_pdf="sample.pdf")

    edge_keys = {
        (candidate.start_entity_type, candidate.relation_type, candidate.end_entity_type)
        for candidate in candidates
    }
    assert ("sub_diseases", "requires_test", "tests") in edge_keys
    assert ("sub_diseases", "implements_by", "plans") not in edge_keys
    assert len([candidate for candidate in candidates if candidate.relation_type == "requires_test"]) == 1


def test_build_candidate_relationships_allows_nearby_cross_chunk_by_default() -> None:
    chunks = [
        {"chunk_id": "CH1", "text": "胆源性急性胰腺炎。"},
        {"chunk_id": "CH2", "text": "需要血清淀粉酶检查。"},
    ]
    entities = [
        _entity("E1", "sub_diseases", "胆源性急性胰腺炎", "CH1"),
        _entity("E2", "tests", "血清淀粉酶", "CH2"),
    ]

    cross_chunk_candidates = build_candidate_relationships(entities, chunks, document_id="DOC1", source_pdf="sample.pdf")
    same_chunk_candidates = build_candidate_relationships(
        entities,
        chunks,
        document_id="DOC1",
        source_pdf="sample.pdf",
        same_chunk_only=True,
    )
    assert len(cross_chunk_candidates) == 1
    assert cross_chunk_candidates[0].search_level == "adjacent_chunk"
    assert cross_chunk_candidates[0].chunk_distance == 1
    assert same_chunk_candidates == []


def test_build_candidate_relationships_uses_etiology_to_sub_disease_cause_direction() -> None:
    chunks = [
        {"chunk_id": "CH1", "text": "糖尿病是复杂性尿路感染的重要危险因素。"},
    ]
    entities = [
        _entity("E1", "sub_diseases", "复杂性尿路感染", "CH1"),
        _entity("E2", "etiologies", "糖尿病", "CH1"),
    ]

    candidates = build_candidate_relationships(entities, chunks, document_id="DOC1", source_pdf="sample.pdf")

    edge_keys = {
        (candidate.start_entity_type, candidate.relation_type, candidate.end_entity_type)
        for candidate in candidates
    }
    assert ("etiologies", "causes", "sub_diseases") in edge_keys
    assert ("sub_diseases", "causes", "etiologies") not in edge_keys
    cause_candidate = next(candidate for candidate in candidates if candidate.relation_type == "causes")
    assert cause_candidate.start_entity_id == "E2"
    assert cause_candidate.end_entity_id == "E1"
    assert cause_candidate.search_level == "same_chunk"


def test_rejected_audit_drops_relationship(tmp_path: Path) -> None:
    entities_path = tmp_path / "sample.entity_base.jsonl"
    chunks_path = tmp_path / "sample.chunk.json"
    output_path = tmp_path / "sample.relationship_base.jsonl"
    raw_path = tmp_path / "sample.relationship_llm_raw.jsonl"
    candidate_path = tmp_path / "sample.candidate_relationships.jsonl"
    _write_jsonl(
        entities_path,
        [
            _entity("E1", "sub_diseases", "胆源性急性胰腺炎", "CH1"),
            _entity("E2", "tests", "血清淀粉酶", "CH1"),
        ],
    )
    chunks_path.write_text(
        json.dumps(
            {"doc_id": "DOC1", "pdf_path": "sample.pdf", "chunks": [{"chunk_id": "CH1", "text": "检查章节。"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def fake_llm(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "ok", "model": "fake", "text": '{"approved": false, "evidence": "", "confidence": 0}'}

    summary = extract_relationship_base_for_file(
        entities_path=entities_path,
        chunks_path=chunks_path,
        output_path=output_path,
        candidate_output_path=candidate_path,
        raw_output_path=raw_path,
        llm_func=fake_llm,
    )

    assert summary["candidates"] == 1
    assert summary["confirmed_relationships"] == 0
    assert _read_jsonl(output_path) == []
    assert _read_jsonl(candidate_path)[0]["relation_type"] == "requires_test"
    audit_rows = [row for row in _read_jsonl(raw_path) if row["phase"] == "audit"]
    assert [row["context_level"] for row in audit_rows] == list(CONTEXT_LEVELS)


def test_approved_relationship_extracts_and_sanitizes_properties(tmp_path: Path) -> None:
    entities_path = tmp_path / "sample.entity_base.jsonl"
    chunks_path = tmp_path / "sample.chunk.json"
    output_path = tmp_path / "sample.relationship_base.jsonl"
    raw_path = tmp_path / "sample.relationship_llm_raw.jsonl"
    _write_jsonl(
        entities_path,
        [
            _entity("E1", "sub_diseases", "胆源性急性胰腺炎", "CH1"),
            _entity("E2", "tests", "血清淀粉酶", "CH1"),
        ],
    )
    chunks_path.write_text(
        json.dumps(
            {
                "doc_id": "DOC1",
                "pdf_path": "sample.pdf",
                "chunks": [
                    {
                        "chunk_id": "CH1",
                        "text": "胆源性急性胰腺炎需要检测血清淀粉酶，超过正常上限可支持诊断。",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_llm(prompt: str, **kwargs: Any) -> dict[str, Any]:
        if '"approved"' in prompt:
            calls.append("audit")
            return {
                "status": "ok",
                "model": "fake",
                "text": '{"approved": true, "evidence": "需要检测血清淀粉酶", "confidence": 0.8}',
            }
        calls.append("properties")
        return {
            "status": "ok",
            "model": "fake",
            "text": json.dumps(
                {
                    "properties": {
                        "diagnostic_role": "supportive",
                        "unit": "U/L",
                        "exam_name": "旧字段",
                    },
                    "evidence": "超过正常上限可支持诊断",
                    "confidence": 0.7,
                },
                ensure_ascii=False,
            ),
        }

    summary = extract_relationship_base_for_file(
        entities_path=entities_path,
        chunks_path=chunks_path,
        output_path=output_path,
        raw_output_path=raw_path,
        llm_func=fake_llm,
    )

    records = _read_jsonl(output_path)
    assert summary["confirmed_relationships"] == 1
    assert calls == ["audit", "properties"]
    assert records[0]["relation_type"] == "requires_test"
    assert records[0]["start_entity_id"] == "E1"
    assert records[0]["end_entity_id"] == "E2"
    assert records[0]["properties"]["diagnostic_role"] == "supportive"
    assert records[0]["properties"]["unit"] == "U/L"
    assert "exam_name" not in records[0]["properties"]
    assert records[0]["status"] == "confirmed"


def test_relationship_extraction_writes_raw_trace_incrementally(tmp_path: Path) -> None:
    entities_path = tmp_path / "sample.entity_base.jsonl"
    chunks_path = tmp_path / "sample.chunk.json"
    output_path = tmp_path / "sample.relationship_base.jsonl"
    raw_path = tmp_path / "sample.relationship_llm_raw.jsonl"
    _write_jsonl(
        entities_path,
        [
            _entity("E1", "sub_diseases", "复杂性尿路感染", "CH1"),
            _entity("E2", "tests", "尿培养", "CH1"),
        ],
    )
    chunks_path.write_text(
        json.dumps(
            {"doc_id": "DOC1", "pdf_path": "sample.pdf", "chunks": [{"chunk_id": "CH1", "text": "复杂性尿路感染需要尿培养。"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    calls = 0

    def fake_llm(prompt: str, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            rows = _read_jsonl(raw_path)
            assert any(row["phase"] == "audit" for row in rows)
        if '"approved"' in prompt:
            return {
                "status": "ok",
                "model": "fake",
                "text": '{"approved": true, "evidence": "需要尿培养", "confidence": 0.8}',
            }
        return {"status": "ok", "model": "fake", "text": '{"properties": {}, "evidence": "", "confidence": 0}'}

    extract_relationship_base_for_file(
        entities_path=entities_path,
        chunks_path=chunks_path,
        output_path=output_path,
        raw_output_path=raw_path,
        llm_func=fake_llm,
    )

    assert raw_path.exists()
    assert output_path.exists()


def test_cross_chunk_candidates_expand_context_after_failed_audit(tmp_path: Path) -> None:
    entities_path = tmp_path / "sample.entity_base.jsonl"
    chunks_path = tmp_path / "sample.chunk.json"
    output_path = tmp_path / "sample.relationship_base.jsonl"
    raw_path = tmp_path / "sample.relationship_llm_raw.jsonl"
    _write_jsonl(
        entities_path,
        [
            _entity("E1", "sub_diseases", "胆源性急性胰腺炎", "CH2"),
            _entity("E2", "tests", "血清淀粉酶", "CH4"),
        ],
    )
    chunks_path.write_text(
        json.dumps(
            {
                "doc_id": "DOC1",
                "pdf_path": "sample.pdf",
                "chunks": [
                    {"chunk_id": "CH1", "text": "前文。"},
                    {"chunk_id": "CH2", "text": "胆源性急性胰腺炎。"},
                    {"chunk_id": "CH3", "text": "过渡。"},
                    {"chunk_id": "CH4", "text": "全文中说明该病需要血清淀粉酶检查。"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    audit_calls = 0

    def fake_llm(prompt: str, **kwargs: Any) -> dict[str, Any]:
        nonlocal audit_calls
        if '"approved"' in prompt:
            audit_calls += 1
            approved = audit_calls == len(CONTEXT_LEVELS)
            return {
                "status": "ok",
                "model": "fake",
                "text": json.dumps(
                    {
                        "approved": approved,
                        "evidence": "该病需要血清淀粉酶检查" if approved else "",
                        "confidence": 0.8 if approved else 0,
                    },
                    ensure_ascii=False,
                ),
            }
        return {"status": "ok", "model": "fake", "text": '{"properties": {}, "evidence": "", "confidence": 0}'}

    summary = extract_relationship_base_for_file(
        entities_path=entities_path,
        chunks_path=chunks_path,
        output_path=output_path,
        raw_output_path=raw_path,
        llm_func=fake_llm,
    )

    raw_rows = _read_jsonl(raw_path)
    audit_levels = [row["context_level"] for row in raw_rows if row["phase"] == "audit"]
    assert summary["candidates"] == 1
    assert summary["confirmed_relationships"] == 1
    assert audit_levels == list(CONTEXT_LEVELS)
    assert _read_jsonl(output_path)[0]["context_level"] == CONTEXT_LEVELS[-1]


def test_review_entities_are_relationship_candidates_by_default(tmp_path: Path) -> None:
    entities_path = tmp_path / "sample.entity_nodes.jsonl"
    chunks_path = tmp_path / "sample.chunk.json"
    output_path = tmp_path / "sample.relationship_base.jsonl"
    _write_jsonl(
        entities_path,
        [
            {**_entity("E1", "sub_diseases", "complicated UTI", "CH1"), "status": "review", "entity_status": "review"},
            {**_entity("E2", "tests", "urine culture", "CH1"), "status": "accepted", "entity_status": "accepted"},
        ],
    )
    chunks_path.write_text(
        json.dumps(
            {"doc_id": "DOC1", "pdf_path": "sample.pdf", "chunks": [{"chunk_id": "CH1", "text": "complicated UTI requires urine culture"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def fake_llm(prompt: str, **kwargs: Any) -> dict[str, Any]:
        if '"approved"' in prompt:
            return {"status": "ok", "model": "fake", "text": '{"approved": true, "evidence": "requires urine culture", "confidence": 0.8}'}
        return {"status": "ok", "model": "fake", "text": '{"properties": {}, "evidence": "", "confidence": 0}'}

    summary = extract_relationship_base_for_file(
        entities_path=entities_path,
        chunks_path=chunks_path,
        output_path=output_path,
        llm_func=fake_llm,
    )

    assert summary["raw_entities"] == 2
    assert summary["entities"] == 2
    assert summary["include_review_entities"] is True
    assert summary["confirmed_relationships"] == 1
    assert _read_jsonl(output_path)[0]["relation_type"] == "requires_test"


def test_accepted_only_relationships_exclude_review_entities(tmp_path: Path) -> None:
    entities_path = tmp_path / "sample.entity_nodes.jsonl"
    chunks_path = tmp_path / "sample.chunk.json"
    output_path = tmp_path / "sample.relationship_base.jsonl"
    _write_jsonl(
        entities_path,
        [
            {**_entity("E1", "sub_diseases", "complicated UTI", "CH1"), "status": "review", "entity_status": "review"},
            {**_entity("E2", "tests", "urine culture", "CH1"), "status": "accepted", "entity_status": "accepted"},
        ],
    )
    chunks_path.write_text(
        json.dumps(
            {"doc_id": "DOC1", "pdf_path": "sample.pdf", "chunks": [{"chunk_id": "CH1", "text": "complicated UTI requires urine culture"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No sub_diseases found"):
        extract_relationship_base_for_file(
            entities_path=entities_path,
            chunks_path=chunks_path,
            output_path=output_path,
            include_review_entities=False,
            llm_func=lambda *args, **kwargs: {"status": "ok", "text": "{}"},
        )


def test_accepted_only_keeps_review_disease_anchor_for_document_has_sub_disease(tmp_path: Path) -> None:
    entities_path = tmp_path / "sample.entity_nodes.jsonl"
    chunks_path = tmp_path / "sample.chunk.json"
    output_path = tmp_path / "sample.relationship_base.jsonl"
    candidate_path = tmp_path / "sample.candidate_relationships.jsonl"
    _write_jsonl(
        entities_path,
        [
            {**_entity("D1", "diseases", "Behcet disease", "CH1"), "status": "review", "entity_status": "review"},
            {**_entity("S1", "sub_diseases", "ocular Behcet syndrome", "CH3"), "status": "accepted", "entity_status": "accepted"},
        ],
    )
    chunks_path.write_text(
        json.dumps(
            {
                "doc_id": "DOC1",
                "pdf_path": "sample.pdf",
                "chunks": [
                    {"chunk_id": "CH1", "text": "Behcet disease guideline."},
                    {"chunk_id": "CH2", "text": "middle section"},
                    {"chunk_id": "CH3", "text": "ocular Behcet syndrome is an organ involvement subtype."},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def fake_llm(prompt: str, **kwargs: Any) -> dict[str, Any]:
        if '"approved"' in prompt:
            return {"status": "ok", "model": "fake", "text": '{"approved": true, "evidence": "organ involvement subtype", "confidence": 0.8}'}
        return {"status": "ok", "model": "fake", "text": '{"properties": {}, "evidence": "", "confidence": 0}'}

    summary = extract_relationship_base_for_file(
        entities_path=entities_path,
        chunks_path=chunks_path,
        output_path=output_path,
        candidate_output_path=candidate_path,
        include_review_entities=False,
        llm_func=fake_llm,
    )

    candidates = _read_jsonl(candidate_path)
    records = _read_jsonl(output_path)
    assert summary["entities"] == 2
    assert candidates[0]["relation_type"] == "has_sub_disease"
    assert candidates[0]["search_level"] == "document"
    assert records[0]["relation_type"] == "has_sub_disease"


def test_implements_by_requires_confirmed_treatment_anchor(tmp_path: Path) -> None:
    entities_path = tmp_path / "sample.entity_nodes.jsonl"
    chunks_path = tmp_path / "sample.chunk.json"
    output_path = tmp_path / "sample.relationship_base.jsonl"
    _write_jsonl(
        entities_path,
        [
            _entity("E1", "sub_diseases", "complicated UTI", "CH1"),
            _entity("E2", "treatments", "antibiotic principle", "CH1"),
            _entity("E3", "plans", "levofloxacin", "CH1"),
        ],
    )
    chunks_path.write_text(
        json.dumps(
            {"doc_id": "DOC1", "pdf_path": "sample.pdf", "chunks": [{"chunk_id": "CH1", "text": "complicated UTI uses antibiotic principle with levofloxacin"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def fake_llm(prompt: str, **kwargs: Any) -> dict[str, Any]:
        if '"approved"' in prompt:
            approved = '"implements_by"' in prompt
            return {
                "status": "ok",
                "model": "fake",
                "text": json.dumps({"approved": approved, "evidence": "levofloxacin" if approved else "", "confidence": 0.8}),
            }
        return {"status": "ok", "model": "fake", "text": '{"properties": {}, "evidence": "", "confidence": 0}'}

    summary = extract_relationship_base_for_file(
        entities_path=entities_path,
        chunks_path=chunks_path,
        output_path=output_path,
        llm_func=fake_llm,
    )

    assert summary["confirmed_relationships"] == 0
    assert _read_jsonl(output_path) == []


def test_relationship_extraction_fails_without_sub_disease(tmp_path: Path) -> None:
    entities_path = tmp_path / "sample.entity_base.jsonl"
    chunks_path = tmp_path / "sample.chunk.json"
    output_path = tmp_path / "sample.relationship_base.jsonl"
    _write_jsonl(
        entities_path,
        [
            _entity("E1", "treatments", "抗菌药物治疗", "CH1"),
            _entity("E2", "plans", "左氧氟沙星", "CH1"),
        ],
    )
    chunks_path.write_text(
        json.dumps(
            {"doc_id": "DOC1", "pdf_path": "sample.pdf", "chunks": [{"chunk_id": "CH1", "text": "治疗方案。"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No sub_diseases found"):
        extract_relationship_base_for_file(
            entities_path=entities_path,
            chunks_path=chunks_path,
            output_path=output_path,
            llm_func=lambda *args, **kwargs: {"status": "ok", "text": "{}"},
        )
