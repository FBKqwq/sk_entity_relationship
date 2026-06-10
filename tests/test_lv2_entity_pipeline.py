"""Tests for Lv2 entity typing and post-Lv2 property extraction."""

from __future__ import annotations

import json
from typing import Any

from src.entity_extraction.entity_property_extractor import extract_entity_properties
from src.weak_supervision.common.lf_output import LFOutput
from src.weak_supervision_lv2.entity_signal_builder import build_entity_label_results, chunk_by_id


class FixedLF:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def apply(self, entity: dict[str, Any], chunk: dict[str, Any], label: str) -> LFOutput:
        confidence = self.scores.get(label)
        if confidence is None:
            return LFOutput(lf_name="fixed", label=label)
        return LFOutput(lf_name="fixed", label=label, vote=1, confidence=confidence)


class NamedFixedLF(FixedLF):
    def __init__(self, name: str, scores: dict[str, float]) -> None:
        super().__init__(scores)
        self.name = name

    def apply(self, entity: dict[str, Any], chunk: dict[str, Any], label: str) -> LFOutput:
        confidence = self.scores.get(label)
        if confidence is None:
            return LFOutput(lf_name=self.name, label=label, metadata={"entity_id": entity.get("entity_id")})
        return LFOutput(
            lf_name=self.name,
            label=label,
            vote=1,
            confidence=confidence,
            metadata={
                "entity_id": entity.get("entity_id"),
                "prompt_name": self.name.removeprefix("lv2_entity_prompted_llm_"),
            },
        )


def _entity(entity_id: str, entity_type: str = "sub_diseases", evidence: str = "fever syndrome") -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "document_id": "DOC1",
        "chunk_id": "CH1",
        "entity_type": entity_type,
        "name": evidence,
        "content": evidence,
        "evidence_text": evidence,
        "confidence": 0.8,
        "status": "candidate",
    }


def test_lv2_status_layers_accept_review_and_reject() -> None:
    chunks = [{"chunk_id": "CH1", "text": "fever syndrome"}]
    accepted, conflicts, report = build_entity_label_results(
        [_entity("E1")],
        chunks,
        labels=["sub_diseases", "symptoms"],
        lfs=[FixedLF({"sub_diseases": 1.0})],
    )
    assert accepted[0]["status"] == "accepted"
    assert conflicts == []
    assert report["accepted"] == 1

    review, conflicts, report = build_entity_label_results(
        [_entity("E2")],
        chunks,
        labels=["sub_diseases", "symptoms"],
        lfs=[FixedLF({"sub_diseases": 1.0, "symptoms": 0.95})],
    )
    assert review[0]["status"] == "review"
    assert "type_conflict_top2_gap" in review[0]["conflict_reasons"]
    assert conflicts[0]["status"] == "review"
    assert report["review"] == 1

    rejected, conflicts, report = build_entity_label_results(
        [_entity("E3", evidence="missing evidence")],
        chunks,
        labels=["sub_diseases", "symptoms"],
        lfs=[FixedLF({"sub_diseases": 1.0})],
    )
    assert rejected[0]["status"] == "rejected"
    assert rejected[0]["conflict_reasons"] == ["evidence_not_located"]
    assert conflicts[0]["status"] == "rejected"
    assert report["rejected"] == 1


def test_entity_properties_run_after_lv2_without_changing_type_or_deleting_on_failure() -> None:
    labels = [
        {
            "entity_id": "E1",
            "document_id": "DOC1",
            "chunk_id": "CH1",
            "final_entity_type": "tests",
            "name": "urine culture",
            "evidence_text": "urine culture positive",
            "status": "accepted",
            "lv2_probability": 0.72,
        }
    ]
    base = {
        "E1": {
            "entity_id": "E1",
            "entity_type": "symptoms",
            "name": "urine culture",
            "chunk_id": "CH1",
            "evidence_text": "urine culture positive",
        }
    }

    def fake_llm(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "ok", "text": "not json"}

    property_rows, entity_nodes, conflicts, raw_rows = extract_entity_properties(
        labels,
        base,
        chunk_by_id([{"chunk_id": "CH1", "text": "urine culture positive"}]),
        llm_func=fake_llm,
    )

    assert len(property_rows) == 1
    assert len(entity_nodes) == 1
    assert entity_nodes[0]["entity_type"] == "tests"
    assert entity_nodes[0]["status"] == "accepted"
    assert entity_nodes[0]["property_status"] == "fallback_defaults"
    assert entity_nodes[0]["properties"]["test_id"] == "E1"
    assert entity_nodes[0]["properties"]["test_name"] == "urine culture"
    assert entity_nodes[0]["properties"]["confidence"] == entity_nodes[0]["confidence"]
    assert conflicts[0]["status"] == "property_incomplete"
    assert raw_rows[0]["parsed"] is None


def test_lv2_accepts_candidate_type_when_all_llm_prompts_support_it() -> None:
    chunks = [{"chunk_id": "CH1", "text": "局部创伤、某些食物、疲劳、失眠、月经可能为触发因素。"}]
    entity = _entity("E1", entity_type="etiologies", evidence="局部创伤")
    entity["evidence_text"] = "局部创伤、某些食物、疲劳、失眠、月经可能为触发因素。"
    lfs = [
        NamedFixedLF("lv2_entity_context_window", {"symptoms": 0.68}),
        NamedFixedLF("lv2_entity_prompted_llm_type_boundary", {"etiologies": 0.99}),
        NamedFixedLF("lv2_entity_prompted_llm_evidence_support", {"etiologies": 0.98}),
        NamedFixedLF("lv2_entity_prompted_llm_schema_contrast", {"etiologies": 0.97}),
    ]

    results, conflicts, report = build_entity_label_results(
        [entity],
        chunks,
        labels=["symptoms", "etiologies"],
        lfs=lfs,
    )

    assert results[0]["final_entity_type"] == "etiologies"
    assert results[0]["status"] == "accepted"
    assert results[0]["llm_candidate_type_protected"] is True
    assert conflicts == []
    assert report["accepted"] == 1


def test_lv2_promotes_dose_or_administration_treatment_to_plan() -> None:
    chunks = [{"chunk_id": "CH1", "text": "秋水仙碱（0.5mg,每日2~3次）可用于改善口腔溃疡。"}]
    entity = _entity("E1", entity_type="treatments", evidence="秋水仙碱（0.5mg,每日2~3次）")
    lfs = [
        NamedFixedLF("lv2_entity_prompted_llm_type_boundary", {"treatments": 0.99}),
        NamedFixedLF("lv2_entity_prompted_llm_evidence_support", {"treatments": 0.98}),
        NamedFixedLF("lv2_entity_prompted_llm_schema_contrast", {"treatments": 0.97}),
    ]

    results, conflicts, report = build_entity_label_results(
        [entity],
        chunks,
        labels=["treatments", "plans"],
        lfs=lfs,
    )

    assert results[0]["final_entity_type"] == "plans"
    assert results[0]["status"] == "accepted"
    assert results[0]["plan_execution_override"] is True
    assert conflicts == []
    assert report["accepted"] == 1


def test_entity_properties_can_include_review_entities_when_configured() -> None:
    labels = [
        {
            "entity_id": "E1",
            "document_id": "DOC1",
            "chunk_id": "CH1",
            "final_entity_type": "sub_diseases",
            "name": "complicated UTI",
            "evidence_text": "complicated UTI",
            "status": "review",
            "lv2_probability": 0.51,
        }
    ]

    def fake_llm(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "ok",
            "text": json.dumps({"properties": {"sub_disease_name": "complicated UTI"}, "confidence": 0.6}),
        }

    _, default_nodes, default_conflicts, _ = extract_entity_properties(
        labels,
        {"E1": labels[0]},
        chunk_by_id([{"chunk_id": "CH1", "text": "complicated UTI"}]),
        include_review=False,
        llm_func=fake_llm,
    )
    _, included_nodes, _, _ = extract_entity_properties(
        labels,
        {"E1": labels[0]},
        chunk_by_id([{"chunk_id": "CH1", "text": "complicated UTI"}]),
        include_review=True,
        llm_func=fake_llm,
    )

    assert default_nodes[0]["status"] == "review"
    assert default_nodes[0]["property_status"] == "review_defaults"
    assert default_conflicts[0]["status"] == "property_incomplete"
    assert included_nodes[0]["status"] == "review"
    assert included_nodes[0]["property_status"] == "ok"
