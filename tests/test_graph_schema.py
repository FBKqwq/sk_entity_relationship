"""Tests for the canonical graph schema contract."""

from src.weak_supervision.common.graph_schema import (
    ACTIVE_ENTITY_LABELS,
    ENTITY_LABELS,
    ENTITY_PROPERTY_MAP,
    GRAPH_RELATIONS,
    RELATION_PROPERTY_MAP,
)


def test_graph_labels_match_final_v1_schema() -> None:
    assert ENTITY_LABELS == [
        "diseases",
        "sub_diseases",
        "symptoms",
        "tests",
        "treatments",
        "plans",
        "etiologies",
        "pathogeneses",
    ]
    assert ACTIVE_ENTITY_LABELS == ENTITY_LABELS


def test_graph_properties_match_final_v1_schema() -> None:
    assert ENTITY_PROPERTY_MAP["diseases"] == [
        "disease_id",
        "disease_name",
        "normalized_name",
        "aliases",
        "source_document_ids",
        "evidence_ids",
        "confidence",
    ]
    assert ENTITY_PROPERTY_MAP["sub_diseases"] == [
        "sub_disease_id",
        "sub_disease_name",
        "parent_disease_id",
        "disease_subtype",
        "clinical_stage",
        "severity",
        "population",
        "gender",
        "age_min",
        "age_max",
        "evidence_ids",
        "confidence",
    ]
    assert ENTITY_PROPERTY_MAP["tests"] == [
        "test_id",
        "test_name",
        "normal_range_min",
        "normal_range_max",
        "is_digital",
        "evidence_ids",
        "confidence",
    ]


def test_graph_relations_match_final_v1_schema() -> None:
    relation_triples = [
        (relation.source, relation.relation, relation.target) for relation in GRAPH_RELATIONS
    ]
    assert relation_triples == [
        ("Disease", "has_sub_disease", "Sub_disease"),
        ("Sub_disease", "manifests_as", "Symptom"),
        ("Sub_disease", "requires_test", "Test"),
        ("Sub_disease", "follows_treatment", "Treatment"),
        ("Treatment", "implements_by", "Plan"),
        ("Etiology", "causes", "Sub_disease"),
        ("Sub_disease", "explained_by", "Pathogenesis"),
    ]


def test_relation_properties_include_diagnostic_threshold_contract() -> None:
    assert RELATION_PROPERTY_MAP["manifests_as"] == [
        "relation_id",
        "relation_name",
        "relation_type",
        "diagnostic_role",
        "type_info",
        "weight",
        "criterion_group_id",
        "required_count",
        "typicality",
        "duration_condition",
        "section_priority",
        "evidence_ids",
        "confidence",
    ]
    assert RELATION_PROPERTY_MAP["requires_test"] == [
        "relation_id",
        "relation_name",
        "relation_type",
        "type_info",
        "diagnostic_role",
        "value_min",
        "value_max",
        "operator",
        "unit",
        "result_text",
        "weight",
        "gold_standard",
        "criterion_group_id",
        "required_count",
        "polarity",
        "typicality",
        "section_priority",
        "evidence_level",
        "evidence_ids",
        "confidence",
    ]
    assert RELATION_PROPERTY_MAP["causes"] == [
        "relation_id",
        "relation_name",
        "relation_type",
        "causal_strength",
        "typicality",
        "criterion_group_id",
        "required_count",
        "evidence_ids",
        "confidence",
    ]
