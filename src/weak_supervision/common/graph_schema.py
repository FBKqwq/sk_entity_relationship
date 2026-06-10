"""Canonical medical knowledge graph contract used by weak supervision."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EntitySpec:
    """Definition for one graph entity type."""

    label: str
    zh_name: str
    entity_type: str
    properties: tuple[str, ...]
    extract_enabled: bool = True
    description: str = ""


@dataclass(frozen=True)
class RelationSpec:
    """Definition for one graph relation type."""

    source: str
    relation: str
    target: str
    zh_name: str
    properties: tuple[str, ...]
    description: str = ""


GRAPH_ENTITIES: tuple[EntitySpec, ...] = (
    EntitySpec(
        label="Disease",
        zh_name="疾病",
        entity_type="diseases",
        properties=(
            "disease_id",
            "disease_name",
            "normalized_name",
            "aliases",
            "source_document_ids",
            "evidence_ids",
            "confidence",
        ),
        description="Disease category or parent disease node.",
    ),
    EntitySpec(
        label="Sub_disease",
        zh_name="确诊名/疾病分型",
        entity_type="sub_diseases",
        properties=(
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
        ),
        description="Concrete diagnosis name or disease subtype used as the graph hub.",
    ),
    EntitySpec(
        label="Symptom",
        zh_name="症状/体征",
        entity_type="symptoms",
        properties=(
            "symptom_id",
            "symptom_name",
            "body_site",
            "symptom_category",
            "polarity",
            "typicality",
            "evidence_ids",
            "confidence",
        ),
        description="Clinical symptom or physical sign.",
    ),
    EntitySpec(
        label="Test",
        zh_name="检查/指标",
        entity_type="tests",
        properties=(
            "test_id",
            "test_name",
            "normal_range_min",
            "normal_range_max",
            "is_digital",
            "evidence_ids",
            "confidence",
        ),
        description="Diagnostic examination item or indicator required for a subtype.",
    ),
    EntitySpec(
        label="Treatment",
        zh_name="治疗原则",
        entity_type="treatments",
        properties=(
            "treatment_id",
            "treatment_content",
            "recommendation_strength",
            "source_section",
            "evidence_ids",
            "confidence",
        ),
        description="Principle-level treatment guidance followed by a subtype.",
    ),
    EntitySpec(
        label="Plan",
        zh_name="治疗方案",
        entity_type="plans",
        properties=(
            "plan_id",
            "plan_content",
            "plan_level",
            "applicable_condition",
            "contraindication_note",
            "evidence_ids",
            "confidence",
        ),
        description="Treatment plan that implements a treatment principle.",
    ),
    EntitySpec(
        label="Etiology",
        zh_name="病因",
        entity_type="etiologies",
        properties=(
            "etiology_id",
            "etiology_content",
            "etiology_type",
            "typicality",
            "diagnostic_role",
            "evidence_ids",
            "confidence",
        ),
        description="Initial cause of disease occurrence, usually upstream or external.",
    ),
    EntitySpec(
        label="Pathogenesis",
        zh_name="发病机制",
        entity_type="pathogeneses",
        properties=(
            "pathogenesis_id",
            "pathogenesis_content",
            "evidence_ids",
            "confidence",
        ),
        description="Functional, metabolic, or structural disease development mechanism.",
    ),
)

GRAPH_RELATIONS: tuple[RelationSpec, ...] = (
    RelationSpec(
        "Disease",
        "has_sub_disease",
        "Sub_disease",
        "包含分型",
        ("relation_id", "relation_name", "relation_type", "evidence_ids", "confidence"),
    ),
    RelationSpec(
        "Sub_disease",
        "manifests_as",
        "Symptom",
        "表现为",
        (
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
        ),
    ),
    RelationSpec(
        "Sub_disease",
        "requires_test",
        "Test",
        "需要检查",
        (
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
        ),
    ),
    RelationSpec(
        "Sub_disease",
        "follows_treatment",
        "Treatment",
        "遵循治疗原则",
        (
            "relation_id",
            "relation_name",
            "relation_type",
            "clinical_stage",
            "treatment_line",
            "applicable_condition",
            "recommendation_polarity",
            "recommendation_strength",
            "decision_basis",
            "contraindication_note",
            "evidence_level",
            "source_section",
            "confidence",
        ),
    ),
    RelationSpec(
        "Treatment",
        "implements_by",
        "Plan",
        "落实为",
        (
            "relation_id",
            "relation_name",
            "relation_type",
            "plan_role",
            "applicable_condition",
            "contraindication_note",
            "evidence_ids",
            "confidence",
        ),
    ),
    RelationSpec(
        "Etiology",
        "causes",
        "Sub_disease",
        "由病因导致",
        (
            "relation_id",
            "relation_name",
            "relation_type",
            "causal_strength",
            "typicality",
            "criterion_group_id",
            "required_count",
            "evidence_ids",
            "confidence",
        ),
    ),
    RelationSpec(
        "Sub_disease",
        "explained_by",
        "Pathogenesis",
        "由机制解释",
        ("relation_id", "relation_name", "relation_type", "evidence_ids", "confidence"),
    ),
)

ENTITY_LABELS = [entity.entity_type for entity in GRAPH_ENTITIES]
ACTIVE_ENTITY_LABELS = [
    entity.entity_type for entity in GRAPH_ENTITIES if entity.extract_enabled
]
ENTITY_PROPERTY_MAP = {
    entity.entity_type: list(entity.properties) for entity in GRAPH_ENTITIES
}
RELATION_PROPERTY_MAP = {
    relation.relation: list(relation.properties) for relation in GRAPH_RELATIONS
}
GRAPH_ENTITY_BY_TYPE = {entity.entity_type: entity for entity in GRAPH_ENTITIES}
