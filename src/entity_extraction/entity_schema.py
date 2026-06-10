"""Entity candidate schema helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.weak_supervision.common.graph_schema import ACTIVE_ENTITY_LABELS, ENTITY_PROPERTY_MAP


ACTIVE_PRELABEL_TYPES = tuple(ACTIVE_ENTITY_LABELS)
DOCUMENT_LEVEL_TYPES = ("diseases",)

LIST_FIELD_BY_TYPE = {
    "diseases": "diseases_list",
    "sub_diseases": "sub_diseases_list",
    "symptoms": "symptoms_list",
    "tests": "tests_list",
    "treatments": "treatments_list",
    "plans": "plans_list",
    "etiologies": "etiologies_list",
    "pathogeneses": "pathogeneses_list",
}

NAME_FIELD_BY_TYPE = {
    "diseases": "disease_name",
    "sub_diseases": "sub_disease_name",
    "symptoms": "symptom_name",
    "tests": "test_name",
    "treatments": "treatment_content",
    "plans": "plan_content",
    "etiologies": "etiology_content",
    "pathogeneses": "pathogenesis_content",
}

ID_FIELD_BY_TYPE = {
    "diseases": "disease_id",
    "sub_diseases": "sub_disease_id",
    "symptoms": "symptom_id",
    "tests": "test_id",
    "treatments": "treatment_id",
    "plans": "plan_id",
    "etiologies": "etiology_id",
    "pathogeneses": "pathogenesis_id",
}

PROPERTY_FIELDS_BY_TYPE = {
    entity_type: tuple(fields)
    for entity_type, fields in ENTITY_PROPERTY_MAP.items()
}


def _default_properties(entity_type: str) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for field_name in PROPERTY_FIELDS_BY_TYPE.get(entity_type, ()):
        if field_name in {
            "aliases",
            "source_document_ids",
            "evidence_ids",
            "decision_basis",
        }:
            defaults[field_name] = []
        elif field_name in {
            "confidence",
            "age_min",
            "age_max",
            "normal_range_min",
            "normal_range_max",
            "plan_level",
        }:
            defaults[field_name] = None
        elif field_name in {"is_digital", "gold_standard"}:
            defaults[field_name] = None
        else:
            defaults[field_name] = ""
    return defaults


DEFAULT_PROPERTIES_BY_TYPE: dict[str, dict[str, Any]] = {
    entity_type: _default_properties(entity_type)
    for entity_type in LIST_FIELD_BY_TYPE
}
DEFAULT_PROPERTIES_BY_TYPE["sub_diseases"].update(
    {
        "clinical_stage": "不限定",
        "gender": "all",
    }
)
DEFAULT_PROPERTIES_BY_TYPE["symptoms"].update(
    {
        "polarity": "present",
        "typicality": "unknown",
    }
)
DEFAULT_PROPERTIES_BY_TYPE["etiologies"].update(
    {
        "typicality": "unknown",
        "diagnostic_role": "unknown",
    }
)


@dataclass(frozen=True)
class PrelabelEntity:
    """One Teacher LLM pre-labeled candidate entity."""

    entity_type: str
    name: str
    properties: dict[str, Any] = field(default_factory=dict)
    evidence: str = ""
    confidence: float = 0.0
    override_lv1: bool = False

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serializable candidate record."""

        return {
            "entity_type": self.entity_type,
            "name": self.name,
            "properties": self.properties,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "override_lv1": self.override_lv1,
        }


def empty_prelabel_response() -> dict[str, Any]:
    """Return the normalized empty response shape."""

    result = {field: [] for field in LIST_FIELD_BY_TYPE.values()}
    result["lv1_overrides"] = []
    result["entities"] = []
    return result
