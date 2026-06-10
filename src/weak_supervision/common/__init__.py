"""Common utilities shared by Lv1 and Lv2 weak supervision."""

from src.weak_supervision.common.graph_schema import (
    ACTIVE_ENTITY_LABELS,
    ENTITY_PROPERTY_MAP,
    GRAPH_ENTITIES,
    GRAPH_RELATIONS,
    RELATION_PROPERTY_MAP,
)
from src.weak_supervision.common.labels import ABSTAIN, ENTITY_LABELS, NEGATIVE
from src.weak_supervision.common.lf_output import EvidenceSpan, LFOutput

__all__ = [
    "ABSTAIN",
    "NEGATIVE",
    "ENTITY_LABELS",
    "ACTIVE_ENTITY_LABELS",
    "ENTITY_PROPERTY_MAP",
    "GRAPH_ENTITIES",
    "GRAPH_RELATIONS",
    "RELATION_PROPERTY_MAP",
    "EvidenceSpan",
    "LFOutput",
]
