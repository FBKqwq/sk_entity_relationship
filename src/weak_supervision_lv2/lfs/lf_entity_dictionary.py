"""Entity suggested-type and dictionary Lv2 labeling functions."""

from __future__ import annotations

from typing import Any

from src.weak_supervision.common.lf_output import LFOutput
from src.weak_supervision.common.labels import ABSTAIN, NEGATIVE
from src.weak_supervision_lv2.entity_lf_base import EntityLabelingFunction


class EntitySuggestedTypeLF(EntityLabelingFunction):
    """Use the candidate type as a weak but not final Lv2 vote."""

    name = "lv2_entity_suggested_type"

    def apply(self, entity: dict[str, Any], chunk: dict[str, Any], label: str) -> LFOutput:
        entity_id = entity.get("entity_id")
        suggested = str(entity.get("entity_type") or "")
        if not suggested:
            return LFOutput(self.name, label, vote=ABSTAIN, metadata={"entity_id": entity_id})
        if suggested == label:
            return LFOutput(self.name, label, vote=1, confidence=0.72, count=1, metadata={"entity_id": entity_id})
        return LFOutput(self.name, label, vote=NEGATIVE, confidence=0.25, metadata={"entity_id": entity_id})
