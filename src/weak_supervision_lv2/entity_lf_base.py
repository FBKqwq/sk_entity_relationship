"""Base protocol for entity-level Labeling Functions."""

from abc import ABC, abstractmethod
from typing import Any

from src.weak_supervision.common.lf_output import LFOutput


class EntityLabelingFunction(ABC):
    """Apply one LF to an entity-label pair."""

    name: str

    @abstractmethod
    def apply(self, entity: dict[str, Any], chunk: dict[str, Any], label: str) -> LFOutput:
        """Return a structured vote for one entity-label pair."""
