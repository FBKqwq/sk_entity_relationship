"""Base protocol for chunk-level Labeling Functions."""

from abc import ABC, abstractmethod
from typing import Any

from src.weak_supervision.common.lf_output import LFOutput


class ChunkLabelingFunction(ABC):
    """Apply one LF to a chunk-label pair."""

    name: str

    @abstractmethod
    def apply(self, chunk: dict[str, Any], label: str) -> LFOutput:
        """Return a structured vote for one chunk-label pair."""

    def apply_all(self, chunk: dict[str, Any], labels: list[str]) -> dict[str, LFOutput]:
        """Return votes for all labels for one chunk.

        Lv1 is a multi-label layer: every LF should judge every active graph
        entity type for the same chunk. The per-label `apply` method is kept
        as the compatibility surface used by older matrix builders.
        """

        return {label: self.apply(chunk, label) for label in labels}
