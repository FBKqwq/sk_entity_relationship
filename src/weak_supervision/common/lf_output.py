"""Common structured output returned by Labeling Functions."""

from dataclasses import dataclass, field
from typing import Any

from src.weak_supervision.common.labels import ABSTAIN


@dataclass(frozen=True)
class EvidenceSpan:
    """Evidence text and offsets relative to the active text surface."""

    start: int
    end: int
    text: str
    source: str | None = None


@dataclass(frozen=True)
class LFOutput:
    """Normalized LF output used by both Lv1 and Lv2."""

    lf_name: str
    label: str
    vote: int = ABSTAIN
    confidence: float = 0.0
    count: int = 0
    evidence: list[EvidenceSpan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        """Whether this LF returned no usable vote."""

        return self.vote == ABSTAIN
