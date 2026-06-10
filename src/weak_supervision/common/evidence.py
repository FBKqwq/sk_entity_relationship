"""Evidence helpers shared by Lv1 and Lv2."""

from src.weak_supervision.common.lf_output import EvidenceSpan


def make_evidence_span(start: int, end: int, text: str, source: str | None = None) -> EvidenceSpan:
    """Create a normalized evidence span."""

    return EvidenceSpan(start=start, end=end, text=text, source=source)
