"""Shared helpers for Lv1 chunk-level Labeling Functions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from src.utils.io import read_yaml
from src.weak_supervision.common.lf_output import EvidenceSpan, LFOutput


DEFAULT_LABELS = (
    "sub_diseases",
    "symptoms",
    "tests",
    "treatments",
    "plans",
    "etiologies",
    "pathogeneses",
)


def chunk_text(chunk: dict[str, Any]) -> str:
    """Return the text surface used by chunk-level LFs."""

    return str(chunk.get("text") or "")


def section_surface(chunk: dict[str, Any]) -> str:
    """Return normalized section title/path text for section-prior LFs."""

    parts: list[str] = []
    title = chunk.get("section_title")
    if title:
        parts.append(str(title))
    path = chunk.get("section_path")
    if isinstance(path, list):
        parts.extend(str(item) for item in path if item)
    elif path:
        parts.append(str(path))
    return " / ".join(parts)


def make_output(
    lf_name: str,
    label: str,
    *,
    vote: int = 0,
    confidence: float = 0.0,
    evidence: Iterable[EvidenceSpan] | None = None,
    metadata: dict[str, Any] | None = None,
    count: int | None = None,
) -> LFOutput:
    """Build a bounded, normalized LF output."""

    evidence_list = list(evidence or [])
    inferred_count = len(evidence_list) if count is None else count
    return LFOutput(
        lf_name=lf_name,
        label=label,
        vote=vote,
        confidence=max(0.0, min(1.0, confidence)),
        count=max(0, inferred_count),
        evidence=evidence_list,
        metadata=metadata or {},
    )


def abstain(lf_name: str, label: str, reason: str = "no_signal") -> LFOutput:
    """Return a standard abstain output."""

    return make_output(lf_name, label, metadata={"reason": reason})


def find_literal_spans(text: str, term: str, *, max_spans: int = 8) -> list[EvidenceSpan]:
    """Find direct literal spans in chunk text.

    Matching is intentionally character-based to keep LF evidence explainable.
    ASCII terms are matched case-insensitively; Chinese and mixed terms preserve
    the original surface offsets.
    """

    clean_term = term.strip()
    if not text or not clean_term:
        return []

    flags = re.IGNORECASE if clean_term.isascii() else 0
    spans: list[EvidenceSpan] = []
    for match in re.finditer(re.escape(clean_term), text, flags):
        spans.append(EvidenceSpan(match.start(), match.end(), text[match.start() : match.end()]))
        if len(spans) >= max_spans:
            break
    return spans


def find_regex_spans(
    text: str,
    pattern: str,
    *,
    flags: int = re.IGNORECASE,
    max_spans: int = 8,
    source: str | None = None,
) -> list[EvidenceSpan]:
    """Find regex spans with defensive handling for malformed patterns."""

    if not text:
        return []
    spans: list[EvidenceSpan] = []
    try:
        iterator = re.finditer(pattern, text, flags)
    except re.error:
        return []
    for match in iterator:
        spans.append(
            EvidenceSpan(
                match.start(),
                match.end(),
                text[match.start() : match.end()],
                source=source,
            )
        )
        if len(spans) >= max_spans:
            break
    return spans


def default_config_path() -> Path:
    """Return the project weak supervision config path."""

    return Path(__file__).resolve().parents[3] / "configs" / "weak_supervision.yaml"


def load_weak_supervision_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load weak supervision config, returning an empty object if absent."""

    path = Path(config_path) if config_path else default_config_path()
    if not path.exists():
        return {}
    return read_yaml(path)


def active_labels_from_config(config: dict[str, Any]) -> list[str]:
    """Read active labels from config, falling back to the graph defaults."""

    labels = config.get("active_labels") or config.get("labels") or list(DEFAULT_LABELS)
    return [str(label) for label in labels if str(label) in DEFAULT_LABELS]


def merge_evidence(*groups: Iterable[EvidenceSpan], limit: int = 12) -> list[EvidenceSpan]:
    """Merge evidence spans by offset/text while preserving order."""

    seen: set[tuple[int, int, str]] = set()
    merged: list[EvidenceSpan] = []
    for group in groups:
        for span in group:
            key = (span.start, span.end, span.text)
            if key in seen:
                continue
            seen.add(key)
            merged.append(span)
            if len(merged) >= limit:
                return merged
    return merged
