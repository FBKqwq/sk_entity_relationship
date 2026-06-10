"""Normalize candidate entity text."""


def normalize_entity_text(text: str) -> str:
    """Normalize entity surface text for deduplication."""

    return text.strip()
