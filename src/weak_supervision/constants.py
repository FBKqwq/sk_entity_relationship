"""Compatibility constants for weak-supervision modules.

New code should import shared labels and vote constants from
``src.weak_supervision.common.labels``.
"""

from src.weak_supervision.common.labels import ABSTAIN, ENTITY_LABELS, NEGATIVE

__all__ = ["ABSTAIN", "NEGATIVE", "ENTITY_LABELS"]
