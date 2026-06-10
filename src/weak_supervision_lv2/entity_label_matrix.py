"""Build Lv2 entity-label LF matrices and analysis summaries."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.weak_supervision.common.labels import ABSTAIN, NEGATIVE
from src.weak_supervision.common.lf_output import LFOutput


def build_entity_label_matrix(
    entities: list[dict[str, Any]],
    labels: list[str],
    lf_outputs: list[LFOutput],
) -> tuple[list[list[int]], list[str]]:
    """Return a flattened ``entity-label x LF`` matrix.

    The row order is stable: for each entity in input order, iterate labels in
    the configured label order.  This mirrors the official Snorkel matrix
    contract while keeping a lightweight fallback available for environments
    where the package is not importable.
    """

    lf_names = sorted({output.lf_name for output in lf_outputs})
    by_key = {(id(output), output.lf_name): output.vote for output in lf_outputs}
    rows: list[list[int]] = []
    output_index = 0
    for _entity in entities:
        for label in labels:
            votes: dict[str, int] = {}
            for output in lf_outputs[output_index : output_index + len(lf_names)]:
                if output.label == label:
                    votes[output.lf_name] = output.vote
            rows.append([votes.get(name, ABSTAIN) for name in lf_names])
            output_index += len(lf_names)
    if not rows and by_key:
        rows = []
    return rows, lf_names


def analyze_lf_outputs(lf_outputs: list[LFOutput]) -> dict[str, Any]:
    """Compute coverage, overlap and conflict metrics for Lv2 LF traces."""

    by_point: dict[tuple[str, str], list[LFOutput]] = {}
    for output in lf_outputs:
        key = (str(output.metadata.get("entity_id") or ""), output.label)
        by_point.setdefault(key, []).append(output)

    lf_counts = Counter(output.lf_name for output in lf_outputs)
    covered = Counter(output.lf_name for output in lf_outputs if output.vote != ABSTAIN)
    conflicts = 0
    overlaps = 0
    for outputs in by_point.values():
        active_votes = [output.vote for output in outputs if output.vote != ABSTAIN]
        if len(active_votes) >= 2:
            overlaps += 1
        if any(vote > 0 for vote in active_votes) and any(vote == NEGATIVE for vote in active_votes):
            conflicts += 1

    return {
        "lf_count": len(lf_counts),
        "outputs": len(lf_outputs),
        "coverage_by_lf": {
            name: round(covered[name] / lf_counts[name], 4) if lf_counts[name] else 0.0
            for name in sorted(lf_counts)
        },
        "overlap_points": overlaps,
        "conflict_points": conflicts,
    }
