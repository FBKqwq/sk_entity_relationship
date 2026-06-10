"""Build Lv1 chunk label result records from LF outputs."""

from __future__ import annotations

from typing import Any

from src.weak_supervision.common.lf_output import LFOutput
from src.weak_supervision_lv1.chunk_count_regression import (
    lv1_count_config_from_project_config,
    manual_count_fusion,
)
from src.weak_supervision_lv1.chunk_vote_model import (
    lv1_vote_config_from_project_config,
    manual_weighted_vote_label,
)


def build_chunk_label_results(
    chunk: dict[str, Any],
    labels: list[str],
    lf_outputs: list[LFOutput],
    *,
    project_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build fused Lv1 records for one chunk across all active labels."""

    vote_config = lv1_vote_config_from_project_config(project_config)
    count_config = lv1_count_config_from_project_config(project_config)
    records: list[dict[str, Any]] = []
    for label in labels:
        decision = manual_weighted_vote_label(label, lf_outputs, vote_config=vote_config)
        record = decision.to_record()
        count_decision = manual_count_fusion(
            label,
            lf_outputs,
            decision,
            count_config=count_config,
        )
        record.update(count_decision.to_record())
        record.update(
            {
                "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
                "document_id": chunk.get("document_id"),
                "section_title": chunk.get("section_title"),
                "section_path": chunk.get("section_path", []),
            }
        )
        records.append(record)
    return records


def build_chunk_label_results_for_chunks(
    chunks: list[dict[str, Any]],
    labels: list[str],
    outputs_by_chunk_id: dict[str, list[LFOutput]],
    *,
    project_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build fused Lv1 records for multiple chunks."""

    records: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
        records.extend(
            build_chunk_label_results(
                chunk,
                labels,
                outputs_by_chunk_id.get(chunk_id, []),
                project_config=project_config,
            )
        )
    return records
