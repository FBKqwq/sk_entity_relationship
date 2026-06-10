"""Apply entity-level Labeling Functions."""

from typing import Any

from src.weak_supervision.common.lf_output import LFOutput
from src.weak_supervision_lv2.entity_lf_base import EntityLabelingFunction


def _entity_key(entity: dict[str, Any]) -> str:
    return str(entity.get("entity_id") or hash(str(entity)))


def apply_entity_lfs(
    entities: list[dict[str, Any]],
    chunk: dict[str, Any],
    labels: list[str],
    lfs: list[EntityLabelingFunction],
) -> list[LFOutput]:
    """Apply all Lv2 LFs to all entity-label pairs."""

    outputs: list[LFOutput] = []
    entity_chunk_pairs = [(entity, chunk) for entity in entities]
    batched_by_lf: dict[str, dict[str, dict[str, LFOutput]]] = {}
    for lf in lfs:
        lf_name = getattr(lf, "name", "")
        if lf_name and hasattr(lf, "apply_batch"):
            batched_by_lf[lf_name] = lf.apply_batch(entity_chunk_pairs, labels)  # type: ignore[attr-defined]

    for entity in entities:
        entity_key = _entity_key(entity)
        for lf in lfs:
            lf_name = getattr(lf, "name", "")
            if lf_name in batched_by_lf:
                label_outputs = batched_by_lf[lf_name].get(entity_key)
                if label_outputs is None and hasattr(lf, "apply_all"):
                    label_outputs = lf.apply_all(entity, chunk, labels)  # type: ignore[attr-defined]
            elif hasattr(lf, "apply_all"):
                label_outputs = lf.apply_all(entity, chunk, labels)  # type: ignore[attr-defined]
            else:
                label_outputs = {label: lf.apply(entity, chunk, label) for label in labels}
            outputs.extend(label_outputs[label] for label in labels)
    return outputs
