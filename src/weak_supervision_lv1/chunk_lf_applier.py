"""Apply chunk-level Labeling Functions."""

from typing import Any

from src.weak_supervision.common.lf_output import LFOutput
from src.weak_supervision_lv1.chunk_lf_base import ChunkLabelingFunction


def _chunk_key(chunk: dict[str, Any]) -> str:
    for key in ("chunk_id", "id"):
        if chunk.get(key):
            return str(chunk[key])
    return f"anon:{hash(str(chunk.get('text') or ''))}"


def apply_chunk_lfs(
    chunks: list[dict[str, Any]],
    labels: list[str],
    lfs: list[ChunkLabelingFunction],
) -> list[LFOutput]:
    """Apply all Lv1 LFs to all chunk-label pairs."""

    outputs: list[LFOutput] = []
    batched_by_lf: dict[str, dict[str, dict[str, LFOutput]]] = {}
    for lf in lfs:
        if hasattr(lf, "apply_batch"):
            batched_by_lf[lf.name] = lf.apply_batch(chunks, labels)  # type: ignore[attr-defined]

    for chunk in chunks:
        chunk_key = _chunk_key(chunk)
        for lf in lfs:
            if lf.name in batched_by_lf:
                label_outputs = batched_by_lf[lf.name].get(chunk_key)
                if label_outputs is None:
                    label_outputs = lf.apply_all(chunk, labels)
            else:
                label_outputs = lf.apply_all(chunk, labels)
            outputs.extend(label_outputs[label] for label in labels)
    return outputs
