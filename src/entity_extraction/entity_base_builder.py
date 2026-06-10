"""Build entity_base records from LLM candidate entities."""

from __future__ import annotations

from typing import Any

from src.entity_extraction.entity_normalizer import normalize_entity_text
from src.utils.hashing import stable_entity_id


def build_entity_base_records(
    chunk: dict[str, Any],
    entities: list[dict[str, Any]],
    *,
    source: str = "teacher_llm_prelabel",
) -> list[dict[str, Any]]:
    """Build traceable entity_base records from normalized pre-label entities."""

    chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
    document_id = chunk.get("document_id")
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for entity in entities:
        entity_type = str(entity.get("entity_type") or "")
        name = normalize_entity_text(str(entity.get("name") or ""))
        if not entity_type or not name:
            continue
        key = (entity_type, name)
        if key in seen:
            continue
        seen.add(key)
        entity_id = stable_entity_id(str(document_id or ""), chunk_id, entity_type, name)
        candidate_properties = entity.get("properties", {})
        if not isinstance(candidate_properties, dict):
            candidate_properties = {}
        else:
            candidate_properties = dict(candidate_properties)
        records.append(
            {
                "entity_id": entity_id,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "section_title": chunk.get("section_title"),
                "section_path": chunk.get("section_path", []),
                "entity_type": entity_type,
                "name": name,
                "content": name,
                "candidate_properties": candidate_properties,
                "properties": {},
                "evidence_text": str(entity.get("evidence") or ""),
                "confidence": float(entity.get("confidence") or 0.0),
                "source": source,
                "override_lv1": bool(entity.get("override_lv1", False)),
                "status": "candidate",
            }
        )
    return records
