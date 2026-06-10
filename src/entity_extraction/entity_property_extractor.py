"""Extract final entity properties after Lv2 entity typing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from src.entity_extraction.entity_schema import (
    DEFAULT_PROPERTIES_BY_TYPE,
    ID_FIELD_BY_TYPE,
    NAME_FIELD_BY_TYPE,
    PROPERTY_FIELDS_BY_TYPE,
)
from src.utils.llm_client import chat_completion_text

LLMCallable = Callable[..., dict[str, Any]]


ENTITY_PROPERTY_SYSTEM_PROMPT = """你是医学知识图谱实体属性抽取器。
只能为已经通过 Lv2 定型的实体补充 schema 允许的属性；不得改变实体类型。
没有原文依据的属性保持空值/null/空数组，输出严格 JSON。
"""


def _extract_json_object(text: str) -> dict[str, Any] | None:
    clean = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.S)
    if fence_match:
        clean = fence_match.group(1)
    else:
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            clean = clean[start : end + 1]
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def default_entity_properties(
    *,
    entity_id: str,
    entity_type: str,
    name: str,
    confidence: float,
) -> dict[str, Any]:
    """Create schema-safe default properties for a final entity node."""

    properties = dict(DEFAULT_PROPERTIES_BY_TYPE.get(entity_type, {}))
    id_field = ID_FIELD_BY_TYPE.get(entity_type)
    name_field = NAME_FIELD_BY_TYPE.get(entity_type)
    if id_field:
        properties[id_field] = entity_id
    if name_field:
        properties[name_field] = name
    if "normalized_name" in properties and not properties["normalized_name"]:
        properties["normalized_name"] = name
    if "evidence_ids" in properties:
        properties["evidence_ids"] = [entity_id]
    if "source_document_ids" in properties:
        properties["source_document_ids"] = []
    if "confidence" in properties:
        properties["confidence"] = round(confidence, 6)
    return properties


def _sanitize_properties(entity_type: str, raw: Any, defaults: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    allowed = set(PROPERTY_FIELDS_BY_TYPE.get(entity_type, ()))
    properties = dict(defaults)
    for key, value in raw.items():
        if key in allowed:
            properties[key] = value
    return properties


def _property_prompt(entity: dict[str, Any], chunk: dict[str, Any], defaults: dict[str, Any]) -> str:
    entity_type = str(entity.get("final_entity_type") or entity.get("entity_type") or "")
    fields = list(PROPERTY_FIELDS_BY_TYPE.get(entity_type, ()))
    payload = {
        "entity_id": entity.get("entity_id"),
        "entity_type": entity_type,
        "name": entity.get("name"),
        "evidence_text": entity.get("evidence_text"),
        "status": entity.get("status"),
        "allowed_fields": fields,
        "property_template": defaults,
        "chunk": {
            "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
            "section_title": chunk.get("section_title"),
            "section_path": chunk.get("section_path", []),
            "text": chunk.get("text"),
        },
    }
    return f"""Extract properties for this final Lv2 typed entity.
Return strict JSON only.

Input:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Rules:
- Do not change entity_type or name.
- Fill only allowed_fields.
- Keep relation attributes out of entity properties.
- Unsupported attributes must remain default.

Output JSON:
{{
  "properties": {{}},
  "evidence": "minimal property evidence or empty string",
  "confidence": 0.0
}}
"""


def extract_entity_properties(
    label_results: list[dict[str, Any]],
    base_entities: dict[str, dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    *,
    include_review: bool = False,
    config_path: str | Path | None = None,
    llm_func: LLMCallable = chat_completion_text,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return property rows, final nodes, conflicts and raw traces."""

    node_statuses = {"accepted", "review"}
    property_rows: list[dict[str, Any]] = []
    entity_nodes: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for label in label_results:
        entity_id = str(label.get("entity_id") or "")
        base = base_entities.get(entity_id, {})
        entity_type = str(label.get("final_entity_type") or base.get("entity_type") or "")
        name = str(label.get("name") or base.get("name") or "")
        confidence = float(label.get("lv2_probability") or label.get("confidence") or 0.0)
        defaults = default_entity_properties(
            entity_id=entity_id,
            entity_type=entity_type,
            name=name,
            confidence=confidence,
        )
        status = str(label.get("status") or "")
        if status not in node_statuses:
            conflicts.append(
                {
                    "entity_id": entity_id,
                    "name": name,
                    "final_entity_type": entity_type,
                    "status": status,
                    "reasons": label.get("conflict_reasons") or ["not_selected_for_property_extraction"],
                }
            )
            continue
        chunk = chunks_by_id.get(str(label.get("chunk_id") or base.get("chunk_id") or ""), {})
        should_extract_properties = status == "accepted" or (status == "review" and include_review)
        result: dict[str, Any] = {}
        parsed: dict[str, Any] | None = None
        if should_extract_properties:
            prompt = _property_prompt(label, chunk, defaults)
            result = llm_func(prompt, system_prompt=ENTITY_PROPERTY_SYSTEM_PROMPT, config_path=config_path)
            parsed = _extract_json_object(str(result.get("text") or "")) if result.get("status") == "ok" else None
            raw_rows.append({"entity_id": entity_id, "phase": "entity_properties", "raw_response": result, "parsed": parsed})
        property_status = "ok" if parsed else "fallback_defaults"
        if status == "review" and not include_review:
            property_status = "review_defaults"
        properties = _sanitize_properties(entity_type, parsed.get("properties") if parsed else {}, defaults)
        prop_conf = parsed.get("confidence") if parsed else None
        if prop_conf not in (None, ""):
            try:
                properties["confidence"] = max(confidence, min(1.0, float(prop_conf)))
            except (TypeError, ValueError):
                properties["confidence"] = confidence
        else:
            properties["confidence"] = confidence
        property_row = {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "name": name,
            "status": property_status,
            "properties": properties,
            "property_evidence": str(parsed.get("evidence") or "") if parsed else "",
            "property_confidence": properties.get("confidence"),
        }
        property_rows.append(property_row)
        if property_status != "ok":
            reason = "review_entity_property_extraction_not_enabled"
            if should_extract_properties:
                reason = str(result.get("reason") or "property_llm_unavailable_or_unparseable")
            conflicts.append(
                {
                    "entity_id": entity_id,
                    "name": name,
                    "final_entity_type": entity_type,
                    "status": "property_incomplete",
                    "reasons": [reason],
                }
            )
        entity_nodes.append(
            {
                "entity_id": entity_id,
                "document_id": label.get("document_id") or base.get("document_id"),
                "chunk_id": label.get("chunk_id") or base.get("chunk_id"),
                "section_title": label.get("section_title") or base.get("section_title"),
                "section_path": label.get("section_path") or base.get("section_path", []),
                "entity_type": entity_type,
                "name": name,
                "content": name,
                "properties": properties,
                "evidence_text": label.get("evidence_text") or base.get("evidence_text") or "",
                "confidence": properties.get("confidence", confidence),
                "source": "lv2_entity_property_extraction",
                "status": status,
                "entity_status": status,
                "property_status": property_status,
                "source_lfs": label.get("source_lfs", []),
                "lv2_probability": label.get("lv2_probability"),
                "top2_gap": label.get("top2_gap"),
            }
        )
    return property_rows, entity_nodes, conflicts, raw_rows
