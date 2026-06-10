"""Link discovered PDF figures and tables to semantic chunks."""

from __future__ import annotations

import re
from typing import Any

from .layout_regions import bbox_horizontal_overlap_ratio, bbox_vertical_distance, build_chunk_page_regions

REF_RE = re.compile(r"(表|图|Table|Figure)\s*(\d+)", re.IGNORECASE)


def find_explicit_figure_table_refs(text: str) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for match in REF_RE.finditer(text):
        prefix = match.group(1)
        number = match.group(2)
        object_type = "table" if prefix.lower().startswith("table") or prefix == "表" else "figure"
        refs.append({"object_type": object_type, "number": number, "raw": match.group(0)})
    return refs


def _object_number(obj: dict[str, Any]) -> str | None:
    if obj.get("object_type") == "table":
        return obj.get("table_number") or _number_from_text(str(obj.get("caption", "")), "table")
    if obj.get("object_type") == "figure":
        return obj.get("figure_number") or _number_from_text(str(obj.get("caption", "")), "figure")
    return None


def _number_from_text(text: str, object_type: str) -> str | None:
    prefix = r"(?:表|Table)" if object_type == "table" else r"(?:图|Figure)"
    match = re.search(prefix + r"\s*(\d+)", text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _link_summary(obj: dict[str, Any], reason: str, confidence: float) -> dict[str, Any]:
    return {
        "object_id": obj.get("object_id"),
        "object_type": obj.get("object_type"),
        "page": obj.get("page"),
        "caption": obj.get("caption", ""),
        "bbox": obj.get("bbox", []),
        "link_reasons": [reason],
        "link_confidence": confidence,
        "ocr_status": obj.get("ocr_status", "pending"),
    }


def link_by_explicit_reference(chunk: dict[str, Any], objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    refs = find_explicit_figure_table_refs(str(chunk.get("text", "")))
    if not refs:
        return links
    for ref in refs:
        for obj in objects:
            if obj.get("object_type") != ref["object_type"]:
                continue
            if _object_number(obj) == ref["number"]:
                links.append(_link_summary(obj, "explicit_reference", 0.95))
    return links


def link_by_layout_proximity(
    chunk_regions: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    vertical_threshold = float(config.get("proximity_vertical_threshold", 120))
    overlap_threshold = float(config.get("proximity_horizontal_overlap_threshold", 0.30))
    candidates: list[tuple[float, dict[str, Any]]] = []
    for region in chunk_regions:
        region_bbox = region.get("bbox", [])
        page = int(region.get("page", 0) or 0)
        if len(region_bbox) != 4:
            continue
        for obj in objects:
            if int(obj.get("page", 0) or 0) != page:
                continue
            object_bbox = obj.get("bbox", [])
            if len(object_bbox) != 4:
                continue
            distance = bbox_vertical_distance(region_bbox, object_bbox)
            overlap = bbox_horizontal_overlap_ratio(region_bbox, object_bbox)
            if distance <= vertical_threshold and overlap >= overlap_threshold:
                score = distance - (overlap * 10.0)
                candidates.append((score, _link_summary(obj, "layout_proximity", 0.65)))
    candidates.sort(key=lambda item: item[0])
    return [item[1] for item in candidates]


def _merge_links(links: list[dict[str, Any]], max_links: int) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for link in links:
        object_id = str(link.get("object_id", ""))
        if not object_id:
            continue
        existing = merged.get(object_id)
        if existing is None:
            merged[object_id] = dict(link)
            continue
        existing_reasons = set(existing.get("link_reasons", []))
        existing_reasons.update(link.get("link_reasons", []))
        existing["link_reasons"] = sorted(existing_reasons)
        existing["link_confidence"] = max(float(existing.get("link_confidence", 0)), float(link.get("link_confidence", 0)))
    ordered = sorted(merged.values(), key=lambda item: float(item.get("link_confidence", 0)), reverse=True)
    return ordered[:max_links]


def attach_figure_table_links(
    payload: dict[str, Any],
    page_layouts: dict[int, list[dict[str, Any]]],
    figure_table_objects: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    max_links = int(config.get("max_linked_objects_per_chunk", 3))
    output = dict(payload)
    chunks = []
    for chunk in payload.get("chunks", []):
        next_chunk = dict(chunk)
        regions = build_chunk_page_regions(next_chunk, page_layouts)
        next_chunk["page_regions"] = regions
        explicit = link_by_explicit_reference(next_chunk, figure_table_objects) if bool(config.get("link_by_explicit_reference", True)) else []
        proximity = link_by_layout_proximity(regions, figure_table_objects, config) if bool(config.get("link_by_layout_proximity", True)) else []
        links = _merge_links(explicit + proximity, max_links)
        next_chunk["linked_tables"] = [link for link in links if link.get("object_type") == "table"]
        next_chunk["linked_figures"] = [link for link in links if link.get("object_type") == "figure"]
        chunks.append(next_chunk)
    output["chunks"] = chunks
    output["figure_table_objects"] = figure_table_objects
    return output
