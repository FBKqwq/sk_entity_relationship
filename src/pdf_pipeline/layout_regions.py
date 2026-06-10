"""Map chunks back to approximate PDF page regions."""

from __future__ import annotations

from typing import Any


def union_bboxes(bboxes: list[list[float]]) -> list[float]:
    if not bboxes:
        return []
    return [
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    ]


def bbox_vertical_distance(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return float("inf")
    if a[3] < b[1]:
        return b[1] - a[3]
    if b[3] < a[1]:
        return a[1] - b[3]
    return 0.0


def bbox_horizontal_overlap_ratio(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    width = max(1.0, min(a[2] - a[0], b[2] - b[0]))
    return overlap / width


def _normalize_text(text: str) -> str:
    return "".join(str(text).split())


def _line_overlaps_chunk(line_text: str, chunk_text: str) -> bool:
    line = _normalize_text(line_text)
    chunk = _normalize_text(chunk_text)
    if not line or not chunk:
        return False
    if line in chunk:
        return True
    probe_len = min(len(line), 16)
    return probe_len >= 8 and line[:probe_len] in chunk


def build_page_layouts(pages: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    layouts: dict[int, list[dict[str, Any]]] = {}
    for page in pages:
        page_number = int(page["page_number"])
        meta = page.get("meta", {}) or {}
        lines = meta.get("layout_lines", []) or []
        layouts[page_number] = [line for line in lines if isinstance(line, dict)]
    return layouts


def build_chunk_page_regions(chunk: dict[str, Any], page_layouts: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    chunk_text = str(chunk.get("text", ""))
    page_start = int(chunk.get("page_start", 0) or 0)
    page_end = int(chunk.get("page_end", page_start) or page_start)
    regions: list[dict[str, Any]] = []

    for page_number in range(page_start, page_end + 1):
        lines = page_layouts.get(page_number, [])
        matched = [
            list(line["bbox"])
            for line in lines
            if isinstance(line.get("bbox"), list) and _line_overlaps_chunk(str(line.get("text", "")), chunk_text)
        ]
        if not matched and lines:
            matched = [list(line["bbox"]) for line in lines if isinstance(line.get("bbox"), list)]
        bbox = union_bboxes(matched)
        if bbox:
            regions.append({"page": page_number, "bbox": bbox})
    return regions
