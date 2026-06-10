"""chunk JSON 基础质量校验。"""

from __future__ import annotations

import math
import re
from typing import Any

NOISE_PATTERNS = [
    ("doi", r"DOI[:：]?\s*10\.\d{4,9}/[^\s]+|10\.\d{4,9}/[^\s]+"),
    ("received", r"收稿日期"),
    ("cite_this", r"引用本文"),
    ("editor", r"本文编辑"),
    ("page_marker", r"<<PAGE"),
    ("page_number", r"^[•・]\s*\d+\s*[•・]?$"),
]


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    index = int(math.ceil(percentile * len(sorted_values))) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def validate_chunk_payload(payload: dict[str, Any], *, min_chars: int = 200, max_chars: int = 1200) -> dict[str, Any]:
    """校验 chunk payload 的字段、span、长度与残留噪声。"""
    chunks = payload.get("chunks", [])
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    required_fields = ["chunk_id", "section_path", "section_title", "page_start", "page_end", "text_span", "text"]

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        for field in required_fields:
            if field not in chunk:
                issues.append({"type": "MISSING_FIELD", "chunk_id": chunk_id, "field": field})
        if chunk.get("page_start", 0) > chunk.get("page_end", 0):
            issues.append({"type": "BAD_PAGE_RANGE", "chunk_id": chunk_id})
        text_span = chunk.get("text_span", {})
        if text_span.get("start", 0) >= text_span.get("end", 0):
            issues.append({"type": "BAD_TEXT_SPAN", "chunk_id": chunk_id})
        text = chunk.get("text", "") or ""
        if not text.strip():
            issues.append({"type": "EMPTY_TEXT", "chunk_id": chunk_id})
        if len(text) > max_chars:
            issues.append({"type": "TOO_LONG", "chunk_id": chunk_id, "len": len(text), "max": max_chars})
        if len(text) < min_chars and len(chunks) > 1:
            warnings.append({"type": "TOO_SHORT", "chunk_id": chunk_id, "len": len(text), "min": min_chars})
        for name, pattern in NOISE_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
                issues.append({"type": "NOISE_IN_CHUNK", "chunk_id": chunk_id, "pattern_name": name})
        for region in chunk.get("page_regions", []) or []:
            bbox = region.get("bbox", [])
            if region.get("page") is None or not isinstance(bbox, list) or len(bbox) != 4:
                issues.append({"type": "BAD_PAGE_REGION", "chunk_id": chunk_id})
        for field, expected_type in (("linked_tables", "table"), ("linked_figures", "figure")):
            for linked in chunk.get(field, []) or []:
                if linked.get("object_type") != expected_type:
                    issues.append({"type": "BAD_LINKED_OBJECT_TYPE", "chunk_id": chunk_id, "field": field})
                if not linked.get("object_id"):
                    issues.append({"type": "MISSING_LINKED_OBJECT_ID", "chunk_id": chunk_id, "field": field})
                bbox = linked.get("bbox", [])
                if bbox and (not isinstance(bbox, list) or len(bbox) != 4):
                    issues.append({"type": "BAD_LINKED_OBJECT_BBOX", "chunk_id": chunk_id, "field": field})

    lengths = [len(chunk.get("text", "") or "") for chunk in chunks]
    return {
        "stage": "chunk_validation",
        "total_chunks": len(chunks),
        "length_stats": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "p50": _percentile(lengths, 0.50),
            "p90": _percentile(lengths, 0.90),
            "avg": (sum(lengths) / len(lengths)) if lengths else 0,
        },
        "warnings": warnings,
        "issues": issues,
        "pass": len(issues) == 0,
    }
