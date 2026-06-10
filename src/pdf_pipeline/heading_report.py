"""章节命中率报告生成。"""

from __future__ import annotations

from collections import Counter
from typing import Any

GENERIC_TITLES = {"正文", "前言/摘要", "Abstract"}
SUSPICIOUS_TOKENS = ("vs.", "RR", "CI", "P=", "P<", "g/d", "mg", "%")


def _is_suspicious_title(title: str) -> tuple[bool, str]:
    clean = (title or "").strip()
    if not clean:
        return True, "empty_title"
    if len(clean) > 80:
        return True, "title_too_long"
    for token in SUSPICIOUS_TOKENS:
        if token.lower() in clean.lower():
            return True, f"contains_token:{token}"
    digit_count = sum(char.isdigit() for char in clean)
    if digit_count >= 4 and len(clean) <= 12:
        return True, "dense_digits_short_title"
    return False, ""


def build_heading_hit_report(payload: dict[str, Any]) -> dict[str, Any]:
    """按 chunk 统计章节命中率并输出可解释报告。"""
    chunks = payload.get("chunks", [])
    total_chunks = len(chunks)
    if total_chunks == 0:
        return {
            "stage": "heading_hit_report",
            "version": "v1",
            "doc_id": payload.get("doc_id"),
            "source_title": payload.get("source_title"),
            "calculation_method": {
                "heading_hit_rate": "non_generic_heading_chunks / total_chunks",
                "hierarchical_hit_rate": "hierarchical_chunks / total_chunks",
                "weighted_hit_score": "0.7 * heading_hit_rate + 0.3 * hierarchical_hit_rate",
            },
            "metrics": {
                "total_chunks": 0,
                "non_generic_heading_chunks": 0,
                "hierarchical_chunks": 0,
                "heading_hit_rate": 0.0,
                "hierarchical_hit_rate": 0.0,
                "weighted_hit_score": 0.0,
            },
            "title_stats": {"generic_titles": sorted(GENERIC_TITLES), "top_titles": []},
            "suspicious_titles": [],
        }

    non_generic_heading_chunks = sum(1 for chunk in chunks if chunk.get("section_title") not in GENERIC_TITLES)
    hierarchical_chunks = sum(1 for chunk in chunks if len(chunk.get("section_path", [])) >= 3)
    heading_hit_rate = non_generic_heading_chunks / total_chunks
    hierarchical_hit_rate = hierarchical_chunks / total_chunks
    weighted_hit_score = 0.7 * heading_hit_rate + 0.3 * hierarchical_hit_rate

    depth_counter = Counter(len(chunk.get("section_path", [])) for chunk in chunks)
    title_counter = Counter(str(chunk.get("section_title", "")).strip() for chunk in chunks)

    suspicious_titles: list[dict[str, Any]] = []
    for chunk in chunks:
        title = str(chunk.get("section_title", "")).strip()
        bad, reason = _is_suspicious_title(title)
        if not bad:
            continue
        suspicious_titles.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "title": title,
                "reason": reason,
                "section_path": chunk.get("section_path", []),
            }
        )

    return {
        "stage": "heading_hit_report",
        "version": "v1",
        "doc_id": payload.get("doc_id"),
        "source_title": payload.get("source_title"),
        "calculation_method": {
            "overview": "命中率按 chunk 级别统计，分母均为 total_chunks。",
            "heading_hit_rate": "non_generic_heading_chunks / total_chunks",
            "hierarchical_hit_rate": "hierarchical_chunks / total_chunks",
            "weighted_hit_score": "0.7 * heading_hit_rate + 0.3 * hierarchical_hit_rate",
            "generic_title_definition": sorted(GENERIC_TITLES),
            "hierarchical_chunk_definition": "len(section_path) >= 3",
            "suspicious_title_definition": {
                "title_too_long": "标题长度超过 80 字符",
                "contains_token": list(SUSPICIOUS_TOKENS),
                "dense_digits_short_title": "短标题中数字过密，疑似 OCR 或统计行",
            },
        },
        "metrics": {
            "total_chunks": total_chunks,
            "non_generic_heading_chunks": non_generic_heading_chunks,
            "hierarchical_chunks": hierarchical_chunks,
            "heading_hit_rate": round(heading_hit_rate, 4),
            "hierarchical_hit_rate": round(hierarchical_hit_rate, 4),
            "weighted_hit_score": round(weighted_hit_score, 4),
            "path_depth_distribution": dict(sorted(depth_counter.items())),
        },
        "title_stats": {
            "generic_titles": sorted(GENERIC_TITLES),
            "top_titles": [
                {"title": title, "count": count}
                for title, count in title_counter.most_common(15)
            ],
        },
        "suspicious_titles": suspicious_titles,
    }
