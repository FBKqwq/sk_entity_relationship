"""章节优先的语义 chunk 切分。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pdf_pipeline.page_cleaner import remove_non_knowledge_tail
from src.pdf_pipeline.section_detector import Section, detect_sections, find_subsection_spans
from src.pdf_pipeline.toc_llm_pipeline import build_sections_with_dual_toc
from src.utils.hashing import sha1_12, stable_doc_id
from src.utils.text_span import build_fulltext_and_maps, global_to_page_span

SENTENCE_ENDINGS = "。！？；;.!?"
MIN_SECTION_MERGE_CHARS = 500


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    content_count = sum(1 for char in text if not char.isspace())
    return cjk_count / max(1, content_count)


def _should_drop_chunk_by_language(text: str, config: dict[str, Any]) -> tuple[bool, str, float]:
    cfg = config.get("chunk_language_filter") or {}
    if not bool(cfg.get("enabled", True)):
        return False, "", _cjk_ratio(text)
    min_chars = int(cfg.get("min_chars", 80))
    min_cjk_ratio = float(cfg.get("min_cjk_ratio", 0.20))
    cjk_ratio = _cjk_ratio(text)
    if len(text.strip()) >= min_chars and cjk_ratio < min_cjk_ratio:
        return True, "low_cjk_ratio", cjk_ratio
    return False, "", cjk_ratio


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """按段落返回局部 span。"""
    spans: list[tuple[int, int]] = []
    position = 0
    for block in text.split("\n\n"):
        if not block.strip():
            position += len(block) + 2
            continue
        start = text.find(block, position)
        end = start + len(block)
        spans.append((start, end))
        position = end + 2
    return spans or [(0, len(text))]


def _sentence_safe_cut(text: str, max_chars: int) -> int:
    """尽量在句末切分，避免截断医学数值或句子。"""
    if len(text) <= max_chars:
        return len(text)
    window_start = max(0, int(max_chars * 0.65))
    best = -1
    for index in range(max_chars, window_start, -1):
        if text[index - 1] in SENTENCE_ENDINGS:
            best = index
            break
    return best if best > 0 else max_chars


def _merge_spans_to_chunks(
    text: str,
    spans: list[tuple[int, int]],
    *,
    min_chars: int,
    max_chars: int,
) -> list[tuple[str, int, int]]:
    """合并段落或小节 span 到目标 chunk 长度。"""
    chunks: list[tuple[str, int, int]] = []
    buffer = ""
    buffer_start: int | None = None
    buffer_end = 0

    def flush() -> None:
        nonlocal buffer, buffer_start, buffer_end
        if buffer_start is None or not buffer.strip():
            buffer = ""
            buffer_start = None
            return
        chunks.append((buffer.strip(), buffer_start, buffer_end))
        buffer = ""
        buffer_start = None

    for start, end in spans:
        piece = text[start:end].strip()
        if not piece:
            continue
        if buffer_start is None:
            buffer_start = start
            buffer = piece
            buffer_end = end
            continue
        candidate = f"{buffer}\n\n{piece}"
        if len(candidate) > max_chars and len(buffer) >= min_chars:
            flush()
            buffer_start = start
            buffer = piece
            buffer_end = end
        else:
            buffer = candidate
            buffer_end = end
    flush()

    final_chunks: list[tuple[str, int, int]] = []
    for chunk_text, start, end in chunks:
        if len(chunk_text) <= max_chars:
            final_chunks.append((chunk_text, start, end))
            continue
        local_offset = 0
        remaining = chunk_text
        while remaining:
            cut = _sentence_safe_cut(remaining, max_chars)
            piece = remaining[:cut].strip()
            if piece:
                piece_start = start + local_offset
                final_chunks.append((piece, piece_start, piece_start + len(piece)))
            local_offset += cut
            remaining = remaining[cut:].lstrip()
    return final_chunks


def chunk_section(section: Section, full_text: str, *, min_chars: int, max_chars: int) -> list[tuple[list[str], str, int, int]]:
    """将单个章节切分为 chunk。"""
    section_text = full_text[section.start : section.end].strip()
    if not section_text:
        return []
    if len(section_text) <= max_chars:
        return [(section.path, section_text, section.start, section.end)]

    local_spans = find_subsection_spans(section_text) or _paragraph_spans(section_text)
    local_chunks = _merge_spans_to_chunks(section_text, local_spans, min_chars=min_chars, max_chars=max_chars)
    return [(section.path, text, section.start + local_start, section.start + local_end) for text, local_start, local_end in local_chunks]


def _merge_small_same_level_chunks(
    chunks: list[tuple[list[str], str, int, int]],
    *,
    min_merge_chars: int,
    max_chars: int,
) -> list[tuple[list[str], str, int, int]]:
    """Merge safe short chunks without assigning sibling content to the wrong section."""
    if not chunks:
        return []
    merged: list[tuple[list[str], str, int, int]] = []
    for path, text, start, end in chunks:
        clean_text = text.strip()
        if not clean_text:
            continue
        if not merged:
            merged.append((path, clean_text, start, end))
            continue
        prev_path, prev_text, prev_start, prev_end = merged[-1]
        same_section = prev_path == path
        same_level_sibling = len(prev_path) == len(path) and prev_path[:-1] == path[:-1]
        named_common_parent = same_level_sibling and len(prev_path[:-1]) >= 2
        prev_is_parent = len(prev_path) < len(path) and path[: len(prev_path)] == prev_path
        should_merge = (len(prev_text) < min_merge_chars or len(clean_text) < min_merge_chars) and same_section
        should_merge_siblings_to_parent = (
            named_common_parent
            and (len(prev_text) < min_merge_chars or len(clean_text) < min_merge_chars)
        )
        should_merge_parent = (
            prev_is_parent
            and len(prev_path) >= 2
            and (len(prev_text) < min_merge_chars or len(clean_text) < min_merge_chars)
        )
        if should_merge and len(prev_text) + len(clean_text) + 2 <= max_chars:
            merged[-1] = (prev_path, f"{prev_text}\n\n{clean_text}", prev_start, end)
        elif should_merge_siblings_to_parent and len(prev_text) + len(clean_text) + 2 <= max_chars:
            parent_path = prev_path[:-1]
            merged[-1] = (parent_path, f"{prev_text}\n\n{clean_text}", prev_start, end)
        elif should_merge_parent and len(prev_text) + len(clean_text) + 2 <= max_chars:
            merged[-1] = (prev_path, f"{prev_text}\n\n{clean_text}", prev_start, end)
        else:
            merged.append((path, clean_text, start, end))
    return merged


def build_chunk_payload(
    pages: list[dict[str, Any]],
    *,
    pdf_path: str | Path,
    chunking_config: dict[str, Any] | None = None,
    pdf_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """从页文本构建 Skill 约定的 chunk JSON payload。"""
    config = chunking_config or {}
    max_chars = int(config.get("max_chars", 1200))
    min_chars = int(config.get("min_chars", 200))
    if config.get("min_section_merge_chars", None) is None:
        min_section_merge_chars = min_chars
    else:
        min_section_merge_chars = int(config.get("min_section_merge_chars", 0))
    method = str(config.get("method", "section_semantic_v1"))

    page_text = {int(page["page_number"]): str(page.get("text", "")) for page in pages}
    full_text, char_to_page, page_start_global = build_fulltext_and_maps(page_text)
    if bool(config.get("filter_non_knowledge_section", True)):
        full_text = remove_non_knowledge_tail(full_text)
        char_to_page = char_to_page[: len(full_text)]
    toc_report: dict[str, Any] | None = None
    pdf_cfg = pdf_config or {}
    toc_cfg = pdf_cfg.get("toc_llm_pipeline") or {}
    if bool(toc_cfg.get("enabled", False)):
        sections, toc_report = build_sections_with_dual_toc(
            page_text,
            full_text,
            char_to_page,
            total_pages=len(pages),
            llm_config_path=pdf_cfg.get("llm_config", "configs/llm.yaml"),
            config=toc_cfg,
        )
    else:
        sections = detect_sections(full_text)

    local_chunks: list[tuple[list[str], str, int, int]] = []
    for section in sections:
        local_chunks.extend(chunk_section(section, full_text, min_chars=min_chars, max_chars=max_chars))

    merged_chunks = _merge_small_same_level_chunks(
        local_chunks,
        min_merge_chars=min_section_merge_chars,
        max_chars=max_chars,
    )

    chunks: list[dict[str, Any]] = []
    dropped_chunks: list[dict[str, Any]] = []
    for section_path, chunk_text, start, end in merged_chunks:
        if not chunk_text.strip():
            continue
        should_drop, drop_reason, cjk_ratio = _should_drop_chunk_by_language(chunk_text, config)
        if should_drop:
            page_start, page_end, span_start, span_end = global_to_page_span(char_to_page, page_start_global, start, end)
            dropped_chunks.append(
                {
                    "reason": drop_reason,
                    "section_title": section_path[-1],
                    "section_path": section_path,
                    "page_start": page_start,
                    "page_end": page_end,
                    "page_span": {
                        "start": {"page": span_start.page, "offset": span_start.offset},
                        "end": {"page": span_end.page, "offset": span_end.offset},
                    },
                    "text_span": {"start": start, "end": end},
                    "len": len(chunk_text.strip()),
                    "cjk_ratio": round(cjk_ratio, 4),
                    "head_80": chunk_text.strip()[:80],
                }
            )
            continue
        page_start, page_end, span_start, span_end = global_to_page_span(char_to_page, page_start_global, start, end)
        chunk_id = f"CH{len(chunks) + 1:04d}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "section_title": section_path[-1],
                "section_path": section_path,
                "page_start": page_start,
                "page_end": page_end,
                "page_span": {
                    "start": {"page": span_start.page, "offset": span_start.offset},
                    "end": {"page": span_end.page, "offset": span_end.offset},
                },
                "text_span": {"start": start, "end": end},
                "text": chunk_text.strip(),
                "text_quality": {"cjk_ratio": round(cjk_ratio, 4)},
                "anchors": {"sha1_12": sha1_12(chunk_text[:400]), "head_80": chunk_text.strip()[:80]},
            }
        )

    path = Path(pdf_path)
    unit_exponent_corrections = [
        correction
        for page in pages
        for correction in ((page.get("meta", {}) or {}).get("unit_exponent_corrections", []) or [])
    ]
    payload = {
        "doc_id": stable_doc_id(path),
        "pdf_path": str(path),
        "source_title": path.stem,
        "chunking_method": method,
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "dropped_chunks": dropped_chunks,
        "unit_exponent_corrections": unit_exponent_corrections,
        "chunks": chunks,
    }
    if toc_report is not None:
        payload["toc_llm_report"] = toc_report
    return payload
