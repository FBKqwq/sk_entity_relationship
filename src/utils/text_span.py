"""字符 span 与页码映射工具。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpanPoint:
    """页内字符位置。"""

    page: int
    offset: int


def build_fulltext_and_maps(page_text: dict[int, str]) -> tuple[str, list[int], dict[int, int]]:
    """拼接全文并建立全局字符到页码的映射。"""
    full_parts: list[str] = []
    char_to_page: list[int] = []
    page_start_global: dict[int, int] = {}
    current = 0

    for page_number in sorted(page_text):
        text = page_text[page_number] or ""
        page_start_global[page_number] = current
        full_parts.append(text)
        char_to_page.extend([page_number] * len(text))
        current += len(text)

        separator = "\n\n"
        full_parts.append(separator)
        char_to_page.extend([page_number] * len(separator))
        current += len(separator)

    full_text = "".join(full_parts).rstrip()
    return full_text, char_to_page[: len(full_text)], page_start_global


def global_to_page_span(
    char_to_page: list[int],
    page_start_global: dict[int, int],
    start: int,
    end: int,
) -> tuple[int, int, SpanPoint, SpanPoint]:
    """将全文 span 转为起止页和页内 offset。"""
    if not char_to_page:
        return 1, 1, SpanPoint(1, 0), SpanPoint(1, 0)

    safe_start = max(0, min(start, len(char_to_page) - 1))
    safe_end = max(safe_start + 1, min(end, len(char_to_page)))
    page_start = char_to_page[safe_start]
    page_end = char_to_page[safe_end - 1]
    point_start = SpanPoint(page_start, safe_start - page_start_global[page_start])
    point_end = SpanPoint(page_end, safe_end - page_start_global[page_end])
    return page_start, page_end, point_start, point_end
