"""表格提取与文本化工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def format_table(table: list[list[str | None]]) -> str:
    """将 pdfplumber 表格转为可放入 chunk 的文本块。"""
    if not table or len(table) < 2:
        return ""
    headers = [((header or "").strip() or f"列{index + 1}") for index, header in enumerate(table[0])]
    rows: list[str] = []
    for row in table[1:]:
        parts: list[str] = []
        for header, cell in zip(headers, row):
            value = (cell or "").strip()
            if value:
                parts.append(f"{header}: {value}")
        if parts:
            rows.append(" | ".join(parts))
    return "\n".join(rows).strip()


def extract_tables_by_page(pdf_path: str | Path) -> dict[int, list[dict[str, Any]]]:
    """提取每页表格，作为后续增强接口。"""
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("表格提取需要安装 pdfplumber。") from exc

    tables_by_page: dict[int, list[dict[str, Any]]] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            rows = []
            for index, table in enumerate(page.extract_tables() or [], start=1):
                text = format_table(table)
                if text:
                    rows.append({"table_id": f"T{page_number}_{index}", "text": text})
            tables_by_page[page_number] = rows
    return tables_by_page


def extract_table_bodies(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Extract table bodies with bounding boxes and cell content."""
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("Table extraction requires pdfplumber") from exc

    bodies: list[dict[str, Any]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.find_tables() or []
            except Exception:
                tables = []
            for index, table in enumerate(tables, start=1):
                try:
                    rows = table.extract() or []
                except Exception:
                    rows = []
                if not rows:
                    continue
                bbox = [float(value) for value in table.bbox]
                bodies.append(
                    {
                        "object_id": f"T_p{page_number:03d}_{index:03d}",
                        "object_type": "table",
                        "page": page_number,
                        "body": rows,
                        "body_bbox": bbox,
                        "bbox": bbox,
                        "source": "pdfplumber_find_tables",
                        "ocr_status": "not_required",
                    }
                )
    return bodies
