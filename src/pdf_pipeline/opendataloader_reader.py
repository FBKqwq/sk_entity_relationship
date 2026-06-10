"""OpenDataLoader pre-parser adapter for PDF pages."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .page_cleaner import clean_page_text, normalize_whitespace


TEXT_TYPES = {"paragraph", "text", "list", "list_item", "caption", "formula"}
HEADING_TYPES = {"heading", "title", "section_header"}
TABLE_TYPES = {"table"}
IMAGE_TYPES = {"image", "picture", "figure"}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _element_type(element: dict[str, Any]) -> str:
    for key in ("type", "category", "element_type", "role"):
        value = element.get(key)
        if value:
            return str(value).strip().lower().replace("-", "_")
    return ""


def _element_text(element: dict[str, Any]) -> str:
    for key in ("text", "content", "markdown", "html"):
        value = element.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_whitespace(value)
    table = element.get("table") or element.get("rows") or element.get("cells")
    if isinstance(table, list):
        rows: list[str] = []
        for row in table:
            if isinstance(row, list):
                rows.append(" | ".join(normalize_whitespace(str(cell)) for cell in row))
            elif isinstance(row, dict):
                values = row.get("cells") or row.get("values") or list(row.values())
                cell_texts: list[str] = []
                for cell in _as_list(values):
                    if isinstance(cell, dict):
                        cell_texts.append(_element_text(cell))
                    else:
                        cell_texts.append(normalize_whitespace(str(cell)))
                rows.append(" | ".join(cell for cell in cell_texts if cell))
        return normalize_whitespace("\n".join(row for row in rows if row.strip()))
    return ""


def _page_number(element: dict[str, Any], fallback: int = 1) -> int:
    for key in ("page_number", "page number", "page", "pageIndex", "page_index"):
        value = element.get(key)
        if isinstance(value, int):
            return value + 1 if key in {"pageIndex", "page_index"} and value == 0 else value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return fallback


def _bbox(element: dict[str, Any]) -> list[float]:
    value = element.get("bbox") or element.get("bounding_box") or element.get("bounding box") or element.get("box")
    if isinstance(value, dict):
        keys = ("x0", "top", "x1", "bottom")
        if all(key in value for key in keys):
            return [float(value[key]) for key in keys]
        alt_keys = ("left", "top", "right", "bottom")
        if all(key in value for key in alt_keys):
            return [float(value[key]) for key in alt_keys]
    if isinstance(value, list) and len(value) == 4:
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            return []
    return []


def _is_usable_bbox(bbox: list[float], *, min_width: float = 1.0, min_height: float = 1.0) -> bool:
    if len(bbox) != 4:
        return False
    x0, top, x1, bottom = bbox
    if x1 <= x0 or bottom <= top:
        return False
    if (x1 - x0) < min_width or (bottom - top) < min_height:
        return False
    if x0 < 0 or top < 0:
        return False
    if x1 > 5000 or bottom > 5000:
        return False
    return True


def _heading_level(element: dict[str, Any]) -> int | None:
    for key in ("heading_level", "heading level", "level", "depth"):
        value = element.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _walk_elements(node: Any, fallback_page: int = 1) -> list[dict[str, Any]]:
    if isinstance(node, list):
        output: list[dict[str, Any]] = []
        for item in node:
            output.extend(_walk_elements(item, fallback_page=fallback_page))
        return output
    if not isinstance(node, dict):
        return []

    page = _page_number(node, fallback=fallback_page)
    candidates: list[Any] = []
    for key in ("elements", "children", "kids", "blocks", "items", "list items"):
        if key in node:
            candidates.append(node[key])
    if "pages" in node:
        pages = node.get("pages")
        if isinstance(pages, list):
            output: list[dict[str, Any]] = []
            for index, page_node in enumerate(pages, start=1):
                output.extend(_walk_elements(page_node, fallback_page=_page_number(page_node, index) if isinstance(page_node, dict) else index))
            return output

    output = [node] if _element_type(node) or _element_text(node) else []
    for candidate in candidates:
        output.extend(_walk_elements(candidate, fallback_page=page))
    return output


def _find_json_output(output_dir: Path) -> Path:
    json_paths = sorted(output_dir.rglob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not json_paths:
        raise FileNotFoundError(f"OpenDataLoader did not produce JSON under {output_dir}")
    scored: list[tuple[int, Path]] = []
    for path in json_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        elements = _walk_elements(payload)
        score = len(elements)
        if isinstance(payload, dict) and "metadata" in payload:
            score += 5
        scored.append((score, path))
    if not scored:
        raise FileNotFoundError(f"No readable OpenDataLoader JSON found under {output_dir}")
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _copy_file_bytes(source: Path, target: Path) -> None:
    """Copy a file without Windows CopyFile2 to avoid path encoding edge cases."""
    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src_file, target.open("wb") as dst_file:
        shutil.copyfileobj(src_file, dst_file, length=1024 * 1024)
    try:
        shutil.copystat(source, target)
    except OSError:
        pass


def _clean_path(path: str | Path) -> Path:
    """Normalize a path value that may contain accidental CLI whitespace."""
    cleaned = str(path).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return Path(cleaned)


def _copy_tree_contents(source_dir: Path, target_dir: Path) -> None:
    """Copy OpenDataLoader outputs from a temporary ASCII workspace."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.iterdir():
        target = target_dir / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True, copy_function=_copy_file_bytes)
        else:
            _copy_file_bytes(source, target)


def _run_opendataloader(pdf_path: Path, output_dir: Path, config: dict[str, Any]) -> Path:
    source_pdf = _clean_path(pdf_path).expanduser().resolve(strict=True)
    command = str(config.get("command", "opendataloader-pdf"))
    timeout = int(config.get("timeout_seconds", 300))
    formats = str(config.get("formats", "json"))
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_args = config.get("extra_args") or []

    with tempfile.TemporaryDirectory(prefix="opendataloader_cli_") as tmpdir:
        workspace = Path(tmpdir)
        cli_pdf = workspace / "input.pdf"
        cli_output_dir = workspace / "output"
        _copy_file_bytes(source_pdf, cli_pdf)

        args = [command, str(cli_pdf), "-o", str(cli_output_dir), "-f", formats]
        if isinstance(extra_args, list):
            args.extend(str(item) for item in extra_args)
        try:
            result = subprocess.run(args, check=False, capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=timeout)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "OpenDataLoader command was not found. "
                f"Configured command: {command!r}. "
                "Install OpenDataLoader or set pdf.opendataloader.command to the full executable path."
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                "OpenDataLoader failed with exit code "
                f"{result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        json_path = _find_json_output(cli_output_dir)
        relative_json_path = json_path.relative_to(cli_output_dir)
        _copy_tree_contents(cli_output_dir, output_dir)
        copied_json_path = output_dir / relative_json_path
        if copied_json_path.exists():
            return copied_json_path
    return _find_json_output(output_dir)


def _build_page_payloads(elements: list[dict[str, Any]], *, clean_text: bool) -> list[dict[str, Any]]:
    by_page: dict[int, list[dict[str, Any]]] = {}
    for element in elements:
        by_page.setdefault(_page_number(element), []).append(element)

    pages: list[dict[str, Any]] = []
    for page_number in sorted(by_page):
        lines: list[dict[str, Any]] = []
        text_parts: list[str] = []
        cursor = 0
        for element in by_page[page_number]:
            kind = _element_type(element)
            text = _element_text(element)
            if not text:
                continue
            level = _heading_level(element)
            if kind in HEADING_TYPES and level is not None:
                text = f"{'#' * max(1, min(level, 6))} {text}"
            bbox = _bbox(element)
            line = {
                "text": text,
                "bbox": bbox,
                "start_offset": cursor,
                "end_offset": cursor + len(text),
                "source_type": kind,
            }
            if level is not None:
                line["heading_level"] = level
            lines.append(line)
            text_parts.append(text)
            cursor += len(text) + 1
        raw_text = normalize_whitespace("\n".join(text_parts))
        text = clean_page_text(raw_text) if clean_text else raw_text
        pages.append(
            {
                "page_number": page_number,
                "text": text,
                "raw_text": raw_text,
                "meta": {
                    "parser": "opendataloader",
                    "layout_lines": lines,
                    "num_elements": len(by_page[page_number]),
                    "num_tables": sum(1 for item in by_page[page_number] if _element_type(item) in TABLE_TYPES),
                    "has_images": any(_element_type(item) in IMAGE_TYPES for item in by_page[page_number]),
                    "opendataloader_elements": [
                        {
                            "type": _element_type(item),
                            "text": _element_text(item)[:300],
                            "bbox": _bbox(item),
                            "heading_level": _heading_level(item),
                        }
                        for item in by_page[page_number]
                    ],
                },
            }
        )
    return pages


def _build_objects(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    counters: dict[tuple[str, int], int] = {}
    for element in elements:
        kind = _element_type(element)
        if kind not in TABLE_TYPES and kind not in IMAGE_TYPES:
            continue
        bbox = _bbox(element)
        if not _is_usable_bbox(bbox, min_width=30.0, min_height=20.0):
            continue
        page = _page_number(element)
        object_type = "table" if kind in TABLE_TYPES else "figure"
        counters[(object_type, page)] = counters.get((object_type, page), 0) + 1
        prefix = "T" if object_type == "table" else "F"
        text = _element_text(element)
        objects.append(
            {
                "object_id": f"{prefix}_p{page:03d}_odl_{counters[(object_type, page)]:03d}",
                "object_type": object_type,
                "page": page,
                "caption": text[:200] if object_type == "table" else str(element.get("caption", "")),
                "body": element.get("rows") or element.get("table") or [],
                "body_bbox": bbox if object_type == "table" else [],
                "bbox": bbox,
                "source": "opendataloader",
                "ocr_status": "not_required" if object_type == "table" else "pending",
            }
        )
    return objects


def read_pdf_pages_with_opendataloader(
    pdf_path: str | Path,
    *,
    config: dict[str, Any] | None = None,
    clean_text: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run OpenDataLoader and return project-compatible pages and objects."""
    pdf = Path(pdf_path)
    cfg = config or {}
    configured_output = cfg.get("output_dir")
    if configured_output:
        output_dir = Path(configured_output) / pdf.stem
        json_path = _run_opendataloader(pdf, output_dir, cfg)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            json_path = _run_opendataloader(pdf, output_dir, cfg)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            elements = _walk_elements(payload)
            pages = _build_page_payloads(elements, clean_text=clean_text)
            objects = _build_objects(elements)
            return pages, objects, {
                "parser": "opendataloader",
                "json_path": str(json_path),
                "total_elements": len(elements),
                "total_objects": len(objects),
                "page_count": len(pages),
            }

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    elements = _walk_elements(payload)
    pages = _build_page_payloads(elements, clean_text=clean_text)
    objects = _build_objects(elements)
    return pages, objects, {
        "parser": "opendataloader",
        "json_path": str(json_path),
        "total_elements": len(elements),
        "total_objects": len(objects),
        "page_count": len(pages),
    }
