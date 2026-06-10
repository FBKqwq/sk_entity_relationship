"""Build complete figure/table objects from layout and PDF detections."""

from __future__ import annotations

import re
from typing import Any

from .layout_regions import bbox_vertical_distance, union_bboxes

CAPTION_RE = re.compile(r"^\s*(表\s*\d+|图\s*\d+|Table\s*\d+|Figure\s*\d+)", re.IGNORECASE)
NOTE_RE = re.compile(r"^\s*(注[:：]|注释[:：]|Note:|Notes:)", re.IGNORECASE)


def extract_caption_candidates(page_layout: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_copy_candidate(line, "caption") for line in page_layout if CAPTION_RE.search(str(line.get("text", "")))]


def extract_note_candidates(page_layout: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_copy_candidate(line, "note") for line in page_layout if NOTE_RE.search(str(line.get("text", "")))]


def _copy_candidate(line: dict[str, Any], kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "text": str(line.get("text", "")),
        "bbox": list(line.get("bbox", [])),
    }


def _object_number(text: str, kind: str) -> str | None:
    prefix = r"(?:表|Table)" if kind == "table" else r"(?:图|Figure)"
    match = re.search(prefix + r"\s*(\d+)", text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _nearest_above(body_bbox: list[float], candidates: list[dict[str, Any]], max_distance: float) -> dict[str, Any] | None:
    above = []
    for candidate in candidates:
        bbox = candidate.get("bbox") or []
        if len(bbox) != 4 or bbox[3] > body_bbox[1]:
            continue
        distance = body_bbox[1] - bbox[3]
        if distance <= max_distance:
            above.append((distance, candidate))
    return min(above, key=lambda item: item[0])[1] if above else None


def _nearest_below(body_bbox: list[float], candidates: list[dict[str, Any]], max_distance: float) -> dict[str, Any] | None:
    below = []
    for candidate in candidates:
        bbox = candidate.get("bbox") or []
        if len(bbox) != 4 or bbox[1] < body_bbox[3]:
            continue
        distance = bbox[1] - body_bbox[3]
        if distance <= max_distance:
            below.append((distance, candidate))
    return min(below, key=lambda item: item[0])[1] if below else None


def build_table_objects(
    table_bodies: list[dict[str, Any]],
    page_layouts: dict[int, list[dict[str, Any]]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    caption_distance = float(config.get("caption_search_distance", 90))
    note_distance = float(config.get("note_search_distance", 120))
    output: list[dict[str, Any]] = []

    for body in table_bodies:
        page = int(body["page"])
        layout = page_layouts.get(page, [])
        captions = extract_caption_candidates(layout)
        notes = extract_note_candidates(layout)
        body_bbox = list(body.get("body_bbox", body.get("bbox", [])))
        table = dict(body)
        caption = _nearest_above(body_bbox, captions, caption_distance)
        note = _nearest_below(body_bbox, notes, note_distance)
        bboxes = [body_bbox]
        if caption:
            table["caption"] = caption["text"]
            table["caption_bbox"] = caption["bbox"]
            table["table_number"] = _object_number(caption["text"], "table")
            bboxes.append(caption["bbox"])
        if note:
            table["note"] = note["text"]
            table["note_bbox"] = note["bbox"]
            bboxes.append(note["bbox"])
        table["bbox"] = union_bboxes(bboxes)
        output.append(table)
    return output


def build_figure_objects_from_images(
    pages: list[dict[str, Any]],
    page_layouts: dict[int, list[dict[str, Any]]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    caption_distance = float(config.get("caption_search_distance", 90))
    min_width = float(config.get("min_figure_width", 80))
    min_height = float(config.get("min_figure_height", 80))
    figures: list[dict[str, Any]] = []
    for page in pages:
        page_number = int(page["page_number"])
        images = ((page.get("meta", {}) or {}).get("images", []) or [])
        captions = extract_caption_candidates(page_layouts.get(page_number, []))
        for index, image in enumerate(images, start=1):
            bbox = list(image.get("bbox", []))
            if len(bbox) != 4:
                continue
            if (bbox[2] - bbox[0]) < min_width or (bbox[3] - bbox[1]) < min_height:
                continue
            object_id = f"F_p{page_number:03d}_{index:03d}"
            figure = {
                "object_id": object_id,
                "object_type": "figure",
                "page": page_number,
                "bbox": bbox,
                "source": image.get("source", "pdf_image"),
                "ocr_status": "pending",
            }
            nearby = [
                (bbox_vertical_distance(bbox, candidate["bbox"]), candidate)
                for candidate in captions
                if len(candidate.get("bbox", [])) == 4 and _object_number(candidate["text"], "figure")
            ]
            nearby = [item for item in nearby if item[0] <= caption_distance]
            if nearby:
                caption = min(nearby, key=lambda item: item[0])[1]
                figure["caption"] = caption["text"]
                figure["caption_bbox"] = caption["bbox"]
                figure["figure_number"] = _object_number(caption["text"], "figure")
                figure["bbox"] = union_bboxes([bbox, caption["bbox"]])
            figures.append(figure)
    return figures
