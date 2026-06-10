"""在图/表与 chunk 关联之后，对关联对象按需 OCR 并注入 [FIGURE_CONTENT]。"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from src.pdf_pipeline.image_caption_stub import DEFAULT_OCR_PROMPT, recognize_image_url


def _pil_image_to_data_url(image: Any) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _index_objects(objects: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(obj.get("object_id", "")): obj for obj in objects if obj.get("object_id")}


def _best_chunk_per_object(payload: dict[str, Any]) -> dict[str, str]:
    """每个 object_id 选择 link_confidence 最高的 chunk 作为注入目标。"""
    best: dict[str, tuple[str, float]] = {}
    for chunk in payload.get("chunks", []):
        chunk_id = str(chunk.get("chunk_id", ""))
        for link in chunk.get("linked_figures", []) + chunk.get("linked_tables", []):
            oid = str(link.get("object_id", ""))
            if not oid:
                continue
            conf = float(link.get("link_confidence", 0.0))
            prev = best.get(oid)
            if prev is None or conf > prev[1]:
                best[oid] = (chunk_id, conf)
    return {oid: cid for oid, (cid, _) in best.items()}


def _intersect_bbox_with_page_bbox(
    bbox: tuple[float, float, float, float],
    page_bbox: tuple[float, float, float, float],
    *,
    min_edge_pt: float = 0.5,
) -> tuple[float, float, float, float] | None:
    """
    将对象 bbox 与 pdfplumber 页 bbox 求交并夹紧，避免浮点误差导致 within_bbox 报
    「not fully within parent page bounding box」。
    """
    x0, top, x1, bottom = bbox
    px0, ptop, px1, pbottom = page_bbox
    nx0 = max(float(x0), float(px0))
    ny0 = max(float(top), float(ptop))
    nx1 = min(float(x1), float(px1))
    ny1 = min(float(bottom), float(pbottom))
    if nx1 - nx0 < min_edge_pt or ny1 - ny0 < min_edge_pt:
        return None
    return (nx0, ny0, nx1, ny1)


def _crop_page_region_to_data_url(
    pdf_path: Path,
    page_number: int,
    bbox: list[float],
    *,
    resolution: int = 150,
) -> str | None:
    try:
        import pdfplumber
    except ImportError:
        return None
    if len(bbox) != 4:
        return None
    x0, top, x1, bottom = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if x1 <= x0 or bottom <= top:
        return None
    with pdfplumber.open(str(pdf_path)) as pdf:
        if page_number < 1 or page_number > len(pdf.pages):
            return None
        page = pdf.pages[page_number - 1]
        page_bbox = tuple(float(x) for x in page.bbox)
        crop = _intersect_bbox_with_page_bbox((x0, top, x1, bottom), page_bbox)
        if crop is None:
            return None
        cropped = page.within_bbox(crop).to_image(resolution=resolution).original
        return _pil_image_to_data_url(cropped)


def run_linked_object_ocr_and_inject(
    payload: dict[str, Any],
    input_pdf: Path,
    pdf_config: dict[str, Any],
    *,
    llm_config_path: str | Path,
    pages: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    对已关联的图/表按 bbox 裁剪并调用视觉模型，记录解析结果。

    若配置允许注入且传入 pages，则将 [FIGURE_CONTENT] 块追加到对象所在页的
    ``pages[*]['text']`` 末尾（不重算 chunk）。

    返回 (更新后的 payload, 图表解析记录列表)。
    """
    ocr_cfg = pdf_config.get("figure_table_ocr") or {}
    if not bool(ocr_cfg.get("enabled", False)):
        return payload, []

    objects = list(payload.get("figure_table_objects") or [])
    if not objects:
        return payload, []

    inject_pages = bool(
        ocr_cfg.get("inject_into_page_text", ocr_cfg.get("inject_into_chunk_text", True))
    )
    resolution = int(ocr_cfg.get("ocr_resolution", 150))
    prompt = str(ocr_cfg.get("ocr_prompt", DEFAULT_OCR_PROMPT))

    obj_by_id = _index_objects(objects)
    target_chunk_by_object = _best_chunk_per_object(payload)

    records: list[dict[str, Any]] = []
    ocr_text_by_object: dict[str, str] = {}

    for oid, obj in obj_by_id.items():
        if oid not in target_chunk_by_object:
            continue
        page = int(obj.get("page", 0) or 0)
        bbox = list(obj.get("bbox") or [])
        data_url = _crop_page_region_to_data_url(input_pdf, page, bbox, resolution=resolution)
        if not data_url:
            records.append(
                {
                    "object_id": oid,
                    "object_type": obj.get("object_type"),
                    "page": page,
                    "chunk_id": target_chunk_by_object[oid],
                    "caption": str(obj.get("caption", "")),
                    "ocr_text": "",
                    "status": "crop_failed",
                    "model": None,
                }
            )
            continue

        caption = str(obj.get("caption") or ("表" if obj.get("object_type") == "table" else "图"))
        result = recognize_image_url(
            data_url,
            page_number=page,
            caption=caption,
            prompt=prompt,
            config_path=llm_config_path,
        )
        text = str(result.get("text") or "").strip()
        status = "ok" if result.get("status") == "ok" and text else str(result.get("status", "stub"))
        records.append(
            {
                "object_id": oid,
                "object_type": obj.get("object_type"),
                "page": page,
                "chunk_id": target_chunk_by_object[oid],
                "caption": caption,
                "ocr_text": text,
                "ocr_status": result.get("status"),
                "model": result.get("model"),
                "status": status,
            }
        )
        if text:
            ocr_text_by_object[oid] = text
        obj["ocr_status"] = "recognized" if text else str(result.get("status", "pending"))

    if inject_pages and pages is not None and ocr_text_by_object:
        page_by_num = {int(p["page_number"]): p for p in pages}
        for oid, ocr_text in ocr_text_by_object.items():
            obj = obj_by_id.get(oid) or {}
            page_num = int(obj.get("page", 0) or 0)
            page_row = page_by_num.get(page_num)
            if page_row is None:
                continue
            caption = str(obj.get("caption") or ("表" if obj.get("object_type") == "table" else "图"))
            block = f"{caption}\n[FIGURE_CONTENT]\n{ocr_text}\n[/FIGURE_CONTENT]"
            base = str(page_row.get("text", "")).rstrip()
            page_row["text"] = f"{base}\n\n{block}".strip() if base else block.strip()
            page_row.setdefault("meta", {})["figure_content_appended"] = True

    payload["figure_table_objects"] = objects
    return payload, records
