"""图片与图表识别接口，支持禁用时安全回退。"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from src.utils.llm_client import image_path_to_data_url, vision_completion_text


DEFAULT_OCR_PROMPT = (
    "请识别图片中的医学图表、表格或示意图内容。"
    "只根据图片可见内容回答，不补充图片外知识。"
    "如为表格，请按行列关系转写；如为图，请描述标题、坐标轴、图例、关键数值和结论性文字。"
    "输出简体中文。"
)


def caption_image_stub(*, page_number: int, caption: str | None = None) -> dict[str, object]:
    """返回图片待识别占位结果，不调用外部模型。"""
    return {
        "page_number": page_number,
        "caption": caption or "图",
        "text": "（图片内容待识别）",
        "status": "stub",
    }


def recognize_image_url(
    image_url: str,
    *,
    page_number: int,
    caption: str | None = None,
    prompt: str = DEFAULT_OCR_PROMPT,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    """调用 OCR/VL 模型识别远程 URL 或 data URL 图片。"""
    result = vision_completion_text(image_url, prompt, config_path=config_path, stream=True)
    if result["status"] != "ok" or not result.get("text"):
        fallback = caption_image_stub(page_number=page_number, caption=caption)
        fallback["reason"] = result.get("reason", "OCR 模型未返回内容。")
        fallback["model"] = result.get("model")
        return fallback
    return {
        "page_number": page_number,
        "caption": caption or "图",
        "text": str(result["text"]),
        "reasoning_content": str(result.get("reasoning_content", "")),
        "status": "recognized",
        "model": result.get("model"),
    }


def recognize_image_file(
    image_path: str | Path,
    *,
    page_number: int,
    caption: str | None = None,
    prompt: str = DEFAULT_OCR_PROMPT,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    """调用 OCR/VL 模型识别本地图片文件。"""
    data_url = image_path_to_data_url(image_path)
    return recognize_image_url(
        data_url,
        page_number=page_number,
        caption=caption,
        prompt=prompt,
        config_path=config_path,
    )


def _pil_image_to_data_url(image: Any) -> str:
    """将 PIL Image 转为 PNG data URL。"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def recognize_pdf_images_by_page(
    pdf_path: str | Path,
    *,
    config_path: str | Path | None = None,
    max_images_per_page: int = 1,
    prompt: str = DEFAULT_OCR_PROMPT,
) -> dict[int, list[dict[str, object]]]:
    """识别 PDF 每页图片，返回可追加到页文本的 OCR 结果。"""
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("PDF 图片识别需要安装 pdfplumber。") from exc

    results: dict[int, list[dict[str, object]]] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_results: list[dict[str, object]] = []
            images = list(getattr(page, "images", []) or [])[:max_images_per_page]
            for index, image in enumerate(images, start=1):
                caption = f"图{page_number}_{index}"
                try:
                    bbox = (image["x0"], image["top"], image["x1"], image["bottom"])
                    cropped = page.within_bbox(bbox).to_image(resolution=150).original
                    data_url = _pil_image_to_data_url(cropped)
                    result = recognize_image_url(
                        data_url,
                        page_number=page_number,
                        caption=caption,
                        prompt=prompt,
                        config_path=config_path,
                    )
                except Exception as exc:  # noqa: BLE001 - OCR 是增强步骤，失败时保留可追踪结果
                    result = caption_image_stub(page_number=page_number, caption=caption)
                    result["reason"] = f"图片裁剪或识别失败: {exc}"
                page_results.append(result)
            if page_results:
                results[page_number] = page_results
    return results


def append_ocr_results_to_page_text(
    page_text: dict[int, str],
    ocr_results: dict[int, list[dict[str, object]]],
) -> dict[int, str]:
    """将 OCR 结果以 FIGURE_CONTENT 块追加到对应页文本。"""
    output = dict(page_text)
    for page_number, results in ocr_results.items():
        blocks: list[str] = []
        for result in results:
            caption = str(result.get("caption") or "图")
            text = str(result.get("text") or "").strip()
            if not text:
                continue
            blocks.append(f"{caption}\n[FIGURE_CONTENT]\n{text}\n[/FIGURE_CONTENT]")
        if blocks:
            original = output.get(page_number, "")
            output[page_number] = (original.rstrip() + "\n\n" + "\n\n".join(blocks)).strip()
    return output
