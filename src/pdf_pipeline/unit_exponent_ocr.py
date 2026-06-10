"""OCR-assisted recovery for unit exponents lost by PDF text extraction."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .page_cleaner import normalize_whitespace


CFU_UNIT_PATTERN = r"(?:c\s*f\s*u|cfu)\s*[／/]\s*m\s*l"
SUSPECT_CFU_PATTERN = re.compile(r"(?P<op>[>≥=])\s*10\s*(?P<unit>" + CFU_UNIT_PATTERN + r")", re.IGNORECASE)
OCR_CFU_EXPONENT_PATTERN = re.compile(
    r"(?:[>≥=]\s*)?10\s*(?:\^)?(?P<exp>[3-9])\s*(?:c\s*f\s*u|[ce]\s*f\s*u|c\s*f\s*a)\s*[／/x]?\s*m?\s*l?",
    re.IGNORECASE,
)
SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def _run_tesseract(image_path: Path, *, psm: int, languages: str) -> str:
    command = ["tesseract", str(image_path), "stdout", "-l", languages, "--psm", str(psm)]
    result = subprocess.run(command, check=False, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if result.returncode != 0:
        return ""
    return normalize_whitespace(result.stdout)


def _ocr_bbox(
    pdf_path: Path,
    page_index: int,
    bbox: list[float],
    *,
    scale: float,
    padding: float,
    psm_values: list[int],
    languages: str,
) -> list[str]:
    try:
        import pypdfium2 as pdfium
    except Exception:
        return []
    try:
        doc = pdfium.PdfDocument(str(pdf_path))
        page = doc[page_index]
        image = page.render(scale=scale).to_pil()
    except Exception:
        return []

    width, height = image.size
    x0, top, x1, bottom = bbox
    left = max(0, int((x0 - padding) * scale))
    upper = max(0, int((top - padding) * scale))
    right = min(width, int((x1 + padding) * scale))
    lower = min(height, int((bottom + padding) * scale))
    if right <= left or lower <= upper:
        return []

    texts: list[str] = []
    with tempfile.TemporaryDirectory(prefix="unit_exp_ocr_") as tmpdir:
        crop_path = Path(tmpdir) / "crop.png"
        image.crop((left, upper, right, lower)).save(crop_path)
        for psm in psm_values:
            ocr_text = _run_tesseract(crop_path, psm=psm, languages=languages)
            if ocr_text:
                texts.append(ocr_text)
    return texts


def _extract_ocr_exponents(ocr_texts: list[str]) -> list[str]:
    exponents: list[str] = []
    for text in ocr_texts:
        normalized = text.translate(SUPERSCRIPT_DIGITS)
        normalized = normalized.replace("／", "/")
        for match in OCR_CFU_EXPONENT_PATTERN.finditer(normalized):
            exponents.append(match.group("exp"))
    return exponents


def _extract_gender_cfu_exponents(
    line_text: str,
    ocr_texts: list[str],
) -> tuple[list[str], list[str]]:
    compact_line = re.sub(r"\s+", "", line_text)
    if "\u5973\u6027" not in compact_line or "\u7537\u6027" not in compact_line:
        return [], []

    female_exp = ""
    male_exp = ""
    female_source = ""
    male_source = ""
    for text in ocr_texts:
        normalized = re.sub(r"\s+", "", text.translate(SUPERSCRIPT_DIGITS).replace("／", "/"))
        female_match = re.search(r"\u5973\u6027[^\d>，,、。]{0,8}>?10(?:\^)?([3-9])", normalized)
        male_match = re.search(r"\u7537\u6027[^\d>，,、。]{0,8}>?10(?:\^)?([3-9])", normalized)
        if female_match and not female_exp:
            female_exp = female_match.group(1)
            female_source = "ocr_crop"
        if male_match and not male_exp:
            male_exp = male_match.group(1)
            male_source = "ocr_crop"
    if female_exp and male_exp:
        return [female_exp, male_exp], [female_source, male_source]
    return [], []


def _is_gender_cfu_line(line_text: str) -> bool:
    compact = re.sub(r"\s+", "", line_text)
    return "\u5973\u6027" in compact and "\u7537\u6027" in compact


def _apply_cfu_exponents(line_text: str, exponents: list[str]) -> str:
    index = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal index
        if index >= len(exponents):
            return match.group(0)
        exp = exponents[index]
        index += 1
        unit = re.sub(r"\s+", "", match.group("unit"))
        return f"{match.group('op')}10^{exp}{unit}"

    return SUSPECT_CFU_PATTERN.sub(replace, line_text)


def recover_unit_exponents_in_page(
    pdf_path: str | Path,
    page_index: int,
    layout_lines: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recover likely missing exponents in unit expressions using local OCR crops."""
    cfg = config or {}
    if not bool(cfg.get("enabled", False)):
        return layout_lines, []

    scale = float(cfg.get("render_scale", 4.0))
    padding = float(cfg.get("crop_padding", 6.0))
    languages = str(cfg.get("languages", "chi_sim+eng"))
    psm_values = [int(value) for value in cfg.get("psm_values", [6, 11])]
    path = Path(pdf_path)

    corrected_lines: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    for line in layout_lines:
        text = str(line.get("text", ""))
        matches = list(SUSPECT_CFU_PATTERN.finditer(text))
        if not matches:
            corrected_lines.append(line)
            continue

        bbox = line.get("bbox")
        ocr_texts: list[str] = []
        if isinstance(bbox, list) and len(bbox) == 4:
            ocr_texts = _ocr_bbox(
                path,
                page_index,
                [float(value) for value in bbox],
                scale=scale,
                padding=padding,
                psm_values=psm_values,
                languages=languages,
            )

        exponent_sources: list[str] = []
        exponents, exponent_sources = _extract_gender_cfu_exponents(
            text,
            ocr_texts,
        )
        if not exponents and not _is_gender_cfu_line(text):
            exponents = _extract_ocr_exponents(ocr_texts)
            exponent_sources = ["ocr_crop"] * len(exponents)
        source = "ocr_crop"
        confidence = 0.86

        if len(exponents) < len(matches):
            corrected_lines.append(line)
            corrections.append(
                {
                    "type": "unit_exponent_recovery",
                    "status": "unresolved",
                    "page": page_index + 1,
                    "original": text,
                    "ocr_texts": ocr_texts,
                    "reason": "OCR did not provide enough exponent evidence.",
                }
            )
            continue

        corrected = _apply_cfu_exponents(text, exponents)
        next_line = dict(line)
        next_line["text"] = corrected
        corrected_lines.append(next_line)
        corrections.append(
            {
                "type": "unit_exponent_recovery",
                "status": "corrected",
                "page": page_index + 1,
                "original": text,
                "corrected": corrected,
                "exponents": exponents[: len(matches)],
                "source": source,
                "exponent_sources": exponent_sources[: len(matches)],
                "confidence": confidence,
                "bbox": bbox,
                "ocr_texts": ocr_texts,
            }
        )

    return corrected_lines, corrections
