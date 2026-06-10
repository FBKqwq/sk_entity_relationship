"""PDF 逐页读取与双栏文本重排。"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .page_cleaner import clean_page_text, normalize_whitespace
from .unit_exponent_ocr import recover_unit_exponents_in_page

FLOATING_TOKEN_PATTERN = re.compile(r"^[\dA-Za-z%~～≤≥<>.=+\-()/·：:，,；;［\]\[\s]+$")
PAGE_MARK_PATTERN = re.compile(r"^\s*[·•・]?\s*\d{2,4}\s*[·•・]?\s*$")
CNKI_FOOTER_PATTERN = re.compile(r"China Academic Journal Electronic Publishing House|cnki\.net", re.IGNORECASE)
HIGH_VALUE_OCR_PATTERN = re.compile(r"(?:(?<![\d.])[123]\s*急性胰腺炎|推荐\s*\d+\s*[:：])")
TEXT_TABLE_CAPTION_PATTERN = re.compile(
    r"^(?:表\s*)?\d+\s*(?:改良|急性胰腺炎.*(?:评分|系统|特点)|.*评分(?:标准|系统)|.*临床特点)"
    r"|^急性胰腺炎(?:分级诊断系统|局部并发症临床特点)$"
)
TEXT_TABLE_BODY_START_PATTERN = re.compile(r"^(?:特征\s*评分|分级系统|器官系统|评分|临床特点|感染性胰腺|[（(]4[）)]危重型)")
NARRATIVE_SECTION_PATTERN = re.compile(r"^(?:\d+(?:\.\d+)+|\d+\s+)[\u4e00-\u9fff]")
KNOWN_MAIN_HEADINGS = (
    "1 急性胰腺炎的诊断",
    "2 急性胰腺炎的治疗",
    "3 急性胰腺炎病人的随访",
)
RECOMMENDATION_INSERT_BEFORE = {
    "1": r"(?m)^\s*1\.4\s*急性胰腺炎的影像学检查",
}


def _cluster_rows(words: list[dict[str, Any]], y_tol: float = 3.0) -> list[list[dict[str, Any]]]:
    """按纵坐标聚合 pdfplumber word，来自旧 notebook 的行聚类思路。"""
    if not words:
        return []
    sorted_words = sorted(words, key=lambda word: (word["top"], word["x0"]))
    rows: list[list[dict[str, Any]]] = []
    current = [sorted_words[0]]
    current_y = float(sorted_words[0]["top"])

    for word in sorted_words[1:]:
        top = float(word["top"])
        if abs(top - current_y) <= y_tol:
            current.append(word)
            current_y = (current_y * (len(current) - 1) + top) / len(current)
        else:
            rows.append(sorted(current, key=lambda item: item["x0"]))
            current = [word]
            current_y = top
    rows.append(sorted(current, key=lambda item: item["x0"]))
    return rows


def _row_text(row: list[dict[str, Any]]) -> str:
    return "".join(str(word.get("text", "")) for word in sorted(row, key=lambda item: (item["x0"], item["top"]))).strip()


def _row_bbox(row: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    return (
        min(float(word["x0"]) for word in row),
        min(float(word["top"]) for word in row),
        max(float(word["x1"]) for word in row),
        max(float(word["bottom"]) for word in row),
    )


def _horizontal_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left = max(a[0], b[0])
    right = min(a[2], b[2])
    return max(0.0, right - left)


def _looks_like_floating_token_row(row: list[dict[str, Any]]) -> bool:
    text = _row_text(row)
    if not text or len(text) > 40:
        return False
    if not FLOATING_TOKEN_PATTERN.match(text):
        return False
    return any(ch.isdigit() or ch.isalpha() for ch in text)


def _merge_floating_token_rows(rows: list[list[dict[str, Any]]], y_tol: float = 8.5) -> list[list[dict[str, Any]]]:
    """Merge baseline-shifted numbers/Latin tokens back into the nearest text row."""
    if not rows:
        return []
    merged: list[list[dict[str, Any]]] = []
    skip: set[int] = set()
    for index, row in enumerate(rows):
        if index in skip:
            continue
        if not _looks_like_floating_token_row(row):
            merged.append(row)
            continue
        bbox = _row_bbox(row)
        previous_row = merged[-1] if merged else (rows[index - 1] if index > 0 else [])
        previous_text = _row_text(previous_row) if previous_row else ""
        if (
            previous_row
            and merged
            and re.match(r"^\d{1,2}(?:\.\d{1,2})$", _row_text(row))
            and _row_bbox(previous_row)[0] > bbox[0]
            and abs(bbox[1] - _row_bbox(previous_row)[1]) <= y_tol
        ):
            merged[-1] = sorted([*merged[-1], *row], key=lambda item: (item["x0"], item["top"]))
            skip.add(index)
            continue
        if (
            previous_row
            and merged
            and re.match(r"^\d$", _row_text(row))
            and "合项" in previous_text
            and abs(bbox[1] - _row_bbox(previous_row)[1]) <= y_tol
        ):
            merged[-1] = sorted([*merged[-1], *row], key=lambda item: (item["x0"], item["top"]))
            skip.add(index)
            continue
        if (
            previous_row
            and merged
            and not previous_text.endswith(("。", "）", ")", "：", ":"))
            and (
                _horizontal_overlap(bbox, _row_bbox(previous_row)) > 0
                or 0 <= _row_bbox(previous_row)[0] - bbox[2] <= 5
            )
            and abs(bbox[1] - _row_bbox(previous_row)[1]) <= y_tol
        ):
            merged[-1] = sorted([*merged[-1], *row], key=lambda item: (item["x0"], item["top"]))
            skip.add(index)
            continue
        best_index: int | None = None
        best_score = -1.0
        for candidate_index in (index - 1, index + 1):
            if candidate_index < 0 or candidate_index >= len(rows) or candidate_index in skip:
                continue
            candidate = rows[candidate_index]
            if _looks_like_floating_token_row(candidate):
                continue
            candidate_bbox = _row_bbox(candidate)
            y_gap = abs(bbox[1] - candidate_bbox[1])
            if y_gap > y_tol:
                continue
            overlap = _horizontal_overlap(bbox, candidate_bbox)
            inside_bonus = 25.0 if bbox[0] >= candidate_bbox[0] - 4 and bbox[2] <= candidate_bbox[2] + 4 else 0.0
            score = overlap + inside_bonus - y_gap
            if score > best_score:
                best_score = score
                best_index = candidate_index
        if best_index is None:
            merged.append(row)
            continue
        if best_index == index - 1 and merged:
            merged[-1] = sorted([*merged[-1], *row], key=lambda item: (item["x0"], item["top"]))
        else:
            rows[best_index] = sorted([*rows[best_index], *row], key=lambda item: (item["x0"], item["top"]))
        skip.add(index)

    return sorted(merged, key=lambda row: (_row_bbox(row)[1], _row_bbox(row)[0]))


def _rows_to_text(rows: list[list[dict[str, Any]]], x_gap: float = 10.0) -> str:
    """将 word 行还原为文本。"""
    lines: list[str] = []
    for row in rows:
        if not row:
            continue
        parts = [str(row[0].get("text", ""))]
        for previous, word in zip(row, row[1:]):
            gap = float(word["x0"]) - float(previous["x1"])
            if gap >= x_gap:
                parts.append(" ")
            parts.append(str(word.get("text", "")))
        line = "".join(parts).strip()
        if line:
            lines.append(line)
    return normalize_whitespace("\n".join(lines))


def _row_to_line(row: list[dict[str, Any]], x_gap: float = 10.0) -> dict[str, Any] | None:
    if not row:
        return None
    parts = [str(row[0].get("text", ""))]
    for previous, word in zip(row, row[1:]):
        gap = float(word["x0"]) - float(previous["x1"])
        if gap >= x_gap:
            parts.append(" ")
        parts.append(str(word.get("text", "")))
    text = "".join(parts).strip()
    if not text:
        return None
    return {
        "text": text,
        "bbox": [
            min(float(word["x0"]) for word in row),
            min(float(word["top"]) for word in row),
            max(float(word["x1"]) for word in row),
            max(float(word["bottom"]) for word in row),
        ],
    }


def _rows_to_layout_lines(rows: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    cursor = 0
    for row in rows:
        line = _row_to_line(row)
        if line is None:
            continue
        text = str(line["text"])
        line["start_offset"] = cursor
        line["end_offset"] = cursor + len(text)
        lines.append(line)
        cursor += len(text) + 1
    return lines


def _layout_lines_to_text(lines: list[dict[str, Any]]) -> str:
    return normalize_whitespace("\n".join(str(line.get("text", "")) for line in lines if str(line.get("text", "")).strip()))


def _refresh_layout_line_offsets(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refreshed: list[dict[str, Any]] = []
    cursor = 0
    for line in lines:
        next_line = dict(line)
        text = str(next_line.get("text", ""))
        next_line["start_offset"] = cursor
        next_line["end_offset"] = cursor + len(text)
        refreshed.append(next_line)
        cursor += len(text) + 1
    return refreshed


def _filter_header_footer(
    words: list[dict[str, Any]],
    page_height: float,
    header_pct: float = 0.06,
    footer_pct: float = 0.06,
) -> list[dict[str, Any]]:
    """根据页面高度粗略过滤页眉页脚区域。"""
    if not words or page_height <= 0:
        return words
    header_y = page_height * header_pct
    footer_y = page_height * (1.0 - footer_pct)
    return [word for word in words if word["top"] > header_y and word["top"] < footer_y]


def _bbox_overlap_ratio(
    a: list[float] | tuple[float, float, float, float],
    b: list[float] | tuple[float, float, float, float],
) -> float:
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    area = max(1.0, (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1])))
    return intersection / area


def _find_table_bboxes(page: Any) -> list[list[float]]:
    try:
        tables = page.find_tables() or []
    except Exception:
        return []
    bboxes: list[list[float]] = []
    for table in tables:
        bbox = getattr(table, "bbox", None)
        if bbox and len(bbox) == 4:
            bboxes.append([float(value) for value in bbox])
    return bboxes


def _is_text_table_caption(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if TEXT_TABLE_CAPTION_PATTERN.search(compact):
        return True
    return False


def _is_text_table_body_start(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if TEXT_TABLE_BODY_START_PATTERN.search(compact):
        return True
    if re.fullmatch(r"\d+分\d+分\d+分", compact):
        return True
    return False


def _is_text_table_row(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False
    if compact.startswith(("注：", "注:")):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?", compact):
        return True
    if _is_text_table_caption(compact) or _is_text_table_body_start(compact):
        return True
    digit_count = len(re.findall(r"\d", compact))
    has_table_symbol = bool(re.search(r"[≤≥<>～~%]|mmHg|FiO2|PaO2|RAC|DBC|SAP|MAP|MSAP|WON|ANC|PP", compact))
    if digit_count >= 2 and has_table_symbol:
        return True
    if len(compact) <= 28 and bool(re.search(r"(胰腺炎性反应|正常胰腺|胰腺坏死|坏死范围|并发症|器官功能|呼吸系统|肾脏|心血管系统)", compact)):
        return True
    if bool(re.search(r"(感染性胰腺|危重型急性胰腺炎|持续性器官功能障碍伴感染性|SOFA)", compact)):
        return True
    if "发生于病程" in compact or "包膜" in compact:
        return True
    return False


def _extract_text_table_regions(rows: list[list[dict[str, Any]]], page_number: int, region_label: str = "txt") -> list[dict[str, Any]]:
    """Detect borderless text tables from ordered layout rows."""
    regions: list[dict[str, Any]] = []
    index = 0
    object_index = 1
    while index < len(rows):
        text = _row_text(rows[index])
        starts_caption = _is_text_table_caption(text)
        starts_body = _is_text_table_body_start(text) and any(
            _is_text_table_row(_row_text(row)) for row in rows[index + 1 : min(index + 5, len(rows))]
        )
        if not starts_caption and not starts_body:
            index += 1
            continue

        start = index
        end = index + 1
        saw_note = False
        while end < len(rows):
            current = _row_text(rows[end])
            current_bbox = _row_bbox(rows[end])
            prev_bbox = _row_bbox(rows[end - 1])
            vertical_gap = current_bbox[1] - prev_bbox[3]
            if end > start and vertical_gap > 34:
                break
            if end > start and NARRATIVE_SECTION_PATTERN.match(current) and not _is_text_table_row(current):
                break
            if end > start and _is_text_table_caption(current):
                break
            if end > start and current.startswith("推荐"):
                break
            if end > start and not _is_text_table_row(current):
                if starts_caption and not (saw_note and vertical_gap > 18):
                    end += 1
                    continue
                if saw_note and vertical_gap <= 16:
                    end += 1
                    continue
                break
            if current.startswith(("注：", "注:")):
                saw_note = True
            end += 1

        if end - start < 3:
            index += 1
            continue
        table_rows = rows[start:end]
        body_bbox = list(_row_bbox([word for row in table_rows for word in row]))
        caption = _row_text(rows[start]) if starts_caption else ""
        regions.append(
            {
                "object_id": f"T_p{page_number:03d}_{region_label}_{object_index:03d}",
                "object_type": "table",
                "page": page_number,
                "caption": caption,
                "body": [[_row_text(row)] for row in table_rows],
                "body_bbox": body_bbox,
                "bbox": body_bbox,
                "source": "layout_text_table_heuristic",
                "ocr_status": "not_required",
            }
        )
        object_index += 1
        index = end
    return regions


def _filter_table_rows(rows: list[list[dict[str, Any]]], table_bboxes: list[list[float]]) -> list[list[dict[str, Any]]]:
    if not table_bboxes:
        return rows
    kept: list[list[dict[str, Any]]] = []
    for row in rows:
        bbox = _row_bbox(row)
        if any(_bbox_overlap_ratio(bbox, table_bbox) >= 0.35 for table_bbox in table_bboxes):
            continue
        kept.append(row)
    return kept


def _detect_two_column_kmeans(
    words: list[dict[str, Any]],
    page_width: float,
    min_side_ratio: float = 0.20,
    min_separation_ratio: float = 0.28,
) -> tuple[bool, float | None]:
    """用两中心聚类判断双栏，并返回分栏 x 坐标。"""
    if not words or page_width <= 0:
        return False, None
    centers = [(float(word["x0"]) + float(word["x1"])) / 2.0 for word in words]
    if len(centers) < 80:
        return False, None

    sorted_centers = sorted(centers)
    left_center = sorted_centers[int(len(centers) * 0.30)]
    right_center = sorted_centers[int(len(centers) * 0.70)]
    if abs(right_center - left_center) < page_width * 0.20:
        return False, None

    for _ in range(15):
        left: list[float] = []
        right: list[float] = []
        for x_value in centers:
            if abs(x_value - left_center) <= abs(x_value - right_center):
                left.append(x_value)
            else:
                right.append(x_value)
        if not left or not right:
            return False, None
        next_left = sum(left) / len(left)
        next_right = sum(right) / len(right)
        if abs(next_left - left_center) < 0.5 and abs(next_right - right_center) < 0.5:
            break
        left_center, right_center = next_left, next_right

    left_count = sum(1 for value in centers if abs(value - left_center) <= abs(value - right_center))
    right_count = len(centers) - left_count
    if min(left_count, right_count) / len(centers) < min_side_ratio:
        return False, None
    if abs(right_center - left_center) / page_width < min_separation_ratio:
        return False, None
    return True, (left_center + right_center) / 2.0


def _run_tesseract_image(image_path: Path) -> str:
    command = ["tesseract", str(image_path), "stdout", "-l", "chi_sim+eng", "--psm", "6"]
    result = subprocess.run(command, check=False, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if result.returncode != 0:
        return ""
    return result.stdout


def _extract_high_value_ocr_blocks(pdf_path: Path, page_index: int) -> list[str]:
    """Use local OCR only for headings/recommendation lines missing from embedded text."""
    try:
        import pypdfium2 as pdfium
    except Exception:
        return []
    try:
        doc = pdfium.PdfDocument(str(pdf_path))
        image = doc[page_index].render(scale=2).to_pil()
    except Exception:
        return []

    width, height = image.size
    crops = [
        image.crop((0, 0, int(width * 0.53), height)),
        image.crop((int(width * 0.47), 0, width, height)),
    ]
    if page_index == 0:
        crops.append(image)
    lines: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for index, crop in enumerate(crops, start=1):
            image_path = Path(tmpdir) / f"page_{page_index + 1}_{index}.png"
            crop.save(image_path)
            for raw_line in _run_tesseract_image(image_path).splitlines():
                line = normalize_whitespace(raw_line)
                if line and not CNKI_FOOTER_PATTERN.search(line):
                    lines.append(line)

    blocks: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        match = HIGH_VALUE_OCR_PATTERN.search(line)
        if not match:
            index += 1
            continue
        block = re.split(r"\s{3,}", line[match.start() :].strip(), maxsplit=1)[0]
        for heading in KNOWN_MAIN_HEADINGS:
            compact = re.sub(r"\s+", "", block)
            if re.sub(r"\s+", "", heading) in compact:
                block = heading
                break
        if re.match(r"^\s*推荐\s*\d+\s*[:：]", block):
            cursor = index + 1
            while cursor < len(lines) and cursor <= index + 3:
                next_line = lines[cursor]
                next_match = HIGH_VALUE_OCR_PATTERN.search(next_line)
                if next_match and not re.search(r"推荐强度\s*[:：]", next_line):
                    break
                fragment = next_line.rsplit("”", 1)[-1].strip()
                block = f"{block}{fragment}"
                if "推荐强度" in next_line or "强)" in next_line or "强）" in next_line:
                    break
                cursor += 1
            strength_match = re.search(r"推荐强度\s*[:：]\s*(?:强|一般)\s*[)）]?", block)
            if strength_match:
                block = block[: strength_match.end()]
        blocks.append(block)
        index += 1
    best_recommendations: dict[str, str] = {}
    non_recommendations: list[str] = []
    for block in blocks:
        rec_match = re.match(r"^\s*推荐\s*(\d+)\s*[:：]", block)
        if rec_match:
            rec_id = rec_match.group(1)
            current = best_recommendations.get(rec_id)
            if current is None or _ocr_block_quality(block) > _ocr_block_quality(current):
                best_recommendations[rec_id] = block
            continue
        non_recommendations.append(block)

    deduped: list[str] = []
    seen: set[str] = set()
    for block in [*non_recommendations, *best_recommendations.values()]:
        key = re.sub(r"\s+", "", block)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(block)
    return deduped


def _ocr_block_quality(block: str) -> int:
    penalty = len(re.findall(r"[A-Za-z]{1,2}|[、。，]{2,}", block))
    reward = len(re.findall(r"[\u4e00-\u9fff]", block))
    return reward - penalty * 3


def _normalize_ocr_recommendation(block: str) -> str:
    text = normalize_whitespace(block)
    text = re.sub(r"^\s*推荐\s*(\d+)\s*[:：]\s*", r"推荐\1：", text)
    text = text.replace("(", "（").replace(")", "）")
    text = text.replace(";", "；")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"证据等级[:：]", "证据等级：", text)
    text = re.sub(r"推荐强度[:：]", "推荐强度：", text)
    text = re.sub(r"（证据等级：", "（证据等级：", text)
    return text


def _insert_recommendation_block(text: str, block: str) -> str:
    rec_match = re.match(r"^\s*推荐\s*(\d+)\s*[:：]", block)
    if not rec_match:
        return text
    rec_id = rec_match.group(1)
    normalized = _normalize_ocr_recommendation(block)
    if normalized[:12] in text or re.sub(r"\s+", "", normalized) in re.sub(r"\s+", "", text):
        return text
    before_pattern = RECOMMENDATION_INSERT_BEFORE.get(rec_id)
    if not before_pattern:
        return normalize_whitespace(f"{text}\n\n{normalized}")
    match = re.search(before_pattern, text)
    if not match:
        return normalize_whitespace(f"{text}\n\n{normalized}")
    return normalize_whitespace(f"{text[:match.start()].rstrip()}\n{normalized}\n\n{text[match.start():].lstrip()}")


def _insert_missing_ocr_blocks(text: str, ocr_blocks: list[str]) -> str:
    if not ocr_blocks:
        return text
    output = text
    additions: list[str] = []
    known_heading_keys = {re.sub(r"\s+", "", heading) for heading in KNOWN_MAIN_HEADINGS}
    for block in ocr_blocks:
        if not block:
            continue
        compact = re.sub(r"\s+", "", block)
        if compact in known_heading_keys:
            continue
        if re.match(r"^\s*推荐\s*\d+\s*[:：]", block):
            output = _insert_recommendation_block(output, block)
        elif block[:12] not in output:
            additions.append(block)
    if additions:
        output = normalize_whitespace(f"{output}\n\n" + "\n".join(additions))
    return output


def read_pdf_pages(
    pdf_path: str | Path,
    *,
    enable_double_column: bool = True,
    clean_text: bool = True,
    return_layout: bool = False,
    ocr_missing_high_value_text: bool = False,
    unit_exponent_ocr_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """使用 pdfplumber 按页读取 PDF 文本。"""
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("读取 PDF 需要安装 pdfplumber。") from exc

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {path}")

    pages: list[dict[str, Any]] = []
    with pdfplumber.open(str(path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(extra_attrs=["fontname", "size"]) or []
            kept_words = _filter_header_footer(words, float(page.height))
            two_column, split_x = _detect_two_column_kmeans(kept_words, float(page.width))
            table_bboxes = _find_table_bboxes(page)
            text_table_bodies: list[dict[str, Any]] = []

            if enable_double_column and two_column and split_x is not None:
                left_words = [word for word in kept_words if float(word["x0"]) < split_x]
                right_words = [word for word in kept_words if float(word["x0"]) >= split_x]
                left_rows_all = _merge_floating_token_rows(_cluster_rows(left_words))
                right_rows_all = _merge_floating_token_rows(_cluster_rows(right_words))
                text_table_bodies = [
                    *_extract_text_table_regions(left_rows_all, page_number, "txt_l"),
                    *_extract_text_table_regions(right_rows_all, page_number, "txt_r"),
                ]
                all_table_bboxes = [*table_bboxes, *[list(item["bbox"]) for item in text_table_bodies]]
                left_rows = _filter_table_rows(left_rows_all, all_table_bboxes)
                right_rows = _filter_table_rows(right_rows_all, all_table_bboxes)
                left_lines = _rows_to_layout_lines(left_rows)
                right_lines = _rows_to_layout_lines(right_rows)
                raw_text = _layout_lines_to_text(left_lines) + "\n\n" + _layout_lines_to_text(right_lines)
                layout_lines = left_lines + right_lines
            else:
                rows_all = _merge_floating_token_rows(_cluster_rows(kept_words))
                text_table_bodies = _extract_text_table_regions(rows_all, page_number)
                all_table_bboxes = [*table_bboxes, *[list(item["bbox"]) for item in text_table_bodies]]
                rows = _filter_table_rows(rows_all, all_table_bboxes)
                layout_lines = _rows_to_layout_lines(rows)
                raw_text = _layout_lines_to_text(layout_lines)

            unit_exponent_corrections: list[dict[str, Any]] = []
            layout_lines, unit_exponent_corrections = recover_unit_exponents_in_page(
                path,
                page_number - 1,
                layout_lines,
                unit_exponent_ocr_config,
            )
            if unit_exponent_corrections:
                layout_lines = _refresh_layout_line_offsets(layout_lines)
                raw_text = _layout_lines_to_text(layout_lines)

            if ocr_missing_high_value_text:
                raw_text = _insert_missing_ocr_blocks(raw_text, _extract_high_value_ocr_blocks(path, page_number - 1))

            text = clean_page_text(raw_text) if clean_text else normalize_whitespace(raw_text)
            tables = page.extract_tables() or []
            meta = {
                "two_column": bool(enable_double_column and two_column),
                "split_x": split_x,
                "num_words": len(words),
                "num_tables": len([table for table in tables if table and len(table) > 1]),
                "table_bboxes": table_bboxes,
                "text_table_bodies": text_table_bodies,
                "has_images": bool(getattr(page, "images", []) or []),
                "page_width": page.width,
                "page_height": page.height,
                "unit_exponent_corrections": unit_exponent_corrections,
            }
            if return_layout:
                meta["layout_lines"] = layout_lines
                meta["images"] = [
                    {
                        "object_id": f"F_p{page_number:03d}_{index:03d}",
                        "bbox": [
                            float(image["x0"]),
                            float(image["top"]),
                            float(image["x1"]),
                            float(image["bottom"]),
                        ],
                        "source": "pdfplumber_page_images",
                    }
                    for index, image in enumerate(getattr(page, "images", []) or [], start=1)
                    if all(key in image for key in ("x0", "top", "x1", "bottom"))
                ]
            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                    "raw_text": raw_text,
                    "meta": meta,
                }
            )
    return pages
