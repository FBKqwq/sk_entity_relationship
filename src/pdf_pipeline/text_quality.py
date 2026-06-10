"""Text quality checks for parsed PDF pages."""

from __future__ import annotations

import unicodedata
from typing import Any


DEFAULT_GARBLED_TEXT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "min_text_chars": 80,
    "max_cjk_ratio": 0.02,
    "min_ascii_letter_ratio": 0.20,
    "min_readable_ratio": 0.55,
    "max_weird_ratio": 0.35,
    "max_mojibake_ratio": 0.08,
}

MOJIBAKE_MARKERS = set(
    "锛紝銆鈥鐧濉炵患鍚緛璇婄枟瑙勮寖閮戞枃娲艁酶藛"
    "鐨鎬鐥鍙琛鍜鐜鑰绱鐐绠鑴娌鎮鐤婧澶鐢浜閲瑙鑶鑲鏈璇鎴鍦鍑鍒寰鏄鐪缁鍏闈搴瑕瀵鍐鎹"
)


def _merge_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_GARBLED_TEXT_CONFIG)
    if config:
        merged.update(config)
    return merged


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
    )


def _is_readable_non_cjk(char: str) -> bool:
    if char.isascii() and (char.isalpha() or char.isdigit()):
        return True
    if char in "，。；：、（）《》“”‘’！？-—/%‰±=+<>≤≥~～·,.!?;:()[]{}":
        return True
    return False


def analyze_parsed_pdf_text_quality(
    pages: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a quality report and whether parsed text is unusably garbled."""
    cfg = _merge_config(config)
    full_text = "\n".join(str(page.get("text", "") or "") for page in pages)
    chars = [char for char in full_text if not char.isspace()]
    total = len(chars)
    cjk_count = 0
    ascii_letter_count = 0
    digit_count = 0
    readable_count = 0
    replacement_count = 0
    question_mark_count = 0
    control_count = 0
    symbol_count = 0
    private_use_count = 0
    mojibake_count = 0

    for char in chars:
        category = unicodedata.category(char)
        is_cjk = _is_cjk(char)
        if is_cjk:
            cjk_count += 1
            readable_count += 1
        elif char.isascii() and char.isalpha():
            ascii_letter_count += 1
            readable_count += 1
        elif char.isdigit():
            digit_count += 1
            readable_count += 1
        elif _is_readable_non_cjk(char):
            readable_count += 1

        if char == "\ufffd":
            replacement_count += 1
        if char == "?":
            question_mark_count += 1
        if category.startswith("C"):
            control_count += 1
        if category.startswith("S"):
            symbol_count += 1
        if category == "Co":
            private_use_count += 1
        if char in MOJIBAKE_MARKERS:
            mojibake_count += 1

    def ratio(count: int) -> float:
        return round(count / total, 4) if total else 0.0

    cjk_ratio = ratio(cjk_count)
    ascii_letter_ratio = ratio(ascii_letter_count)
    readable_ratio = ratio(readable_count)
    suspicious_question_count = max(0, question_mark_count - 3)
    weird_count = replacement_count + suspicious_question_count + control_count + symbol_count + private_use_count
    weird_ratio = ratio(weird_count)
    mojibake_ratio = ratio(mojibake_count)

    min_text_chars = int(cfg.get("min_text_chars", 80))
    max_cjk_ratio = float(cfg.get("max_cjk_ratio", 0.02))
    min_ascii_letter_ratio = float(cfg.get("min_ascii_letter_ratio", 0.20))
    min_readable_ratio = float(cfg.get("min_readable_ratio", 0.55))
    max_weird_ratio = float(cfg.get("max_weird_ratio", 0.35))
    max_mojibake_ratio = float(cfg.get("max_mojibake_ratio", 0.08))

    reasons: list[str] = []
    if total < min_text_chars:
        reasons.append("too_little_extracted_text")
    if (
        cjk_ratio <= max_cjk_ratio
        and ascii_letter_ratio < min_ascii_letter_ratio
        and readable_ratio < min_readable_ratio
    ):
        reasons.append("low_chinese_and_low_readable_text")
    if (
        cjk_ratio <= max_cjk_ratio
        and ascii_letter_ratio < min_ascii_letter_ratio
        and weird_ratio >= max_weird_ratio
    ):
        reasons.append("high_symbol_or_control_ratio")
    if mojibake_ratio >= max_mojibake_ratio:
        reasons.append("high_mojibake_marker_ratio")

    skipped = bool(cfg.get("enabled", True)) and bool(reasons)
    return {
        "stage": "parsed_pdf_text_quality",
        "enabled": bool(cfg.get("enabled", True)),
        "status": "skipped_garbled_text" if skipped else "pass",
        "skip": skipped,
        "reasons": reasons,
        "total_chars": total,
        "cjk_chars": cjk_count,
        "ascii_letter_chars": ascii_letter_count,
        "digit_chars": digit_count,
        "readable_chars": readable_count,
        "replacement_chars": replacement_count,
        "question_mark_chars": question_mark_count,
        "control_chars": control_count,
        "symbol_chars": symbol_count,
        "private_use_chars": private_use_count,
        "mojibake_marker_chars": mojibake_count,
        "cjk_ratio": cjk_ratio,
        "ascii_letter_ratio": ascii_letter_ratio,
        "readable_ratio": readable_ratio,
        "weird_ratio": weird_ratio,
        "mojibake_ratio": mojibake_ratio,
        "thresholds": {
            "min_text_chars": min_text_chars,
            "max_cjk_ratio": max_cjk_ratio,
            "min_ascii_letter_ratio": min_ascii_letter_ratio,
            "min_readable_ratio": min_readable_ratio,
            "max_weird_ratio": max_weird_ratio,
            "max_mojibake_ratio": max_mojibake_ratio,
        },
    }
