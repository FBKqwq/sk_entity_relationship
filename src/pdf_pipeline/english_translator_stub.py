"""英文翻译接口，支持禁用时安全回退。"""

from __future__ import annotations

import re
from pathlib import Path

from src.pdf_pipeline.page_cleaner import chinese_ratio
from src.utils.llm_client import chat_completion_text


TRANSLATION_SYSTEM_PROMPT = (
    "你是医学文献翻译助手。只翻译用户提供的英文原文，不补充原文外信息。"
    "保留医学缩写、数值、单位、引用编号和专有名词。输出简体中文。"
)


def is_english_paragraph(text: str, *, min_length: int = 80) -> bool:
    """判断段落是否需要英文翻译。"""
    stripped = text.strip()
    if len(stripped) < min_length:
        return False
    letters = len(re.findall(r"[A-Za-z]", stripped))
    return letters / max(1, len(stripped)) > 0.45 and chinese_ratio(stripped) < 0.2


def translate_english_text(text: str, *, config_path: str | Path | None = None) -> dict[str, object]:
    """调用翻译模型翻译单段英文，禁用时返回原文。"""
    if not text.strip():
        return {"status": "empty", "text": text, "source_text": text, "model": None}

    prompt = (
        "请将以下英文医学文本翻译为简体中文。"
        "不得添加解释，不得省略原文信息，只输出译文。\n\n"
        f"{text}"
    )
    result = chat_completion_text(
        prompt,
        system_prompt=TRANSLATION_SYSTEM_PROMPT,
        model_key="TR_model_name",
        config_path=config_path,
    )
    if result["status"] != "ok" or not result.get("text"):
        return {
            "status": "stub",
            "text": text,
            "source_text": text,
            "reason": result.get("reason", "翻译模型未返回内容。"),
            "model": result.get("model"),
        }
    return {
        "status": "translated",
        "text": str(result["text"]),
        "source_text": text,
        "model": result.get("model"),
    }


def translate_english_paragraphs(
    page_text: dict[int, str],
    *,
    config_path: str | Path | None = None,
) -> tuple[dict[int, str], list[int], list[int]]:
    """翻译页文本中的英文段落，返回新页文本、已翻译页和待处理页。"""
    translated_pages: list[int] = []
    pending_pages: list[int] = []
    output: dict[int, str] = {}

    for page_number, text in page_text.items():
        paragraphs = text.split("\n\n")
        changed = False
        new_paragraphs: list[str] = []
        for paragraph in paragraphs:
            if not is_english_paragraph(paragraph):
                new_paragraphs.append(paragraph)
                continue
            result = translate_english_text(paragraph, config_path=config_path)
            if result["status"] == "translated":
                changed = True
                new_paragraphs.append(str(result["text"]))
            else:
                pending_pages.append(page_number)
                new_paragraphs.append(paragraph)
        if changed:
            translated_pages.append(page_number)
        output[page_number] = "\n\n".join(new_paragraphs)
    return output, sorted(set(translated_pages)), sorted(set(pending_pages))


def translate_english_stub(text: str) -> tuple[str, list[str]]:
    """兼容旧接口：默认不翻译，仅返回原文与空标记。"""
    return text, []
