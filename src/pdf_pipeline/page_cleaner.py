"""页文本清洗工具，复用旧 notebook 的行级去噪思想。"""

from __future__ import annotations

import re


ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\ufeff]")
NOISE_LINE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"中国实用外科杂志\s*20\d{2}年\d+月第\d+卷第\d+期"),
    re.compile(r"中华.*杂志\s*20\d{2}"),
    re.compile(r"Chin\s+J.*?(Vol\.|No\.)", re.IGNORECASE),
    re.compile(r"^\s*[·•・]?\s*\d+\s*[·•・]?\s*$"),
    re.compile(r"^\s*\d+(?:\s+\d+)+\s*$"),
    re.compile(r"^\s*指南与共识\s*$"),
    re.compile(r"^\s*20\d{2}[）)]\s*$"),
    re.compile(r"^\s*(通信作者|Corresponding\s+authors?)[:：]", re.IGNORECASE),
    re.compile(r"^\s*(收稿日期|本文编辑|引用本文)[:：]?"),
    re.compile(r"DOI[:：]?\s*10\.\d{4,9}/[^\s]+", re.IGNORECASE),
    re.compile(r"\b10\.\d{4,9}/[^\s]+", re.IGNORECASE),
]

NON_KNOWLEDGE_TAIL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?m)^\s*执笔\s*[:：]"),
    re.compile(r"(?m)^\s*编写组成员\s*[（(]?.*?[:：]"),
    re.compile(r"(?m)^\s*诊疗规范撰写组名单\s*[（(]?.*?[:：]"),
    re.compile(r"(?m)^\s*指南制定组名单\s*[（(]?.*?[:：]"),
    re.compile(r"(?m)^\s*专家组名单\s*[（(]?.*?[:：]"),
    re.compile(r"(?m)^\s*[（(]\s*按姓氏汉语拼音排\s*\n\s*序\s*[）)]\s*[:：]\s*$"),
    re.compile(r"(?m)^\s*所有作者均声明不存在利益冲突\s*$"),
    re.compile(r"[（(]\s*收稿[日期日][^）)]*[）)]"),
    re.compile(r"[（(]\s*本文编辑[^）)]*[）)]"),
    re.compile(r"(?m)^\s*参\s*考\s*文\s*献\s*$"),
    re.compile(r"(?m)^\s*考\s*文\s*献\s*$"),
    re.compile(r"(?m)^\s*(?:[0-9０-９一二三四五六七八九十]+[\.．、\s]*)?参考文献\s*$"),
    re.compile(r"(?im)^\s*(?:[0-9０-９]+[\.．、\s]*)?references\s*$"),
    re.compile(r"(?m)^\s*[［\[]\s*1\s*[］\]]"),
]

FRONT_MATTER_MARKERS = (
    "Guidelines",
    "Keywords",
    "关键词",
    "中图分类号",
    "文献标志码",
    "MedicalAssociation",
)
FRONT_MATTER_BODY_START_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?m)^急性胰腺炎指"),
    re.compile(r"(?m)^[\u4e00-\u9fff]{2,30}指因"),
]


def normalize_whitespace(text: str) -> str:
    """规范空白字符，保留段落边界。"""
    if not text:
        return ""
    text = text.replace("\u00a0", " ")
    text = ZERO_WIDTH_PATTERN.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])", "", text)
    return text.strip()


def chinese_ratio(text: str) -> float:
    """计算文本中中文字符占比。"""
    if not text:
        return 0.0
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    return chinese_count / max(1, len(text))


def remove_noise_lines(page_text: str) -> str:
    """移除页眉、页脚、页码、DOI 等常见医学期刊噪声行。"""
    kept: list[str] = []
    for line in page_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(pattern.search(stripped) for pattern in NOISE_LINE_PATTERNS):
            continue
        kept.append(line)
    return normalize_whitespace("\n".join(kept))


def remove_front_matter_block(page_text: str) -> str:
    """移除首页题名、作者、英文摘要关键词等题录噪声块。"""
    if not page_text:
        return ""
    head = page_text[:1200]
    if not any(marker in head for marker in FRONT_MATTER_MARKERS):
        return normalize_whitespace(page_text)
    starts = [
        match.start()
        for pattern in FRONT_MATTER_BODY_START_PATTERNS
        if (match := pattern.search(page_text)) is not None
    ]
    if not starts:
        return normalize_whitespace(page_text)
    cut = min(starts)
    if cut <= 0 or cut > 1600:
        return normalize_whitespace(page_text)
    return normalize_whitespace(page_text[cut:])


def clean_page_text(page_text: str) -> str:
    """执行完整页文本清洗。"""
    return remove_front_matter_block(remove_noise_lines(normalize_whitespace(page_text)))


def remove_non_knowledge_tail(text: str) -> str:
    """从全文中截断执笔、撰写组名单、参考文献等后置非知识段落。"""
    if not text:
        return ""
    cut_positions = [
        match.start()
        for pattern in NON_KNOWLEDGE_TAIL_PATTERNS
        if (match := pattern.search(text)) is not None
    ]
    if not cut_positions:
        return normalize_whitespace(text)
    return normalize_whitespace(text[: min(cut_positions)])
