"""章节识别工具，复用旧 notebook 的 H1/H2 规则并补充医学标题。"""

from __future__ import annotations

import re
from dataclasses import dataclass


COMMON_MEDICAL_SECTIONS = [
    "摘要",
    "引言",
    "临床表现",
    "诊断",
    "鉴别诊断",
    "检查",
    "治疗",
    "推荐意见",
    "共识建议",
    "病理生理机制",
    "参考文献",
]

COMMON_MEDICAL_SECTIONS.extend(["概述", "摘要", "临床表现", "诊断", "治疗", "治疗原则", "治疗方案", "预后", "参考文献"])
BODY_TITLE = "正文"
REAL_CN_NUM = "一二三四五六七八九十"

H2_PATTERN = re.compile(
    r"(?m)^\s*((?:\d{1,2})[.．、]\s*(?!\d|%)\S.+|(?:\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\s+\S.+)$"
)
LINE_PATTERN = re.compile(r"(?m)^([^\n]+)$")
CN_NUM = "一二三四五六七八九十百零〇两"
EN_SECTION_WORDS = {
    "abstract",
    "introduction",
    "background",
    "methods",
    "materials and methods",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
}
WEAK_HEADING_SUFFIXES = ("比较", "解读", "概述", "总论", "小结", "总结")
NUMERIC_HEADING_BLOCKLIST = ("vs.", "rr", "ci", "p<", "p=", "g/d", "mg", "mmol", "%")


@dataclass(frozen=True)
class Section:
    """全文中的章节范围。"""

    title: str
    start: int
    end: int
    path: list[str]


@dataclass(frozen=True)
class HeadingAnchor:
    title: str
    start: int
    level: int


def normalize_section_title(title: str) -> str:
    """清理章节标题首尾空白。"""
    return re.sub(r"\s+", " ", title).strip()


def _common_heading_matches(full_text: str) -> list[re.Match[str]]:
    escaped = [re.escape(item) for item in COMMON_MEDICAL_SECTIONS]
    pattern = re.compile(r"(?m)^\s*(" + "|".join(escaped) + r")\s*$")
    return list(pattern.finditer(full_text))


def _looks_like_heading_line(title: str) -> bool:
    """过滤明显不是标题的长句与噪声行。"""
    clean = normalize_section_title(title)
    if not clean:
        return False
    if len(clean) > 90:
        return False
    if clean.count("。") >= 2:
        return False
    return True


def _normalize_heading_candidate(title: str) -> str:
    """标准化候选标题，兼容 OCR 产生的方括号包裹。"""
    clean = normalize_section_title(title)
    if clean.startswith("[") and clean.endswith("]") and len(clean) > 2:
        clean = clean[1:-1].strip()
    if clean.startswith("【") and clean.endswith("】") and len(clean) > 2:
        clean = clean[1:-1].strip()
    return clean


def _heading_level_from_title(title: str) -> int | None:
    """识别标题层级，返回 1~4。"""
    clean = _normalize_heading_candidate(title)
    if not _looks_like_heading_line(clean):
        return None

    if clean in COMMON_MEDICAL_SECTIONS:
        return 1
    if clean.lower() in EN_SECTION_WORDS:
        return 1
    if re.match(rf"^第[{REAL_CN_NUM}\d]+[章节篇部分]\s*\S*", clean):
        return 1
    if re.match(r"^\d{1,2}\s+[\u4e00-\u9fff]\S+", clean):
        return 1
    if re.match(r"^\d{1,2}\.\d{1,2}\s*[\u4e00-\u9fff]\S+", clean):
        return 2
    if re.match(rf"^[{REAL_CN_NUM}]{{1,3}}[、.．\s]\s*\S+", clean):
        return 1
    if re.match(rf"^[（(][{REAL_CN_NUM}]{{1,3}}[）)]\s*\S+", clean):
        return 2
    if re.match(r"^\d{1,2}[）)]\s*\S+", clean):
        return 3
    numeric_real = re.match(r"^(\d{1,2}(?:\.\d{1,2}){1,3})\s*(?![%~])\S+", clean)
    if numeric_real:
        lower_clean = clean.lower()
        if len(clean) > 60:
            return None
        if not re.search(r"[\u4e00-\u9fff]", clean):
            return None
        if any(token in lower_clean for token in NUMERIC_HEADING_BLOCKLIST):
            return None
        if re.search(r"[，,。；;/]", clean):
            return None
        depth = numeric_real.group(1).count(".") + 1
        return min(depth + 2, 4)
    if re.match(rf"^第[{CN_NUM}\d]+[章节篇部分卷]\s*\S*", clean):
        return 1
    if re.match(rf"^[{CN_NUM}]{{1,3}}\s*[、.．]\s*\S+", clean):
        return 1
    if re.match(rf"^[（(][{CN_NUM}]{{1,3}}[)）]\s*\S+", clean):
        return 2
    if re.match(r"^\d{1,2}[)）]\s*\S+", clean):
        return 2
    if re.match(r"^\d{1,2}[.．、]\s*(?!\d|%)\S+", clean):
        return 2
    if re.match(r"^\d{1,2}(?=\d{4}年).{0,40}(比较|解读)$", clean):
        return 2

    numeric_match = re.match(r"^(\d{1,2}(?:\.\d{1,2}){1,3})\s*(?![%~])\S+", clean)
    if numeric_match:
        lower_clean = clean.lower()
        if len(clean) > 60:
            return None
        if any(token in lower_clean for token in NUMERIC_HEADING_BLOCKLIST):
            return None
        if re.search(r"[，,。；;]", clean):
            return None
        if not re.search(r"[\u4e00-\u9fff]", clean):
            return None
        depth = numeric_match.group(1).count(".") + 1
        return min(depth + 1, 4)
    if any(clean.endswith(suffix) for suffix in WEAK_HEADING_SUFFIXES) and len(clean) <= 40:
        if "指南" in clean or "共识" in clean:
            return 2
    return None


def _collect_heading_anchors(full_text: str) -> list[HeadingAnchor]:
    """提取全文标题锚点并附带层级。"""
    anchors: list[HeadingAnchor] = []
    for line_match in LINE_PATTERN.finditer(full_text):
        raw_line = line_match.group(1)
        title = normalize_section_title(raw_line)
        level = _heading_level_from_title(title)
        if level is None:
            continue
        anchors.append(HeadingAnchor(title=title, start=line_match.start(1), level=level))

    for match in _common_heading_matches(full_text):
        title = normalize_section_title(match.group(1))
        anchors.append(HeadingAnchor(title=title, start=match.start(1), level=1))

    anchors = sorted(anchors, key=lambda item: item.start)
    deduped: list[HeadingAnchor] = []
    last_start = -1
    for anchor in anchors:
        if anchor.start == last_start:
            continue
        deduped.append(anchor)
        last_start = anchor.start
    return deduped


def detect_sections(full_text: str) -> list[Section]:
    """按层级标题识别章节，失败时回退为全文。"""
    anchors = _collect_heading_anchors(full_text)
    if not anchors:
        return [Section(title="正文", start=0, end=len(full_text), path=["正文"])]

    base_level = min(anchor.level for anchor in anchors)
    sections: list[Section] = []
    if anchors[0].start > 0 and full_text[: anchors[0].start].strip():
        sections.append(Section(title="前言/摘要", start=0, end=anchors[0].start, path=["正文", "前言/摘要"]))

    title_stack: list[str] = []
    for index, anchor in enumerate(anchors):
        start = anchor.start
        end = anchors[index + 1].start if index + 1 < len(anchors) else len(full_text)
        effective_level = max(1, anchor.level - base_level + 1)
        while len(title_stack) >= effective_level:
            title_stack.pop()
        title_stack.append(anchor.title)
        path = ["正文", *title_stack]
        title = anchor.title
        sections.append(Section(title=title, start=start, end=end, path=path))
    return sections


def detect_sections(full_text: str) -> list[Section]:
    """Detect sections and return normalized real-text root paths."""
    anchors = _collect_heading_anchors(full_text)
    if not anchors:
        return [Section(title=BODY_TITLE, start=0, end=len(full_text), path=[BODY_TITLE])]

    base_level = min(anchor.level for anchor in anchors)
    sections: list[Section] = []
    if anchors[0].start > 0 and full_text[: anchors[0].start].strip():
        sections.append(Section(title="前言/概述", start=0, end=anchors[0].start, path=[BODY_TITLE, "前言/概述"]))

    title_stack: list[str] = []
    for index, anchor in enumerate(anchors):
        start = anchor.start
        end = anchors[index + 1].start if index + 1 < len(anchors) else len(full_text)
        effective_level = max(1, anchor.level - base_level + 1)
        while len(title_stack) >= effective_level:
            title_stack.pop()
        title_stack.append(anchor.title)
        sections.append(Section(title=anchor.title, start=start, end=end, path=[BODY_TITLE, *title_stack]))
    return sections


def find_subsection_spans(section_text: str) -> list[tuple[int, int]]:
    """在过长章节内按二级标题切分。"""
    anchors = _collect_heading_anchors(section_text)
    subsection_anchors = [anchor for anchor in anchors if anchor.level >= 2]
    if not subsection_anchors:
        return []
    spans: list[tuple[int, int]] = []
    for index, anchor in enumerate(subsection_anchors):
        start = anchor.start
        end = subsection_anchors[index + 1].start if index + 1 < len(subsection_anchors) else len(section_text)
        spans.append((start, end))
    return spans
