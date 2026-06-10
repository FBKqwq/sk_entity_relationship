"""Dual-source TOC parsing, LLM arbitration, and local quality checks."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.pdf_pipeline.section_detector import HeadingAnchor, Section, _collect_heading_anchors
from src.utils.llm_client import chat_completion_text, is_llm_available, load_teacher_llm_config

BODY_TITLE = "正文"
MAX_TITLE_CHARS = 90
SUSPICIOUS_TOKENS = ("vs.", "rr", "ci", "p<", "p=", "g/d", "mg", "mmol", "%")
NOISE_TITLE_PATTERNS = (
    r"^(?:[0-9０-９一二三四五六七八九十]+[\.．、\s]*)?参考文献$",
    r"^(?:[0-9０-９]+[\.．、\s]*)?references$",
    r"^通信作者",
    r"^基金项目",
    r"^作者单位",
    r"^利益冲突",
    r"^\[\d+\]",
    r"^［\d+］",
)

LLM_EXTRACT_SYSTEM_PROMPT = (
    "You extract table-of-contents headings from medical PDF text. "
    "Only copy headings that appear verbatim in the input. "
    "Return strict JSON with key sections only."
)

LLM_EXTRACT_USER_TEMPLATE = """
Extract the detailed section heading structure from the PDF text below.
Do not rewrite, summarize, normalize, translate, or invent any character.
Each title and evidence_text must be copied verbatim from the input.
Return JSON only:
{{
  "sections": [
    {{"level": 1, "title": "...", "page": 1, "evidence_text": "..."}}
  ]
}}
If a level is uncertain, use null. Do not include body paragraphs.

Pages:
{body}
""".strip()

LLM_JUDGE_SYSTEM_PROMPT = (
    "You arbitrate between a rule-based TOC parser and an LLM TOC parser. "
    "Accept only headings supported by source evidence. Return strict JSON."
)

LLM_JUDGE_USER_TEMPLATE = """
Merge and judge the candidate TOC entries.
Rules:
- A final accepted title must be copied from evidence_text.
- Prefer entries that exact-match the PDF text and have plausible heading shape.
- Reject invented, paragraph-like, reference-list, author, or statistical lines.
- Preserve a hierarchical level from 1 to 4 when evidence supports it.

Return JSON only:
{{
  "sections": [
    {{
      "level": 1,
      "title": "...",
      "page": 1,
      "evidence_text": "...",
      "decision": "accept",
      "reason": "..."
    }}
  ]
}}

Document metrics:
{metrics}

Candidates:
{candidates}
""".strip()


@dataclass(frozen=True)
class TocCandidate:
    title: str
    level: int | None
    start: int | None
    page: int | None
    source: str
    evidence_text: str
    confidence: float


@dataclass(frozen=True)
class TocDecision:
    title: str
    level: int
    start: int
    page: int | None
    sources: list[str]
    evidence_text: str
    decision: str
    reason: str
    confidence: float


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _page_for_offset(char_to_page: list[int], start: int | None) -> int | None:
    if start is None or not char_to_page:
        return None
    safe = max(0, min(int(start), len(char_to_page) - 1))
    return int(char_to_page[safe])


def _find_title_start(full_text: str, title: str, page: int | None = None) -> int | None:
    clean = str(title or "")
    if not clean:
        return None
    pos = full_text.find(clean)
    if pos >= 0:
        return pos
    normalized_title = _normalize_space(clean)
    if not normalized_title:
        return None
    for match in re.finditer(re.escape(normalized_title), _normalize_space(full_text)):
        return match.start()
    return None


def _title_shape_issue(title: str) -> str:
    clean = _normalize_space(title)
    if not clean:
        return "empty_title"
    if len(clean) > MAX_TITLE_CHARS:
        return "title_too_long"
    if clean.endswith(("，", ",", "、")):
        return "truncated_line_fragment"
    if clean.count("。") >= 1 or clean.count("；") >= 1 or clean.count(";") >= 1 or clean.count("，") >= 1:
        return "paragraph_punctuation"
    if clean.count(".") >= 5:
        return "paragraph_like"
    if ("：" in clean or ":" in clean) and len(re.split(r"[：:]", clean, maxsplit=1)[-1].strip()) >= 12:
        return "colon_body_fragment"
    if re.match(r"^\d{1,2}[．.、)]", clean) and re.search(r"[，,：:。；;]", clean):
        return "numbered_body_fragment"
    lower = clean.lower()
    for token in SUSPICIOUS_TOKENS:
        if token in lower:
            return f"contains_token:{token}"
    for pattern in NOISE_TITLE_PATTERNS:
        if re.search(pattern, clean, flags=re.IGNORECASE):
            return "noise_section"
    if re.match(r"^\d+(?:\.\d+)?\s*[%~]", clean):
        return "numeric_measurement"
    return ""


def _candidate_validation(candidate: TocCandidate, full_text: str) -> dict[str, Any]:
    evidence = candidate.evidence_text or candidate.title
    exact = bool(evidence and evidence in full_text)
    start = candidate.start if candidate.start is not None else _find_title_start(full_text, evidence)
    issue = _title_shape_issue(candidate.title)
    return {
        "title": candidate.title,
        "source": candidate.source,
        "exact_match": exact,
        "start": start,
        "page": candidate.page,
        "shape_issue": issue,
        "valid": bool(exact and start is not None and not issue),
    }


def collect_rule_toc_candidates(full_text: str, char_to_page: list[int]) -> list[TocCandidate]:
    candidates: list[TocCandidate] = []
    for anchor in _collect_heading_anchors(full_text):
        candidates.append(
            TocCandidate(
                title=anchor.title,
                level=anchor.level,
                start=anchor.start,
                page=_page_for_offset(char_to_page, anchor.start),
                source="rule",
                evidence_text=anchor.title,
                confidence=0.80,
            )
        )
    return candidates


def _pages_prompt_body(page_text: dict[int, str], max_chars: int) -> tuple[str, bool]:
    parts: list[str] = []
    used = 0
    truncated = False
    for page in sorted(page_text):
        block = f"\n[PAGE {page}]\n{page_text[page] or ''}\n"
        if used + len(block) > max_chars:
            remaining = max(0, max_chars - used)
            if remaining:
                parts.append(block[:remaining])
            truncated = True
            break
        parts.append(block)
        used += len(block)
    return "".join(parts), truncated


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_llm_sections(raw: str, full_text: str, source: str) -> tuple[list[TocCandidate], str]:
    parsed = _extract_json_object(raw)
    if not parsed:
        return [], "invalid_json"
    rows = parsed.get("sections")
    if not isinstance(rows, list):
        return [], "missing_sections"

    candidates: list[TocCandidate] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        title = _normalize_space(str(item.get("title", "")))
        evidence = str(item.get("evidence_text", title))
        if not title:
            continue
        try:
            raw_level = item.get("level")
            level = None if raw_level is None else max(1, min(4, int(raw_level)))
        except (TypeError, ValueError):
            level = None
        try:
            page = None if item.get("page") is None else int(item.get("page"))
        except (TypeError, ValueError):
            page = None
        start = _find_title_start(full_text, evidence)
        candidates.append(
            TocCandidate(
                title=title,
                level=level,
                start=start,
                page=page,
                source=source,
                evidence_text=evidence,
                confidence=0.70,
            )
        )
    return candidates, ""


def collect_llm_toc_candidates(
    page_text: dict[int, str],
    full_text: str,
    *,
    llm_config_path: str | Path,
    config: dict[str, Any],
) -> tuple[list[TocCandidate], dict[str, Any]]:
    report: dict[str, Any] = {"enabled": bool(config.get("enabled", False)), "status": "skipped", "reason": ""}
    if not report["enabled"]:
        return [], report
    teacher = load_teacher_llm_config(llm_config_path)
    if not is_llm_available(teacher):
        report.update({"status": "disabled", "reason": "teacher_llm_unavailable"})
        return [], report

    body, truncated = _pages_prompt_body(page_text, int(config.get("max_input_chars", 120000)))
    result = chat_completion_text(
        LLM_EXTRACT_USER_TEMPLATE.format(body=body),
        system_prompt=LLM_EXTRACT_SYSTEM_PROMPT,
        model_key="model_name",
        config_path=llm_config_path,
    )
    report["truncated_input"] = truncated
    report["model"] = result.get("model")
    if result.get("status") != "ok":
        report.update({"status": "failed", "reason": str(result.get("reason", ""))})
        return [], report
    candidates, err = _parse_llm_sections(str(result.get("text", "")), full_text, "llm")
    if err:
        report.update({"status": "failed", "reason": err})
        return [], report
    report.update({"status": "ok", "candidate_count": len(candidates)})
    return candidates, report


def _dedupe_candidates(candidates: list[TocCandidate], full_text: str) -> list[TocDecision]:
    grouped: dict[tuple[int, str], list[TocCandidate]] = {}
    for candidate in candidates:
        validation = _candidate_validation(candidate, full_text)
        if not validation["valid"]:
            continue
        start = int(validation["start"])
        key = (start, _normalize_space(candidate.title))
        grouped.setdefault(key, []).append(candidate)

    decisions: list[TocDecision] = []
    for (start, title), group in grouped.items():
        sources = sorted({item.source for item in group})
        levels = [item.level for item in group if item.level is not None]
        level = int(levels[0]) if levels else 1
        confidence = max(item.confidence for item in group)
        if len(sources) > 1:
            confidence = max(confidence, 0.90)
        first = group[0]
        decisions.append(
            TocDecision(
                title=title,
                level=max(1, min(4, level)),
                start=start,
                page=first.page,
                sources=sources,
                evidence_text=first.evidence_text,
                decision="accept",
                reason="local_validated" if len(sources) == 1 else "rule_llm_agree",
                confidence=round(confidence, 4),
            )
        )
    return sorted(decisions, key=lambda item: item.start)


def _local_metrics(
    candidates: list[TocCandidate],
    decisions: list[TocDecision],
    full_text: str,
    total_pages: int,
) -> dict[str, Any]:
    validations = [_candidate_validation(candidate, full_text) for candidate in candidates]
    valid_count = sum(1 for item in validations if item["valid"])
    page_count = max(1, int(total_pages or 1))
    title_count = len(decisions)
    starts = [item.start for item in decisions]
    monotonic_pairs = sum(1 for idx in range(len(starts) - 1) if starts[idx] < starts[idx + 1])
    level_jumps = sum(
        1
        for idx in range(len(decisions) - 1)
        if decisions[idx + 1].level - decisions[idx].level > 1
    )
    density = title_count / page_count
    suspicious_density = density < 0.15 or density > 6.0
    return {
        "candidate_count": len(candidates),
        "accepted_title_count": title_count,
        "title_match_rate": round(valid_count / len(candidates), 4) if candidates else 0.0,
        "heading_noise_rate": round(
            sum(1 for item in validations if item["shape_issue"]) / len(validations), 4
        )
        if validations
        else 0.0,
        "page_density": round(density, 4),
        "suspicious_density": suspicious_density,
        "page_monotonicity_rate": round(monotonic_pairs / max(1, len(starts) - 1), 4) if starts else 0.0,
        "level_jump_count": level_jumps,
    }


def _judge_with_llm(
    candidates: list[TocCandidate],
    local_decisions: list[TocDecision],
    metrics: dict[str, Any],
    full_text: str,
    *,
    llm_config_path: str | Path,
    config: dict[str, Any],
) -> tuple[list[TocDecision], dict[str, Any]]:
    report: dict[str, Any] = {"enabled": bool(config.get("enabled", False)), "status": "skipped", "reason": ""}
    if not report["enabled"]:
        return local_decisions, report
    teacher = load_teacher_llm_config(llm_config_path)
    if not is_llm_available(teacher):
        report.update({"status": "disabled", "reason": "teacher_llm_unavailable"})
        return local_decisions, report

    candidate_payload = [
        {
            **asdict(candidate),
            "local_validation": _candidate_validation(candidate, full_text),
        }
        for candidate in candidates
    ]
    prompt = LLM_JUDGE_USER_TEMPLATE.format(
        metrics=json.dumps(metrics, ensure_ascii=False),
        candidates=json.dumps(candidate_payload, ensure_ascii=False),
    )
    result = chat_completion_text(
        prompt,
        system_prompt=LLM_JUDGE_SYSTEM_PROMPT,
        model_key="model_name",
        config_path=llm_config_path,
    )
    report["model"] = result.get("model")
    if result.get("status") != "ok":
        report.update({"status": "failed", "reason": str(result.get("reason", ""))})
        return local_decisions, report

    judged_candidates, err = _parse_llm_sections(str(result.get("text", "")), full_text, "judge")
    if err:
        report.update({"status": "failed", "reason": err})
        return local_decisions, report
    judged = _dedupe_candidates(judged_candidates, full_text)
    if not judged:
        report.update({"status": "failed", "reason": "judge_returned_no_valid_sections"})
        return local_decisions, report
    report.update({"status": "ok", "accepted_title_count": len(judged)})
    return judged, report


def _sections_from_decisions(decisions: list[TocDecision], full_text: str) -> list[Section]:
    if not decisions:
        return [Section(title=BODY_TITLE, start=0, end=len(full_text), path=[BODY_TITLE])]

    anchors = [HeadingAnchor(title=item.title, start=item.start, level=item.level) for item in decisions]
    anchors = sorted(anchors, key=lambda item: item.start)
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


def build_sections_with_dual_toc(
    page_text: dict[int, str],
    full_text: str,
    char_to_page: list[int],
    *,
    total_pages: int,
    llm_config_path: str | Path,
    config: dict[str, Any],
) -> tuple[list[Section], dict[str, Any]]:
    """Build sections with rule TOC, optional LLM TOC, optional LLM judge, and local checks."""
    cfg = config or {}
    rule_candidates = collect_rule_toc_candidates(full_text, char_to_page)
    llm_candidates, llm_report = collect_llm_toc_candidates(
        page_text,
        full_text,
        llm_config_path=llm_config_path,
        config=cfg.get("first_layer_llm") or {},
    )
    candidates = [*rule_candidates, *llm_candidates]
    local_decisions = _dedupe_candidates(candidates, full_text)
    metrics = _local_metrics(candidates, local_decisions, full_text, total_pages)
    judged_decisions, judge_report = _judge_with_llm(
        candidates,
        local_decisions,
        metrics,
        full_text,
        llm_config_path=llm_config_path,
        config=cfg.get("judge_llm") or {},
    )
    final_metrics = _local_metrics(
        [TocCandidate(d.title, d.level, d.start, d.page, "+".join(d.sources), d.evidence_text, d.confidence) for d in judged_decisions],
        judged_decisions,
        full_text,
        total_pages,
    )
    report = {
        "enabled": True,
        "mode": "dual_toc_v1",
        "rule_candidate_count": len(rule_candidates),
        "llm_candidate_count": len(llm_candidates),
        "local_accepted_title_count": len(local_decisions),
        "final_accepted_title_count": len(judged_decisions),
        "llm_extract_report": llm_report,
        "llm_judge_report": judge_report,
        "local_metrics": metrics,
        "final_metrics": final_metrics,
        "final_toc": [asdict(item) for item in judged_decisions],
    }
    return _sections_from_decisions(judged_decisions, full_text), report
