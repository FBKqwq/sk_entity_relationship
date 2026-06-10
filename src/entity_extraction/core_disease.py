"""Infer document-level core Disease entities from expert consensus titles."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.entity_extraction.entity_normalizer import normalize_entity_text


DOC_TITLE_MARKERS = (
    "诊断和治疗指南",
    "诊断与治疗指南",
    "诊治指南",
    "诊疗指南",
    "防治专家共识",
    "防治指南",
    "治疗指南",
    "专家共识",
    "诊疗规范",
    "防治规范",
    "管理推荐",
    "临床路径",
    "指南",
    "共识",
    "规范",
)

LEADING_SCOPE_PREFIXES = ("中国", "中华", "我国")
CORE_DISEASE_PREFIX_MODIFIERS = (
    "急性",
    "慢性",
    "重症",
    "轻症",
    "轻型",
    "中型",
    "中度",
    "重型",
    "复杂性",
    "单纯性",
    "原发性",
    "继发性",
)


def _clean_title(title: str) -> str:
    cleaned = Path(str(title)).stem
    cleaned = cleaned.strip().strip("《》")
    cleaned = re.sub(r"[（(]\s*(?:19|20)\d{2}(?:\s*版)?\s*[）)]", "", cleaned)
    cleaned = re.sub(r"[（(]\s*\d{4}\s*[）)]", "", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned.strip(" ：:，,。.")


def infer_core_disease_name(source_title: str | None = None, pdf_path: str | Path | None = None) -> str:
    """Infer the parent Disease name from a guideline/consensus document title.

    Disease means the document-level core disease category, normally the phrase
    before "诊治指南" / "专家共识" / "诊疗规范" in the PDF title.
    """

    raw_title = source_title or (Path(pdf_path).stem if pdf_path else "")
    title = _clean_title(raw_title)
    if not title:
        return ""

    candidate = ""
    for marker in DOC_TITLE_MARKERS:
        if marker in title:
            candidate = title.split(marker, 1)[0]
            break
    if not candidate:
        return ""

    for prefix in LEADING_SCOPE_PREFIXES:
        if candidate.startswith(prefix) and len(candidate) > len(prefix) + 2:
            candidate = candidate[len(prefix) :]
            break

    candidate = re.sub(r"^[\u4e00-\u9fff]{0,12}(?:学会|协会|分会|学组)", "", candidate)
    return normalize_entity_text(candidate)


def build_core_disease_entity(
    source_title: str | None = None,
    pdf_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Build one document-level Disease candidate from the PDF title."""

    name = infer_core_disease_name(source_title=source_title, pdf_path=pdf_path)
    if not name:
        return None
    evidence = str(source_title or (Path(pdf_path).stem if pdf_path else name))
    return {
        "entity_type": "diseases",
        "name": name,
        "properties": {
            "disease_id": "",
            "disease_name": name,
            "normalized_name": normalize_entity_text(name),
            "aliases": [],
            "source_document_ids": [],
            "evidence_ids": [],
            "confidence": 0.98,
        },
        "evidence": evidence,
        "confidence": 0.98,
        "override_lv1": False,
    }


def core_disease_anchors(core_disease: str) -> tuple[str, ...]:
    """Return conservative surface anchors for judging Disease-child relation."""

    normalized = normalize_entity_text(core_disease)
    anchors: list[str] = []
    if normalized:
        anchors.append(normalized)
    for modifier in CORE_DISEASE_PREFIX_MODIFIERS:
        if normalized.startswith(modifier) and len(normalized) > len(modifier) + 1:
            base = normalized[len(modifier) :]
            anchors.append(base)
            organ_match = re.match(r"([\u4e00-\u9fff]{2,6})(?:炎|感染|综合征|病|症)$", base)
            if organ_match:
                anchors.append(organ_match.group(1))
            break
    return tuple(dict.fromkeys(anchor for anchor in anchors if len(anchor) >= 2))


def is_core_disease_child_name(name: str, core_disease: str, evidence: str = "") -> bool:
    """Return whether a candidate sub-disease belongs under the document Disease.

    The check is intentionally conservative: a sub-disease must contain the
    core disease name or a stable core anchor such as "胰腺炎" from "急性胰腺炎".
    Exact parent-name repeats are not child diagnoses.
    """

    normalized_name = normalize_entity_text(name)
    normalized_core = normalize_entity_text(core_disease)
    if not normalized_name or not normalized_core:
        return True
    if normalized_name == normalized_core:
        return False
    return any(anchor in normalized_name for anchor in core_disease_anchors(normalized_core))
