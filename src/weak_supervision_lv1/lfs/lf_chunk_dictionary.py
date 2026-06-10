"""Chunk dictionary LF for Lv1."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.weak_supervision.common.lf_output import EvidenceSpan, LFOutput
from src.weak_supervision_lv1.chunk_lf_base import ChunkLabelingFunction
from src.weak_supervision_lv1.lfs._chunk_lf_utils import (
    DEFAULT_LABELS,
    abstain,
    chunk_text,
    find_literal_spans,
    load_weak_supervision_config,
    make_output,
    merge_evidence,
)


CONFIG_TERM_GROUPS = {
    "disease_terms": ("diseases", "sub_diseases"),
    "sub_disease_terms": ("sub_diseases",),
    "symptom_terms": ("symptoms",),
    "indicator_terms": ("tests",),
    "test_terms": ("tests",),
    "treatment_terms": ("treatments", "plans"),
    "plan_terms": ("plans",),
    "mechanism_terms": ("pathogeneses",),
    "pathogenesis_terms": ("pathogeneses",),
    "etiology_terms": ("etiologies",),
}

SEED_TERMS: dict[str, tuple[str, ...]] = {
    "diseases": (
        "综合征",
        "疾病",
        "炎症性疾病",
        "自身免疫病",
        "感染性疾病",
    ),
    "sub_diseases": (
        "原发性",
        "继发性",
        "确诊",
        "诊断为",
        "分型",
        "亚型",
    ),
    "symptoms": (
        "发热",
        "乏力",
        "疼痛",
        "皮疹",
        "口干",
        "眼干",
        "腹痛",
        "腹泻",
        "关节痛",
        "血栓形成",
        "流产",
    ),
    "tests": (
        "CRP",
        "C反应蛋白",
        "PCT",
        "ESR",
        "血沉",
        "ANA",
        "抗体",
        "抗SSA",
        "抗SSB",
        "唇腺活检",
        "评分",
        "阳性",
        "阴性",
    ),
    "treatments": (
        "治疗原则",
        "治疗目标",
        "规范治疗",
        "合理诊治",
        "控制炎症",
    ),
    "plans": (
        "治疗方案",
        "药物治疗",
        "随访",
        "减量",
        "诱导缓解",
        "维持缓解",
    ),
    "methods": (
        "糖皮质激素",
        "免疫抑制剂",
        "生物制剂",
        "手术",
        "抗凝",
        "抗血小板",
        "美沙拉嗪",
    ),
    "etiologies": (
        "病因",
        "诱因",
        "危险因素",
        "遗传",
        "病毒感染",
        "病原体",
        "基因突变",
    ),
    "pathogeneses": (
        "发病机制",
        "免疫紊乱",
        "炎症反应",
        "炎症级联",
        "病理生理",
        "结构变化",
    ),
}


def _unique_terms(terms: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        clean = str(term).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        unique.append(clean)
    return tuple(unique)


def _load_terms(config_path: str | Path | None) -> dict[str, tuple[str, ...]]:
    terms: dict[str, list[str]] = {label: list(seed_terms) for label, seed_terms in SEED_TERMS.items()}
    config = load_weak_supervision_config(config_path)
    dictionary = config.get("dictionary", {})
    if not isinstance(dictionary, dict):
        return {
            label: _unique_terms(values)
            for label, values in terms.items()
            if label in DEFAULT_LABELS
        }

    for group_name, labels in CONFIG_TERM_GROUPS.items():
        raw_terms = dictionary.get(group_name, [])
        if not isinstance(raw_terms, list):
            continue
        for label in labels:
            terms.setdefault(label, []).extend(str(term) for term in raw_terms)

    return {
        label: _unique_terms(values)
        for label, values in terms.items()
        if label in DEFAULT_LABELS
    }


@dataclass
class ChunkDictionaryLF(ChunkLabelingFunction):
    """Character-based multi-label dictionary LF for chunk text."""

    config_path: str | Path | None = None
    name: str = "lv1_chunk_dictionary"
    min_confidence: float = 0.58
    terms_by_label: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.terms_by_label:
            self.terms_by_label = _load_terms(self.config_path)

    def apply_all(self, chunk: dict[str, Any], labels: list[str]) -> dict[str, LFOutput]:
        text = chunk_text(chunk)
        outputs: dict[str, LFOutput] = {}
        if not text:
            return {label: abstain(self.name, label, "missing_text") for label in labels}

        for label in labels:
            matched_terms: list[str] = []
            evidence_groups: list[list[EvidenceSpan]] = []
            for term in self.terms_by_label.get(label, ()):
                spans = find_literal_spans(text, term, max_spans=4)
                if not spans:
                    continue
                matched_terms.append(term)
                evidence_groups.append(spans)

            evidence = merge_evidence(*evidence_groups)
            if not evidence:
                outputs[label] = abstain(self.name, label)
                continue

            confidence = min(0.92, self.min_confidence + 0.08 * min(len(matched_terms), 4))
            outputs[label] = make_output(
                self.name,
                label,
                vote=1,
                confidence=confidence,
                evidence=evidence,
                count=len(evidence),
                metadata={
                    "matched_terms": matched_terms,
                    "match_mode": "literal_character",
                },
            )
        return outputs

    def apply(self, chunk: dict[str, Any], label: str) -> LFOutput:
        """Return dictionary vote for one label."""

        return self.apply_all(chunk, [label])[label]
