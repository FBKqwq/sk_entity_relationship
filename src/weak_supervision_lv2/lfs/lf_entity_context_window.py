"""Context-window Lv2 labeling function."""

from __future__ import annotations

import re
from typing import Any

from src.weak_supervision.common.lf_output import EvidenceSpan, LFOutput
from src.weak_supervision.common.labels import ABSTAIN, NEGATIVE
from src.weak_supervision_lv2.entity_lf_base import EntityLabelingFunction


CONTEXT_TERMS: dict[str, tuple[str, ...]] = {
    "symptoms": ("症状", "体征", "表现", "临床"),
    "tests": ("检查", "检测", "诊断", "指标", "培养"),
    "treatments": ("治疗", "原则", "推荐", "处理"),
    "plans": ("方案", "剂量", "用法", "疗程", "手术"),
    "etiologies": ("病因", "风险", "危险因素", "诱发", "病原"),
    "pathogeneses": ("机制", "病理", "生理", "免疫", "炎症"),
    "sub_diseases": ("诊断", "分型", "亚型", "类型"),
    "diseases": ("指南", "共识", "规范", "疾病"),
}

STRUCTURAL_CONTEXT_PATTERNS: dict[str, tuple[str, ...]] = {
    "etiologies": (
        r"(可能为|可为|属于|是).{0,20}(触发因素|诱发因素|诱因|危险因素|风险因素)",
        r"(病因|诱因|触发因素|诱发因素|危险因素|风险因素|病原体)",
    ),
    "plans": (
        r"(mg|每日|每晚|疗程|冲击治疗|序贯|逐渐减量|口服|注射|鞘内注射)",
        r"(秋水仙碱|沙利度胺|甲泼尼龙|泼尼松|环磷酰胺|硫唑嘌呤|甲氨蝶呤)",
    ),
    "treatments": (
        r"(治疗目标|治疗原则|治疗策略|基础治疗|一般治疗|局部治疗|全身药物治疗)",
    ),
    "pathogeneses": (
        r"(发病机制|病理生理|机制|血管炎|免疫反应|炎症)",
    ),
}

NEGATIVE_CONTEXT_PATTERNS: dict[str, tuple[str, ...]] = {
    "symptoms": (
        r"(触发因素|诱发因素|诱因|危险因素|风险因素)",
        r"(治疗目标|治疗原则|治疗策略|mg|每日|疗程|冲击治疗)",
    ),
}


class EntityContextWindowLF(EntityLabelingFunction):
    """Vote from the chunk text around the entity/evidence."""

    name = "lv2_entity_context_window"

    def apply(self, entity: dict[str, Any], chunk: dict[str, Any], label: str) -> LFOutput:
        entity_id = entity.get("entity_id")
        chunk_text = str(chunk.get("text") or "")
        evidence_text = str(entity.get("evidence_text") or entity.get("name") or "").strip()
        if evidence_text and evidence_text in chunk_text:
            index = chunk_text.find(evidence_text)
            window = chunk_text[max(0, index - 80) : index + len(evidence_text) + 80]
        else:
            window = chunk_text[:200]
        negative_patterns = [
            pattern for pattern in NEGATIVE_CONTEXT_PATTERNS.get(label, ()) if re.search(pattern, window, flags=re.I)
        ]
        if negative_patterns:
            return LFOutput(
                self.name,
                label,
                vote=NEGATIVE,
                confidence=0.58,
                metadata={"entity_id": entity_id, "matched_negative_patterns": negative_patterns[:5]},
            )
        structural_patterns = [
            pattern for pattern in STRUCTURAL_CONTEXT_PATTERNS.get(label, ()) if re.search(pattern, window, flags=re.I)
        ]
        terms = [term for term in CONTEXT_TERMS.get(label, ()) if term in window]
        if not terms and not structural_patterns:
            return LFOutput(self.name, label, vote=ABSTAIN, metadata={"entity_id": entity_id})
        confidence = min(0.78, 0.50 + 0.06 * len(terms))
        if structural_patterns:
            confidence = max(confidence, min(0.88, 0.70 + 0.06 * len(structural_patterns)))
        return LFOutput(
            self.name,
            label,
            vote=1,
            confidence=confidence,
            count=len(terms) + len(structural_patterns),
            evidence=[EvidenceSpan(0, min(len(window), 160), window[:160], "context_window")],
            metadata={"entity_id": entity_id, "matched_terms": terms[:5], "matched_structural_patterns": structural_patterns[:5]},
        )
