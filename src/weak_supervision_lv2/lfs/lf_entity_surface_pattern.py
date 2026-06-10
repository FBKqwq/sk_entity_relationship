"""Entity surface-form Lv2 labeling function."""

from __future__ import annotations

import re
from typing import Any

from src.weak_supervision.common.lf_output import EvidenceSpan, LFOutput
from src.weak_supervision.common.labels import ABSTAIN, NEGATIVE
from src.weak_supervision_lv2.entity_lf_base import EntityLabelingFunction


SURFACE_PATTERNS: dict[str, tuple[str, ...]] = {
    "tests": (r"检查", r"检测", r"培养", r"抗体", r"CRP", r"PCT", r"ESR", r"MRI", r"CT", r"PET"),
    "treatments": (r"治疗原则", r"治疗策略", r"处理原则", r"管理", r"推荐"),
    "plans": (r"mg", r"g/", r"每日", r"每[0-9一二三四五六七八九十]+", r"方案", r"手术", r"用药"),
    "etiologies": (r"病因", r"风险因素", r"危险因素", r"诱因", r"感染", r"病原体"),
    "pathogeneses": (r"机制", r"病理", r"生理", r"炎症", r"免疫", r"导致"),
    "symptoms": (r"症状", r"体征", r"发热", r"疼痛", r"皮疹", r"乏力", r"肿大"),
    "sub_diseases": (r"型$", r"期$", r"综合征", r"感染", r"炎", r"病", r"瘤"),
    "diseases": (r"指南", r"共识", r"诊疗规范", r"疾病"),
}

STRUCTURAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "etiologies": (
        r"(病因|诱因|触发因素|诱发因素|危险因素|风险因素|相关因素|病原体)",
        r"(导致|诱发|引起|增加.*风险|可能为.*因素|可为.*因素)",
    ),
    "plans": (
        r"(mg|g|ml|每日|每晚|每周|疗程|冲击治疗|序贯|逐渐减量|口服|注射|鞘内注射)",
        r"(甲泼尼龙|泼尼松|秋水仙碱|沙利度胺|环磷酰胺|硫唑嘌呤|环孢素|甲氨蝶呤|TNF|干扰素)",
    ),
    "treatments": (
        r"(治疗目标|治疗原则|治疗策略|基础治疗|一般治疗|局部治疗|全身药物治疗)",
        r"(控制|防止|预防|改善|推荐|建议).{0,20}(治疗|管理|处理)",
    ),
    "pathogeneses": (
        r"(发病机制|病理生理|机制|炎症级联|免疫反应|血管炎|结构改变|代谢改变)",
        r"(以.*为基础|由于.*引起|病理.*过程)",
    ),
}

NEGATIVE_STRUCTURAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "symptoms": (
        r"(病因|诱因|触发因素|诱发因素|危险因素|风险因素|病原体)",
        r"(治疗目标|治疗原则|治疗策略|mg|每日|疗程|冲击治疗|序贯|逐渐减量)",
    ),
    "diseases": (
        r"(病因未明|诱因|触发因素|危险因素|治疗目标|治疗原则|mg|每日|疗程)",
    ),
    "pathogeneses": (
        r"(病因未明|诱因|触发因素|危险因素|病原体)",
    ),
}


class EntitySurfacePatternLF(EntityLabelingFunction):
    """Vote from interpretable surface patterns in the entity name/evidence."""

    name = "lv2_entity_surface_pattern"

    def apply(self, entity: dict[str, Any], chunk: dict[str, Any], label: str) -> LFOutput:
        text = f"{entity.get('name') or ''} {entity.get('evidence_text') or ''}"
        negative_matched = [
            pattern for pattern in NEGATIVE_STRUCTURAL_PATTERNS.get(label, ()) if re.search(pattern, text, flags=re.I)
        ]
        if negative_matched:
            return LFOutput(
                self.name,
                label,
                vote=NEGATIVE,
                confidence=0.62,
                metadata={"entity_id": entity.get("entity_id"), "matched_negative_patterns": negative_matched[:5]},
            )
        patterns = SURFACE_PATTERNS.get(label, ())
        matched = [pattern for pattern in (*patterns, *STRUCTURAL_PATTERNS.get(label, ())) if re.search(pattern, text, flags=re.I)]
        if not matched:
            return LFOutput(self.name, label, vote=ABSTAIN, metadata={"entity_id": entity.get("entity_id")})
        evidence = [EvidenceSpan(0, len(str(entity.get("name") or "")), str(entity.get("name") or ""), "entity_name")]
        confidence = min(0.9, 0.55 + 0.05 * len(matched))
        if any(pattern in STRUCTURAL_PATTERNS.get(label, ()) for pattern in matched):
            confidence = max(confidence, 0.86)
        return LFOutput(
            self.name,
            label,
            vote=1,
            confidence=confidence,
            count=len(matched),
            evidence=evidence,
            metadata={"entity_id": entity.get("entity_id"), "matched_patterns": matched[:5]},
        )
