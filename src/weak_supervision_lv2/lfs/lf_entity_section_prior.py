"""Section prior Lv2 labeling function."""

from __future__ import annotations

from typing import Any

from src.weak_supervision.common.lf_output import LFOutput
from src.weak_supervision.common.labels import ABSTAIN
from src.weak_supervision_lv2.entity_lf_base import EntityLabelingFunction


SECTION_PRIORS: dict[str, tuple[str, ...]] = {
    "symptoms": ("临床表现", "症状", "体征"),
    "tests": ("检查", "诊断", "实验室", "辅助检查"),
    "treatments": ("治疗", "处理", "管理"),
    "plans": ("治疗", "方案", "用药"),
    "etiologies": ("病因", "危险因素", "流行病学"),
    "pathogeneses": ("发病机制", "病理", "机制"),
    "sub_diseases": ("分型", "诊断", "分类"),
}


class EntitySectionPriorLF(EntityLabelingFunction):
    """Low-weight vote from section path/title."""

    name = "lv2_entity_section_prior"

    def apply(self, entity: dict[str, Any], chunk: dict[str, Any], label: str) -> LFOutput:
        title = str(chunk.get("section_title") or entity.get("section_title") or "")
        path = chunk.get("section_path") or entity.get("section_path") or []
        path_text = " ".join(str(item) for item in path) if isinstance(path, list) else str(path)
        text = f"{path_text} {title}"
        matches = [term for term in SECTION_PRIORS.get(label, ()) if term in text]
        if not matches:
            return LFOutput(self.name, label, vote=ABSTAIN, metadata={"entity_id": entity.get("entity_id")})
        return LFOutput(
            self.name,
            label,
            vote=1,
            confidence=0.42,
            count=len(matches),
            metadata={"entity_id": entity.get("entity_id"), "matched_sections": matches[:5]},
        )
