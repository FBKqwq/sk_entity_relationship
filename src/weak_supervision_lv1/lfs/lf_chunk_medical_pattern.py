"""Chunk medical semantic-pattern LF for Lv1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.weak_supervision.common.lf_output import EvidenceSpan, LFOutput
from src.weak_supervision_lv1.chunk_lf_base import ChunkLabelingFunction
from src.weak_supervision_lv1.lfs._chunk_lf_utils import (
    abstain,
    chunk_text,
    find_regex_spans,
    make_output,
    merge_evidence,
)


MEDICAL_PATTERNS: dict[str, tuple[tuple[str, str, float], ...]] = {
    "sub_diseases": (
        ("diagnosis_name", r"(诊断为|确诊为|疑诊|可诊断为|又称)[^\n。；;]{0,30}", 0.72),
        ("classification", r"(分为|分型为|分型|亚型|临床类型)[^\n。；;]{0,40}", 0.70),
        ("disease_abbreviation", r"[\u4e00-\u9fffA-Za-z]{2,20}[（(][A-Za-z]{2,12}[）)]", 0.66),
    ),
    "symptoms": (
        ("manifestation_context", r"(表现为|临床表现|症状|体征|可出现|常见|首发症状)[^\n。；;]{0,50}", 0.78),
        ("symptom_surface", r"[\u4e00-\u9fff]{0,8}(溃疡|疼痛|困难|皮疹|红斑|结节|瘢痕|出血|发热|乏力)[\u4e00-\u9fff]{0,8}", 0.72),
        ("organ_involvement", r"[\u4e00-\u9fff]{1,12}(受累|损害|病变)[^\n。；;]{0,30}", 0.70),
    ),
    "tests": (
        ("test_context", r"(检查|检测|试验|评分|指数|阳性|阴性|升高|降低)[^\n。；;]{0,40}", 0.74),
        ("pathology_or_imaging", r"(组织病理学|病理|活检|CT|MRI|超声|影像学|实验室)[^\n。；;]{0,40}", 0.74),
    ),
    "treatments": (
        ("treatment_principle", r"(治疗原则|治疗目标|推荐意见|应控制|应避免|首选|规范治疗)[^\n。；;]{0,50}", 0.76),
        ("management_principle", r"(管理原则|综合管理|个体化治疗|控制炎症|预防复发)[^\n。；;]{0,50}", 0.72),
    ),
    "plans": (
        ("treatment_plan", r"(治疗方案|给予|使用|采用|联合|加用|减量|疗程|剂量)[^\n。；;]{0,60}", 0.76),
        ("follow_up_plan", r"(随访|复查|监测|每\d+[周月年]|间隔\d+[周月年])[^\n。；;]{0,50}", 0.72),
    ),
}


@dataclass
class ChunkMedicalPatternLF(ChunkLabelingFunction):
    """Multi-label LF based on local medical semantic patterns in chunk text."""

    name: str = "lv1_chunk_medical_pattern"
    patterns: dict[str, tuple[tuple[str, str, float], ...]] = field(
        default_factory=lambda: MEDICAL_PATTERNS
    )

    def apply_all(self, chunk: dict[str, Any], labels: list[str]) -> dict[str, LFOutput]:
        text = chunk_text(chunk)
        if not text:
            return {label: abstain(self.name, label, "missing_text") for label in labels}

        outputs: dict[str, LFOutput] = {}
        for label in labels:
            evidence_groups: list[list[EvidenceSpan]] = []
            matched_patterns: list[str] = []
            confidences: list[float] = []
            for pattern_name, pattern, confidence in self.patterns.get(label, ()):
                spans = find_regex_spans(text, pattern, max_spans=6, source=pattern_name)
                if not spans:
                    continue
                evidence_groups.append(spans)
                matched_patterns.append(pattern_name)
                confidences.append(confidence)

            evidence = merge_evidence(*evidence_groups)
            if not evidence:
                outputs[label] = abstain(self.name, label)
                continue

            outputs[label] = make_output(
                self.name,
                label,
                vote=1,
                confidence=max(confidences),
                evidence=evidence,
                count=len(evidence),
                metadata={
                    "matched_patterns": matched_patterns,
                    "match_mode": "semantic_context_regex",
                    "role": "primary_multilabel_semantic_lf",
                },
            )
        return outputs

    def apply(self, chunk: dict[str, Any], label: str) -> LFOutput:
        """Return semantic-pattern vote for one label."""

        return self.apply_all(chunk, [label])[label]
