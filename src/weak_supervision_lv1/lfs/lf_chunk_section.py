"""Chunk section-prior LF for Lv1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.weak_supervision.common.lf_output import LFOutput
from src.weak_supervision_lv1.chunk_lf_base import ChunkLabelingFunction
from src.weak_supervision_lv1.lfs._chunk_lf_utils import (
    abstain,
    chunk_text,
    find_literal_spans,
    make_output,
    section_surface,
)


SECTION_PRIORS: dict[str, tuple[tuple[str, float], ...]] = {
    "sub_diseases": (
        ("诊断", 0.42),
        ("分型", 0.44),
        ("分类", 0.42),
        ("临床类型", 0.44),
        ("诊疗要点", 0.38),
    ),
    "symptoms": (
        ("临床表现", 0.45),
        ("症状", 0.45),
        ("体征", 0.45),
        ("表现", 0.40),
        ("受累", 0.38),
    ),
    "tests": (
        ("检查", 0.45),
        ("辅助检查", 0.45),
        ("实验室", 0.45),
        ("指标", 0.42),
        ("评分", 0.40),
        ("诊断标准", 0.42),
        ("分类标准", 0.42),
    ),
    "treatments": (
        ("治疗原则", 0.45),
        ("治疗", 0.38),
        ("管理", 0.36),
        ("推荐意见", 0.40),
    ),
    "plans": (
        ("治疗方案", 0.45),
        ("治疗", 0.38),
        ("用药", 0.42),
        ("随访", 0.40),
        ("管理", 0.36),
    ),
}

GENERIC_SECTION_TERMS = {"正文", "前言", "摘要", "关键词", "引言", "概述", "背景"}


@dataclass
class ChunkSectionPriorLF(ChunkLabelingFunction):
    """Low-weight context prior from chunk section title/path."""

    name: str = "lv1_chunk_section_prior"
    priors: dict[str, tuple[tuple[str, float], ...]] = field(default_factory=lambda: SECTION_PRIORS)

    def apply_all(self, chunk: dict[str, Any], labels: list[str]) -> dict[str, LFOutput]:
        surface = section_surface(chunk)
        if not surface or surface.strip() in GENERIC_SECTION_TERMS:
            return {label: abstain(self.name, label, "missing_or_generic_section") for label in labels}

        text = chunk_text(chunk)
        outputs: dict[str, LFOutput] = {}
        for label in labels:
            matches = [
                (term, confidence)
                for term, confidence in self.priors.get(label, ())
                if term and term in surface
            ]
            if not matches:
                outputs[label] = abstain(self.name, label)
                continue

            best_confidence = max(confidence for _, confidence in matches)
            terms = [term for term, _ in matches]
            evidence = []
            for term in terms:
                evidence.extend(find_literal_spans(text, term, max_spans=2))
            outputs[label] = make_output(
                self.name,
                label,
                vote=1,
                confidence=best_confidence,
                evidence=evidence,
                count=max(1, len(terms)),
                metadata={
                    "section_surface": surface,
                    "matched_terms": terms,
                    "source": "section_title_or_path",
                    "role": "auxiliary_context_prior",
                },
            )
        return outputs

    def apply(self, chunk: dict[str, Any], label: str) -> LFOutput:
        """Return the section-prior vote for one label."""

        return self.apply_all(chunk, [label])[label]
