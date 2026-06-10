"""Chunk regex indicator LF for Lv1."""

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


TEST_PATTERNS: tuple[tuple[str, str, float], ...] = (
    (
        "antibody_status",
        r"(抗[\w\u4e00-\u9fffα-ωΑ-ΩⅠ-Ⅻ/-]{1,24}抗体|ANA|aCL|LAC|抗SSA|抗SSB)"
        r"[^\n。；;，,]{0,18}(阳性|阴性|\+|-)",
        0.86,
    ),
    (
        "lab_value",
        r"(?i)\b(CRP|PCT|ESR|ANA|APTT|dRVVT|IgG|IgM|IgA)\b"
        r"[^\n。；;，,]{0,18}(≥|≤|>|<|=|升高|降低|阳性|阴性)?"
        r"\s*\d+(?::\d+)?(?:\.\d+)?\s*(?:mg/L|ng/ml|mm/h|g/L|U/ml|GPL|MPL|%)?",
        0.84,
    ),
    (
        "score_threshold",
        r"(Mayo|UCEIS|OSS|SLEDAI|ESSDAI|评分|指数|活动性指数)"
        r"[^\n。；;，,]{0,20}(≥|≤|>|<|=|达到|高于|低于)?\s*\d+(?:\.\d+)?",
        0.82,
    ),
    (
        "exam_item",
        r"(活检|造影|超声|CT|MRI|内镜|病理|组织学|血清学|实验室检查|影像学检查|针刺反应)",
        0.72,
    ),
    (
        "diagnostic_threshold",
        r"(至少|满足|符合)[^\n。；;]{0,18}([一二三四五六七八九十\d]+)项"
        r"[^\n。；;]{0,18}(诊断|分类标准|标准)",
        0.78,
    ),
)


@dataclass
class ChunkRegexIndicatorLF(ChunkLabelingFunction):
    """Auxiliary regex LF focused only on objective test/indicator evidence."""

    name: str = "lv1_chunk_regex_indicator"
    test_patterns: tuple[tuple[str, str, float], ...] = field(default_factory=lambda: TEST_PATTERNS)

    def apply_all(self, chunk: dict[str, Any], labels: list[str]) -> dict[str, LFOutput]:
        text = chunk_text(chunk)
        if not text:
            return {label: abstain(self.name, label, "missing_text") for label in labels}

        outputs: dict[str, LFOutput] = {}
        for label in labels:
            patterns = self._patterns_for_label(label)
            evidence_groups: list[list[EvidenceSpan]] = []
            matched_patterns: list[str] = []
            confidences: list[float] = []
            for pattern_name, pattern, confidence in patterns:
                spans = find_regex_spans(text, pattern, source=pattern_name, max_spans=6)
                if not spans:
                    continue
                matched_patterns.append(pattern_name)
                confidences.append(confidence)
                evidence_groups.append(spans)

            evidence = merge_evidence(*evidence_groups)
            if not evidence:
                outputs[label] = abstain(self.name, label)
                continue

            outputs[label] = make_output(
                self.name,
                label,
                vote=1,
                confidence=min(max(confidences), 0.64),
                evidence=evidence,
                count=len(evidence),
                metadata={
                    "matched_patterns": matched_patterns,
                    "primary_label": label == "tests",
                    "role": "auxiliary_test_regex",
                },
            )
        return outputs

    def _patterns_for_label(self, label: str) -> tuple[tuple[str, str, float], ...]:
        if label == "tests":
            return self.test_patterns
        return ()

    def apply(self, chunk: dict[str, Any], label: str) -> LFOutput:
        """Return regex vote for one label."""

        return self.apply_all(chunk, [label])[label]
