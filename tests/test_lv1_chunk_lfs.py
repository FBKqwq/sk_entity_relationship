"""Tests for Lv1 chunk-level Labeling Functions."""

import importlib.util
from pathlib import Path
from typing import Any

from src.weak_supervision.common.labels import ABSTAIN
from src.weak_supervision_lv1.chunk_lf_applier import apply_chunk_lfs
from src.weak_supervision_lv1.lfs import (
    ChunkDictionaryLF,
    ChunkMedicalPatternLF,
    ChunkPromptedLLMLF,
    ChunkRegexIndicatorLF,
    ChunkSectionPriorLF,
)


LABELS = ["sub_diseases", "symptoms", "tests", "treatments", "plans"]


def test_section_prior_is_low_weight_context_signal() -> None:
    chunk = {
        "chunk_id": "c1",
        "section_title": "诊断与治疗",
        "section_path": ["正文", "诊断与治疗"],
        "text": "诊断与治疗\n应根据检查指标确诊，并遵循治疗原则。",
    }

    outputs = ChunkSectionPriorLF().apply_all(chunk, LABELS)

    assert outputs["sub_diseases"].vote == 1
    assert outputs["treatments"].vote == 1
    assert outputs["plans"].vote == 1
    assert outputs["treatments"].confidence <= 0.45
    assert outputs["treatments"].metadata["role"] == "auxiliary_context_prior"
    assert outputs["symptoms"].vote == ABSTAIN


def test_dictionary_lf_matches_all_configured_entity_types() -> None:
    chunk = {
        "chunk_id": "c2",
        "text": "原发性干燥综合征患者可出现口干、眼干，检测抗SSA抗体，并采用药物治疗。",
    }
    lf = ChunkDictionaryLF(
        terms_by_label={
            "sub_diseases": ("原发性干燥综合征",),
            "symptoms": ("口干", "眼干"),
            "tests": ("抗SSA抗体",),
            "plans": ("药物治疗",),
        }
    )

    outputs = lf.apply_all(chunk, ["sub_diseases", "symptoms", "tests", "plans"])

    assert outputs["sub_diseases"].vote == 1
    assert outputs["symptoms"].count == 2
    assert outputs["tests"].evidence[0].text == "抗SSA抗体"
    assert outputs["plans"].vote == 1


def test_regex_indicator_lf_only_votes_for_tests() -> None:
    chunk = {
        "chunk_id": "c3",
        "text": "抗SSA抗体阳性，ANA=1:320，Mayo评分≥3。建议每1~6个月随访1次。",
    }

    outputs = ChunkRegexIndicatorLF().apply_all(chunk, ["tests", "plans", "symptoms"])

    assert outputs["tests"].vote == 1
    assert outputs["tests"].count >= 3
    assert outputs["tests"].confidence <= 0.64
    assert outputs["tests"].metadata["role"] == "auxiliary_test_regex"
    assert outputs["plans"].vote == ABSTAIN
    assert outputs["symptoms"].vote == ABSTAIN


def test_regex_indicator_lf_ignores_duration_and_rate_values() -> None:
    chunk = {
        "chunk_id": "c3b",
        "text": "发生率95%以上，持续2周或6周，51.7%~93%，症状反复发作。",
    }

    outputs = ChunkRegexIndicatorLF().apply_all(chunk, ["tests"])

    assert outputs["tests"].vote == ABSTAIN


def test_medical_pattern_lf_is_multilabel_primary_signal() -> None:
    chunk = {
        "chunk_id": "c3c",
        "text": (
            "白塞综合征又称贝赫切特病。患者可出现口腔溃疡、疼痛和皮疹。"
            "针刺反应阳性，组织病理学检查提示血管炎。治疗原则为控制炎症，"
            "可给予糖皮质激素并定期随访。"
        ),
    }

    outputs = ChunkMedicalPatternLF().apply_all(chunk, LABELS)

    assert outputs["sub_diseases"].vote == 1
    assert outputs["symptoms"].vote == 1
    assert outputs["tests"].vote == 1
    assert outputs["treatments"].vote == 1
    assert outputs["plans"].vote == 1
    assert outputs["symptoms"].metadata["role"] == "primary_multilabel_semantic_lf"


def test_prompted_llm_lf_parses_multilabel_json_without_requiring_evidence_spans() -> None:
    def fake_llm(*args, **kwargs):
        return {
            "status": "ok",
            "model": "fake",
            "text": """
            {
              "sub_diseases": {
                "present": true,
                "count": 1,
                "confidence": 0.9,
                "evidence": ["干燥综合征"],
                "reason": "出现确诊名"
              },
              "symptoms": {
                "present": true,
                "count": 2,
                "confidence": 0.8,
                "evidence": ["口干", "眼干"],
                "reason": "出现症状"
              },
              "tests": {
                "present": true,
                "count": 1,
                "confidence": 0.7,
                "evidence": ["不存在的证据"],
                "reason": "证据无法定位"
              }
            }
            """,
        }

    chunk = {"chunk_id": "c4", "text": "干燥综合征患者常有口干、眼干。"}
    lf = ChunkPromptedLLMLF(enabled=True, llm_func=fake_llm)

    outputs = lf.apply_all(chunk, ["sub_diseases", "symptoms", "tests"])

    assert outputs["sub_diseases"].vote == 1
    assert outputs["symptoms"].count == 2
    assert outputs["tests"].vote == 1
    assert outputs["tests"].evidence == []
    assert len(outputs["tests"].metadata["missing_evidence"]) == 1
    assert outputs["tests"].metadata["evidence_span_validated"] is False
    assert outputs["tests"].metadata["prompt_name"] == "general"


def test_lv1_builds_multiple_complementary_prompted_llm_lfs() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "02_snorkel_lv1_label_chunks.py"
    spec = importlib.util.spec_from_file_location("snorkel_lv1_script", script_path)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    lfs = script.build_default_lfs(
        weak_config_path=None,
        llm_config_path=None,
        llm_enabled=True,
        project_config={
            "lv1_prompted_llm": {
                "prompts": [
                    {"name": "semantic_presence", "focus": "semantic", "instruction": "recall"},
                    {"name": "evidence_anchor", "focus": "evidence", "instruction": "anchor"},
                ]
            }
        },
    )
    llm_lfs = [lf for lf in lfs if isinstance(lf, ChunkPromptedLLMLF)]

    assert [lf.name for lf in llm_lfs] == [
        "lv1_chunk_prompted_llm_semantic_presence",
        "lv1_chunk_prompted_llm_evidence_anchor",
    ]
    assert [lf.prompt_name for lf in llm_lfs] == ["semantic_presence", "evidence_anchor"]


def test_chunk_lf_applier_uses_multilabel_apply_all() -> None:
    chunk = {"chunk_id": "c5", "section_title": "临床表现", "text": "临床表现\n患者发热。"}
    outputs = apply_chunk_lfs([chunk], ["symptoms", "tests"], [ChunkSectionPriorLF()])

    assert [output.label for output in outputs] == ["symptoms", "tests"]
    assert outputs[0].vote == 1
    assert outputs[1].vote == ABSTAIN


def test_lv1_prompted_llm_batch_returns_outputs_by_chunk_id() -> None:
    calls: list[str] = []

    def fake_llm(prompt: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(prompt)
        return {
            "status": "ok",
            "model": "fake",
            "text": """
            {
              "results": [
                {
                  "chunk_id": "c6",
                  "labels": {
                    "symptoms": {
                      "present": true,
                      "count": 1,
                      "confidence": 0.9,
                      "evidence": ["发热"],
                      "reason": "出现症状"
                    }
                  }
                },
                {
                  "chunk_id": "c7",
                  "labels": {
                    "symptoms": {
                      "present": true,
                      "count": 1,
                      "confidence": 0.8,
                      "evidence": ["咳嗽"],
                      "reason": "出现症状"
                    }
                  }
                }
              ]
            }
            """,
        }

    chunks = [
        {"chunk_id": "c6", "text": "患者发热。"},
        {"chunk_id": "c7", "text": "患者咳嗽。"},
    ]
    lf = ChunkPromptedLLMLF(enabled=True, llm_func=fake_llm, batch_size=10)

    outputs = apply_chunk_lfs(chunks, ["symptoms"], [lf])

    assert len(calls) == 1
    assert [output.vote for output in outputs] == [1, 1]
    assert [output.evidence[0].text for output in outputs] == ["发热", "咳嗽"]
