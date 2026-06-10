"""Tests for Lv2 entity-level Labeling Functions."""

from typing import Any

from src.weak_supervision.common.llm_prompt_registry import LV2_ENTITY_PROMPT_SPECS
from src.weak_supervision_lv2.entity_signal_builder import default_entity_lfs
from src.weak_supervision_lv2.lfs.lf_entity_context_window import EntityContextWindowLF
from src.weak_supervision_lv2.lfs.lf_entity_prompted_llm import EntityPromptedLLMLF
from src.weak_supervision_lv2.lfs.lf_entity_surface_pattern import EntitySurfacePatternLF


def test_default_lv2_lfs_include_multiple_prompted_llm_lfs() -> None:
    llm_lfs = [lf for lf in default_entity_lfs() if isinstance(lf, EntityPromptedLLMLF)]

    assert [lf.prompt_spec.name for lf in llm_lfs] == [spec.name for spec in LV2_ENTITY_PROMPT_SPECS]
    assert [lf.name for lf in llm_lfs] == [
        "lv2_entity_prompted_llm_type_boundary",
        "lv2_entity_prompted_llm_evidence_support",
        "lv2_entity_prompted_llm_schema_contrast",
    ]


def test_lv2_prompted_llm_lf_parses_type_votes_and_prompt_trace() -> None:
    def fake_llm(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": "fake",
            "text": """
            {
              "tests": {
                "present": true,
                "confidence": 0.82,
                "evidence": "urine culture",
                "reason": "evidence supports test"
              },
              "symptoms": {
                "present": false,
                "confidence": 0.2,
                "evidence": "",
                "reason": "not symptom"
              }
            }
            """,
        }

    lf = EntityPromptedLLMLF(
        prompt_spec=LV2_ENTITY_PROMPT_SPECS[0],
        llm_func=fake_llm,
    )
    entity = {
        "entity_id": "E1",
        "entity_type": "tests",
        "name": "urine culture",
        "evidence_text": "urine culture",
    }
    chunk = {"chunk_id": "CH1", "text": "The guideline recommends urine culture."}

    outputs = lf.apply_all(entity, chunk, ["tests", "symptoms"])

    assert outputs["tests"].vote == 1
    assert outputs["tests"].confidence == 0.82
    assert outputs["tests"].metadata["prompt_name"] == "type_boundary"
    assert outputs["tests"].metadata["evidence_located"] is True
    assert outputs["symptoms"].vote == 0


def test_lv2_prompted_llm_batch_returns_outputs_by_entity_id() -> None:
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
                  "entity_id": "E1",
                  "labels": {
                    "tests": {
                      "present": true,
                      "confidence": 0.9,
                      "evidence": "urine culture",
                      "reason": "test evidence"
                    }
                  }
                },
                {
                  "entity_id": "E2",
                  "labels": {
                    "tests": {
                      "present": true,
                      "confidence": 0.8,
                      "evidence": "blood test",
                      "reason": "test evidence"
                    }
                  }
                }
              ]
            }
            """,
        }

    lf = EntityPromptedLLMLF(
        prompt_spec=LV2_ENTITY_PROMPT_SPECS[0],
        llm_func=fake_llm,
        batch_size=20,
    )
    pairs = [
        (
            {"entity_id": "E1", "entity_type": "tests", "name": "urine culture", "evidence_text": "urine culture"},
            {"chunk_id": "CH1", "text": "The guideline recommends urine culture."},
        ),
        (
            {"entity_id": "E2", "entity_type": "tests", "name": "blood test", "evidence_text": "blood test"},
            {"chunk_id": "CH2", "text": "The guideline recommends blood test."},
        ),
    ]

    outputs = lf.apply_batch(pairs, ["tests"])

    assert len(calls) == 1
    assert outputs["E1"]["tests"].vote == 1
    assert outputs["E2"]["tests"].vote == 1
    assert outputs["E1"]["tests"].metadata["evidence_located"] is True
    assert outputs["E2"]["tests"].metadata["evidence_located"] is True


def test_lv2_structural_rules_prioritize_trigger_factor_as_etiology() -> None:
    entity = {
        "entity_id": "E1",
        "entity_type": "etiologies",
        "name": "局部创伤",
        "evidence_text": "局部创伤、某些食物、疲劳、失眠、月经可能为触发因素。",
    }
    chunk = {
        "chunk_id": "CH1",
        "section_title": "临床表现",
        "text": "口腔溃疡是常见临床表现。局部创伤、某些食物、疲劳、失眠、月经可能为触发因素。",
    }

    surface_etiology = EntitySurfacePatternLF().apply(entity, chunk, "etiologies")
    surface_symptom = EntitySurfacePatternLF().apply(entity, chunk, "symptoms")
    context_etiology = EntityContextWindowLF().apply(entity, chunk, "etiologies")
    context_symptom = EntityContextWindowLF().apply(entity, chunk, "symptoms")

    assert surface_etiology.vote == 1
    assert surface_etiology.confidence >= 0.86
    assert surface_symptom.vote == -1
    assert context_etiology.vote == 1
    assert context_symptom.vote == -1
