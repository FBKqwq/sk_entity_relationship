"""Tests for schema-constrained entity extraction prompts."""

from src.entity_extraction.core_disease import (
    build_core_disease_entity,
    core_disease_anchors,
    infer_core_disease_name,
    is_core_disease_child_name,
)
from src.entity_extraction.llm_entity_extractor import (
    build_prelabel_prompt,
    enforce_core_disease_subtype_boundary,
    enforce_lv1_label_overrides,
    extract_prelabeled_entities,
    parse_teacher_prelabel_response,
    select_active_entity_prompts,
)


def test_select_active_entity_prompts_extracts_only_present_labels() -> None:
    lv1_results = [
        {"label": "sub_diseases", "present": True, "status": "accepted", "predicted_count": 1},
        {"label": "symptoms", "present": False, "status": "rejected", "predicted_count": 0},
    ]

    selected = select_active_entity_prompts(lv1_results)

    assert selected["sub_diseases"]["mode"] == "extract"
    assert selected["symptoms"]["mode"] == "check_only"
    assert selected["tests"]["mode"] == "check_only"


def test_select_active_entity_prompts_full_extraction_extracts_all_active_labels() -> None:
    selected = select_active_entity_prompts(
        [{"label": "tests", "present": False, "status": "rejected"}],
        full_extraction=True,
    )

    assert selected
    assert all(spec["mode"] == "extract" for spec in selected.values())
    assert all(spec["status"] == "full_extraction" for spec in selected.values())


def test_build_prelabel_prompt_is_schema_driven_and_excludes_old_test_fields() -> None:
    chunk = {
        "chunk_id": "CH1",
        "text": "尿培养提示菌落计数升高。",
        "section_title": "检查",
        "document_core_disease": "急性胰腺炎",
    }
    lv1_results = [
        {
            "label": "tests",
            "present": True,
            "status": "accepted",
            "predicted_count": 1,
            "confidence": 0.8,
            "evidence_texts": ["尿培养"],
        },
        {"label": "plans", "present": False, "status": "rejected", "predicted_count": 0},
    ]

    prompt = build_prelabel_prompt(chunk, lv1_results)

    assert "entity_type: tests" in prompt
    assert "mode: extract" in prompt
    assert "entity_type: plans" in prompt
    assert "mode: check_only" in prompt
    assert "tests_list" in prompt
    assert "allowed_properties" in prompt
    assert "normal_range_min" in prompt
    assert "normal_range_max" in prompt
    assert "exam_name" not in prompt
    assert "exam_department" not in prompt
    assert "不允许根据 PDF 标题、文件名、目录标题或外部常识补造实体" in prompt
    assert "entity_type: diseases" in prompt
    assert "Disease 必须来自 chunk 原文" in prompt
    assert "例如“复杂性尿路感染”应是 sub_diseases，不应放入 symptoms" in prompt
    assert "evidence_rule" in prompt
    assert "relation_handoff" in prompt
    assert "阈值、单位、比较符、阳性/阴性结果、诊断角色" in prompt
    assert "etiologies 与 pathogeneses 必须区分" in prompt
    assert "检查名/指标名是 Test 实体" in prompt


def test_infer_core_disease_from_guideline_title() -> None:
    assert infer_core_disease_name("中国急性胰腺炎诊治指南（2021）") == "急性胰腺炎"
    assert infer_core_disease_name("肾综合征出血热防治专家共识") == "肾综合征出血热"
    assert infer_core_disease_name("抗磷脂综合征诊疗规范") == "抗磷脂综合征"


def test_build_core_disease_entity_uses_title_as_evidence() -> None:
    entity = build_core_disease_entity("中国急性胰腺炎诊治指南（2021）")

    assert entity is not None
    assert entity["entity_type"] == "diseases"
    assert entity["name"] == "急性胰腺炎"
    assert entity["properties"]["disease_name"] == "急性胰腺炎"
    assert entity["evidence"] == "中国急性胰腺炎诊治指南（2021）"


def test_core_disease_child_boundary_uses_core_anchors() -> None:
    assert core_disease_anchors("急性胰腺炎") == ("急性胰腺炎", "胰腺炎", "胰腺")
    assert is_core_disease_child_name("胆源性急性胰腺炎", "急性胰腺炎")
    assert is_core_disease_child_name("感染性胰腺坏死", "急性胰腺炎")
    assert not is_core_disease_child_name("急性胰腺炎", "急性胰腺炎")
    assert not is_core_disease_child_name("尿路感染", "急性胰腺炎", "急性胰腺炎合并尿路感染")


def test_enforce_core_disease_subtype_boundary_keeps_chunk_evidence_subtypes() -> None:
    response = {
        "sub_diseases_list": [
            {"name": "急性胰腺炎", "evidence": "急性胰腺炎", "confidence": 0.9},
            {"name": "胆源性急性胰腺炎", "evidence": "胆源性急性胰腺炎", "confidence": 0.9},
            {"name": "尿路感染", "evidence": "合并尿路感染", "confidence": 0.8},
        ],
        "symptoms_list": [],
        "tests_list": [],
        "treatments_list": [],
        "plans_list": [],
        "etiologies_list": [],
        "pathogeneses_list": [],
        "lv1_overrides": [],
        "schema_warnings": [],
    }

    fixed = enforce_core_disease_subtype_boundary(response, "急性胰腺炎")

    assert [item["name"] for item in fixed["sub_diseases_list"]] == [
        "急性胰腺炎",
        "胆源性急性胰腺炎",
        "尿路感染",
    ]
    assert fixed["symptoms_list"] == []
    assert ("sub_diseases", "尿路感染") in {
        (entity["entity_type"], entity["name"]) for entity in fixed["entities"]
    }


def test_parse_teacher_prelabel_response_filters_old_fields_and_empty_evidence() -> None:
    response = parse_teacher_prelabel_response(
        """
        {
          "tests_list": [
            {
              "name": "尿培养",
              "properties": {"test_name": "尿培养", "exam_name": "旧字段", "unit": "旧字段"},
              "evidence": "尿培养提示菌落计数升高",
              "confidence": 1.7,
              "override_lv1": false
            },
            {
              "name": "C反应蛋白",
              "properties": {"test_name": "C反应蛋白"},
              "evidence": "",
              "confidence": 0.9
            }
          ],
          "symptoms_list": [],
          "sub_diseases_list": [],
          "treatments_list": [],
          "plans_list": [],
          "etiologies_list": [],
          "pathogeneses_list": [],
          "lv1_overrides": []
        }
        """
    )

    assert len(response["entities"]) == 1
    entity = response["entities"][0]
    assert entity["entity_type"] == "tests"
    assert entity["confidence"] == 1.0
    assert entity["properties"]["test_name"] == "尿培养"
    assert "exam_name" not in entity["properties"]
    assert "unit" not in entity["properties"]
    assert any("removed unsupported properties" in warning for warning in response["schema_warnings"])
    assert any("empty evidence" in warning for warning in response["schema_warnings"])


def test_extract_prelabeled_entities_uses_injected_llm() -> None:
    def fake_llm(*args, **kwargs):
        return {
            "status": "ok",
            "model": "fake",
            "text": '{"tests_list":[{"name":"C反应蛋白","evidence":"C反应蛋白升高","confidence":0.8}],"sub_diseases_list":[],"symptoms_list":[],"treatments_list":[],"plans_list":[],"etiologies_list":[],"pathogeneses_list":[],"lv1_overrides":[]}',
        }

    chunk = {"chunk_id": "CH1", "text": "C反应蛋白升高。"}
    lv1_results = [{"label": "tests", "present": True, "status": "accepted", "predicted_count": 1}]

    response = extract_prelabeled_entities(chunk, lv1_results, llm_func=fake_llm)

    assert response["status"] == "ok"
    assert response["tests_list"][0]["name"] == "C反应蛋白"
    assert response["entities"][0]["properties"]["test_name"] == "C反应蛋白"
    assert "test_id" in response["entities"][0]["properties"]


def test_extract_prelabeled_entities_full_extraction_does_not_mark_lv1_override() -> None:
    def fake_llm(*args, **kwargs):
        return {
            "status": "ok",
            "model": "fake",
            "text": '{"tests_list":[{"name":"C反应蛋白","evidence":"C反应蛋白升高","confidence":0.8,"override_lv1":true}],"sub_diseases_list":[],"symptoms_list":[],"treatments_list":[],"plans_list":[],"etiologies_list":[],"pathogeneses_list":[],"lv1_overrides":[{"bad":"llm"}]}',
        }

    response = extract_prelabeled_entities(
        {"chunk_id": "CH1", "text": "C反应蛋白升高。"},
        [{"label": "tests", "present": False, "status": "rejected"}],
        llm_func=fake_llm,
        full_extraction=True,
    )

    assert response["Full_extraction"] is True
    assert response["entities"][0]["override_lv1"] is False
    assert response["lv1_overrides"] == []


def test_parse_teacher_prelabel_response_supports_etiology_and_pathogenesis() -> None:
    response = parse_teacher_prelabel_response(
        """
        {
          "etiologies_list": [{"name": "胆囊结石", "evidence": "胆囊结石是重要病因", "confidence": 0.8}],
          "pathogeneses_list": [{"name": "炎症级联反应", "evidence": "炎症级联反应导致组织损伤", "confidence": 0.7}],
          "sub_diseases_list": [],
          "symptoms_list": [],
          "tests_list": [],
          "treatments_list": [],
          "plans_list": [],
          "lv1_overrides": []
        }
        """
    )

    assert response["entities"][0]["entity_type"] == "etiologies"
    assert response["entities"][0]["properties"]["etiology_content"] == "胆囊结石"
    assert response["entities"][1]["entity_type"] == "pathogeneses"
    assert response["entities"][1]["properties"]["pathogenesis_content"] == "炎症级联反应"


def test_enforce_lv1_label_overrides_recomputes_llm_flags() -> None:
    response = {
        "tests_list": [{"name": "针刺反应", "evidence": "针刺反应阳性", "override_lv1": True, "confidence": 0.8}],
        "plans_list": [{"name": "给予激素", "evidence": "给予激素治疗", "override_lv1": False, "confidence": 0.8}],
        "sub_diseases_list": [],
        "symptoms_list": [],
        "treatments_list": [],
        "etiologies_list": [],
        "pathogeneses_list": [],
        "lv1_overrides": [{"bad": "llm_supplied"}],
        "schema_warnings": [],
    }
    lv1_results = [
        {"label": "tests", "present": True, "status": "accepted"},
        {"label": "plans", "present": False, "status": "rejected"},
    ]

    fixed = enforce_lv1_label_overrides(response, lv1_results)

    assert fixed["tests_list"][0]["override_lv1"] is False
    assert fixed["plans_list"][0]["override_lv1"] is True
    assert fixed["lv1_overrides"] == [
        {
            "entity_type": "plans",
            "name": "给予激素",
            "reason": "teacher_extracted_despite_lv1_negative_label",
        }
    ]
