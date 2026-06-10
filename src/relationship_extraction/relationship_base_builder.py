"""Build relationship_base records from entity_base rows and chunks."""

from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from src.utils.hashing import sha1_12
from src.utils.io import read_json, write_json, write_jsonl
from src.utils.llm_client import chat_completion_text
from src.weak_supervision.common.graph_schema import (
    GRAPH_ENTITIES,
    GRAPH_RELATIONS,
    RELATION_PROPERTY_MAP,
)

LLMCallable = Callable[..., dict[str, Any]]

CONTEXT_LEVELS = ("entity_window", "chunk", "neighbor_chunks")
DEFAULT_MAX_DOCUMENT_CHARS = 20000
DEFAULT_MAX_CROSS_CHUNK_DISTANCE = 6
DEFAULT_SAME_CHUNK_ONLY = False
DEFAULT_MAX_CANDIDATES = 0
DEFAULT_AUDIT_BATCH_SIZE = 20
DEFAULT_INCLUDE_REVIEW_ENTITIES = True
DEFAULT_COVERAGE_TARGETS = {
    "manifests_as": 3,
    "requires_test": 3,
    "causes": 1,
    "explained_by": 1,
    "follows_treatment": 1,
}
RELATION_MAX_CHUNK_DISTANCE = {
    "has_sub_disease": 0,
    "manifests_as": 2,
    "requires_test": 2,
    "follows_treatment": 1,
    "implements_by": 1,
    "causes": 2,
    "explained_by": 2,
}
RELATION_CROSS_CHUNK_TRIGGERS = {
    "manifests_as": ("表现", "症状", "体征", "临床", "可见", "出现", "伴有"),
    "requires_test": ("检查", "检测", "培养", "诊断", "指标", "阳性", "阴性", "升高", "降低"),
    "follows_treatment": ("治疗", "原则", "推荐", "处理", "管理", "应", "建议"),
    "implements_by": ("治疗", "方案", "药", "剂量", "疗程", "手术", "操作"),
    "causes": ("病因", "危险因素", "风险", "导致", "诱发", "相关", "易感", "病原"),
    "explained_by": ("机制", "病理", "生理", "由于", "引起", "过程", "解释"),
}
LLM_HEARTBEAT_SECONDS = 30.0
LLM_FINISH_LOG_SECONDS = 5.0
DETAIL_FIRST_N_CANDIDATES = 5
DETAIL_EVERY_N_CANDIDATES = 25


def _progress(message: str) -> None:
    print(f"[relationship] {message}", flush=True)


def _short_text(value: Any, max_chars: int = 24) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _doc_label(path: str | Path) -> str:
    name = Path(path).name
    for suffix in (".relationship_base.jsonl", ".entity_base.jsonl", ".chunk.json"):
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return Path(path).stem


def _candidate_label(candidate: "CandidateRelationship", index: int, total: int) -> str:
    return (
        f"candidate {index}/{total} "
        f"{candidate.relation_type} "
        f"level={candidate.search_level} distance={candidate.chunk_distance} "
        f"({_short_text(candidate.start_entity_name)} -> {_short_text(candidate.end_entity_name)})"
    )


def _should_detail_candidate(index: int, total: int) -> bool:
    return index <= DETAIL_FIRST_N_CANDIDATES or index % DETAIL_EVERY_N_CANDIDATES == 0 or index == total


def _call_with_heartbeat(func: Callable[[], dict[str, Any]], *, label: str) -> dict[str, Any]:
    done = threading.Event()

    def heartbeat() -> None:
        waited = 0
        while not done.wait(LLM_HEARTBEAT_SECONDS):
            waited += int(LLM_HEARTBEAT_SECONDS)
            _progress(f"{label} still waiting for LLM response... elapsed={waited}s")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        return func()
    finally:
        done.set()
        thread.join(timeout=1.0)
        elapsed = time.monotonic() - started
        if elapsed >= LLM_FINISH_LOG_SECONDS:
            _progress(f"{label} LLM call finished in {elapsed:.1f}s")


LABEL_TO_ENTITY_TYPE = {entity.label: entity.entity_type for entity in GRAPH_ENTITIES}
ENTITY_TYPE_TO_LABEL = {entity.entity_type: entity.label for entity in GRAPH_ENTITIES}
RELATION_SPECS = {relation.relation: relation for relation in GRAPH_RELATIONS}
LEGAL_RELATIONS = tuple(
    (
        LABEL_TO_ENTITY_TYPE[relation.source],
        relation.relation,
        LABEL_TO_ENTITY_TYPE[relation.target],
    )
    for relation in GRAPH_RELATIONS
)

RELATION_SYSTEM_PROMPT = """你是医学知识图谱关系抽取器。
只允许基于给定上下文判断候选关系是否被原文支持或抽取关系属性。
输出必须是严格 JSON，不要输出解释性正文。无法确认时返回 false 或空属性，不得臆造。
"""


RELATION_SEMANTIC_CONTRACTS: dict[str, dict[str, Any]] = {
    "has_sub_disease": {
        "definition": "Disease 包含或上位概括 Sub_disease。只有原文能支持某确诊名/分型属于某疾病大类时才成立。",
        "direction": "Disease -> Sub_disease",
        "positive": ["尿路感染 -> 复杂性尿路感染", "尿路感染 -> 非发热性尿路感染"],
        "negative": ["症状 -> 疾病", "治疗方案 -> 疾病", "两个并列疾病之间没有上位包含语义"],
        "reasoning": "用于从疾病大类推理到具体确诊名，再继续连接症状、检查、治疗、病因和机制。",
    },
    "manifests_as": {
        "definition": "Sub_disease 表现为 Symptom，表示症状/体征/临床表现支持该确诊名或常见于该确诊名。",
        "direction": "Sub_disease -> Symptom",
        "positive": ["复杂性尿路感染 -> 发热", "急性肾盂肾炎 -> 肋脊角压痛"],
        "negative": ["检查项目不是症状", "病因不是症状", "治疗方案不是症状"],
        "reasoning": "用于从表现推理可能诊断，或解释诊断的临床表现。",
    },
    "requires_test": {
        "definition": "Sub_disease 需要 Test 作为诊断、评估、分层、病原学确认或疗效监测依据。",
        "direction": "Sub_disease -> Test",
        "positive": ["复杂性尿路感染 -> 尿培养", "尿源性脓毒血症 -> 血培养"],
        "negative": ["治疗药物不是检查", "单纯并列提到但没有诊断/评估语义不成立"],
        "reasoning": "用于诊断标准、检查依据、阈值和结果推理。",
    },
    "follows_treatment": {
        "definition": "Sub_disease 遵循 Treatment 治疗原则、治疗目标、推荐策略或处理原则。",
        "direction": "Sub_disease -> Treatment",
        "positive": ["复杂性尿路感染 -> 根据尿培养和药敏结果选择敏感抗菌药物"],
        "negative": ["具体药物剂量应连到 Plan，不直接作为 Treatment 原则"],
        "reasoning": "用于从诊断推理治疗原则。",
    },
    "implements_by": {
        "definition": "Treatment 原则由 Plan 具体落实，Plan 是药物、剂量、操作、疗程、监测或随访等可执行方案。",
        "direction": "Treatment -> Plan",
        "positive": ["经验性抗菌药物治疗 -> 左氧氟沙星500mg每日1次"],
        "negative": ["两个原则之间不成立", "疾病到方案不使用 implements_by"],
        "reasoning": "用于从治疗原则推理到可执行方案。",
    },
    "causes": {
        "definition": "Sub_disease 由 Etiology 导致、诱发、增加风险或具有病因/危险因素关联。当前 schema 方向为 Sub_disease -> Etiology。",
        "direction": "Etiology -> Sub_disease",
        "positive": ["复杂性尿路感染 -> 糖尿病", "感染性结石 -> 产尿素酶细菌"],
        "negative": ["症状不是病因", "检查不是病因", "机制过程应使用 explained_by"],
        "reasoning": "用于疾病病因、风险因素和病原体推理。",
    },
    "explained_by": {
        "definition": "Sub_disease 可由 Pathogenesis 机制解释，机制必须是过程性病理生理说明。",
        "direction": "Sub_disease -> Pathogenesis",
        "positive": ["感染性结石 -> 产尿素酶细菌升高尿液pH促进结晶沉积"],
        "negative": ["单个病原体名称不是机制", "治疗方案不是机制"],
        "reasoning": "用于机制解释推理。",
    },
}

RELATION_SEMANTIC_CONTRACTS["causes"] = {
    "definition": "Etiology 导致、诱发、增加风险或与 Sub_disease 具有病因/危险因素关联；方向必须是病因指向确诊名/疾病分型。",
    "direction": "Etiology -> Sub_disease",
    "positive": ["糖尿病 -> 复杂性尿路感染", "产尿素酶细菌 -> 感染性结石"],
    "negative": ["症状不是病因", "检查不是病因", "机制过程应使用 explained_by", "不要输出 Sub_disease -> Etiology 反向关系"],
    "reasoning": "用于从病因、风险因素和病原体推理到可能的确诊名/疾病分型，并支持组合诊断规则表达。",
}

RELATION_PROPERTY_CONTRACTS: dict[str, dict[str, str]] = {
    "relation_id": {
        "meaning": "关系唯一标识；通常由后处理生成。",
        "fill_when": "保持模板中的 relation_id，不要改写。",
        "boundary": "不要填实体 ID、候选 ID 或关系名称。",
        "positive": "REL_xxx",
        "negative": "requires_test",
    },
    "relation_name": {
        "meaning": "关系中文名称。",
        "fill_when": "保持模板中的 relation_name，不要自行改名。",
        "boundary": "不要写证据文本或实体名称。",
        "positive": "需要检查",
        "negative": "尿培养",
    },
    "relation_type": {
        "meaning": "关系类型枚举。",
        "fill_when": "保持模板中的 relation_type。",
        "boundary": "不得输出 schema 外关系名。",
        "positive": "requires_test",
        "negative": "diagnosed_by",
    },
    "evidence_ids": {
        "meaning": "支撑关系或属性的证据 ID 列表。",
        "fill_when": "有上游证据 ID 时填写；没有则空数组，证据原文放 evidence。",
        "boundary": "不要把证据长文本写入 evidence_ids。",
        "positive": '["EV_001"]',
        "negative": '["尿培养阳性支持诊断"]',
    },
    "confidence": {
        "meaning": "关系或属性抽取置信度，0 到 1。",
        "fill_when": "根据原文明确程度填写。",
        "boundary": "不要写百分号字符串或超过 0-1 的值。",
        "positive": "0.82",
        "negative": "82%",
    },
    "diagnostic_role": {
        "meaning": "症状、检查或病因在诊断中的角色。",
        "fill_when": "原文说明必须、支持、排除、风险因素、金标准等语义时填写。",
        "boundary": "不要写检查名、阈值、治疗推荐。",
        "positive": "required",
        "negative": "尿培养",
    },
    "type_info": {
        "meaning": "关系子类型或临床类别。",
        "fill_when": "原文说明表现类型、检查类型、关系类别时填写。",
        "boundary": "不要重复实体名或关系名。",
        "positive": "病原学检查",
        "negative": "requires_test",
    },
    "weight": {
        "meaning": "诊断权重、重要性或证据权重。",
        "fill_when": "原文有主要、关键、辅助、必要、可选等权重语义时填写数值或短语。",
        "boundary": "不要写推荐强度或置信度。",
        "positive": "high",
        "negative": "0.8 confidence",
    },
    "typicality": {
        "meaning": "表现、检查或病因的典型性/常见程度。",
        "fill_when": "原文出现常见、典型、少见、罕见、高危等语义时填写。",
        "boundary": "不要写阳性/阴性或推荐强度。",
        "positive": "common",
        "negative": "positive",
    },
    "duration_condition": {
        "meaning": "症状持续时间、病程或时间条件。",
        "fill_when": "原文明确持续多少天、术后多久、发病后多久等条件时填写。",
        "boundary": "不要写疗程、随访周期或无关时间。",
        "positive": "术后2小时",
        "negative": "每日1次",
    },
    "section_priority": {
        "meaning": "证据所在章节优先级或诊断/治疗章节权重。",
        "fill_when": "需要区分诊断章节、治疗章节、推荐章节时填写。",
        "boundary": "不要写章节标题全文当作普通证据。",
        "positive": "diagnosis_section",
        "negative": "三、治疗 原文长段落",
    },
    "value_min": {
        "meaning": "检查阈值或诊断标准数值下限。",
        "fill_when": "原文给出 >=、>、范围下限等时填写数值。",
        "boundary": "不要写单位、比较符或完整阈值文本。",
        "positive": "100000",
        "negative": ">=10^5 cfu/ml",
    },
    "value_max": {
        "meaning": "检查阈值或诊断标准数值上限。",
        "fill_when": "原文给出 <=、<、范围上限等时填写数值。",
        "boundary": "不要写单位、比较符或完整阈值文本。",
        "positive": "38.0",
        "negative": "<38℃",
    },
    "operator": {
        "meaning": "阈值比较符。",
        "fill_when": "原文出现大于、小于、不低于、范围等比较条件时填写。",
        "boundary": "不要把数值或单位写进 operator。",
        "positive": ">=",
        "negative": "10^5",
    },
    "unit": {
        "meaning": "检查阈值或剂量相关单位。",
        "fill_when": "原文明确单位时填写。",
        "boundary": "不要包含比较符或数值。",
        "positive": "cfu/ml",
        "negative": ">=10^5 cfu/ml",
    },
    "result_text": {
        "meaning": "检查结果文本，如阳性、阴性、菌落计数升高。",
        "fill_when": "原文给出定性或描述性结果时填写。",
        "boundary": "不要写检查名称或完整诊断句。",
        "positive": "尿培养阳性",
        "negative": "尿培养",
    },
    "gold_standard": {
        "meaning": "是否金标准或确诊依据。",
        "fill_when": "原文明确金标准、确诊必须依赖、首选确诊等语义时 true；明确不是则 false。",
        "boundary": "不确定则 null，不要猜测。",
        "positive": "true",
        "negative": "supportive",
    },
    "criterion_group_id": {
        "meaning": "诊断标准组合组 ID，用于表示同组条件。",
        "fill_when": "原文有满足若干项、A+B、三项中两项等组合规则时填写稳定组名。",
        "boundary": "不要写检查名称或实体 ID。",
        "positive": "AP_3_choose_2",
        "negative": "尿培养",
    },
    "required_count": {
        "meaning": "组合诊断标准中需要满足的数量。",
        "fill_when": "原文出现三项中两项、至少两项等时填写数值。",
        "boundary": "不要写总项目数或百分比。",
        "positive": "2",
        "negative": "三项",
    },
    "polarity": {
        "meaning": "关系证据的阳性、阴性、存在、排除极性。",
        "fill_when": "原文明确阳性/阴性、有/无、排除/不支持时填写。",
        "boundary": "不要写典型性或推荐强度。",
        "positive": "positive",
        "negative": "common",
    },
    "evidence_level": {
        "meaning": "证据等级或推荐证据级别。",
        "fill_when": "原文明确 A/B/C、Ⅰ/Ⅱ、证据等级等时填写。",
        "boundary": "不要写置信度或推荐强度，除非原文把二者合并表达。",
        "positive": "A级证据",
        "negative": "0.9",
    },
    "clinical_stage": {
        "meaning": "关系适用的临床阶段或疾病时期。",
        "fill_when": "原文明确急性期、复发期、术后、重症阶段等时填写。",
        "boundary": "不要写治疗线或严重程度。",
        "positive": "术后",
        "negative": "second_line",
    },
    "treatment_line": {
        "meaning": "治疗线别或治疗顺序。",
        "fill_when": "原文明确一线、二线、后线、经验性、目标性等时填写。",
        "boundary": "不要写推荐强度、病情严重程度或剂量。",
        "positive": "first_line",
        "negative": "强推荐",
    },
    "applicable_condition": {
        "meaning": "关系适用条件。",
        "fill_when": "原文明确适用人群、病情、阶段、失败条件、检查结果条件时填写。",
        "boundary": "不要重复实体名称；不要猜测未说明条件。",
        "positive": "重症患者或初始经验性治疗失败患者",
        "negative": "复杂性尿路感染",
    },
    "recommendation_polarity": {
        "meaning": "推荐方向，如推荐、不推荐、可考虑、避免。",
        "fill_when": "原文明确推荐或禁止语义时填写。",
        "boundary": "不要写证据等级、治疗线或置信度。",
        "positive": "recommend",
        "negative": "A级证据",
    },
    "recommendation_strength": {
        "meaning": "推荐强度。",
        "fill_when": "原文明确强推荐、弱推荐、建议、可考虑等时填写。",
        "boundary": "不要写治疗线、适用条件或证据等级。",
        "positive": "strong",
        "negative": "first_line",
    },
    "decision_basis": {
        "meaning": "决策依据列表，如培养结果、药敏、肾功能、病情严重程度。",
        "fill_when": "原文说明治疗/检查选择依据时填写数组。",
        "boundary": "不要写完整治疗方案或证据长文本。",
        "positive": '["尿培养结果", "药敏试验结果"]',
        "negative": '["左氧氟沙星500mg每日1次"]',
    },
    "contraindication_note": {
        "meaning": "禁忌、慎用、剂量调整或注意事项。",
        "fill_when": "原文明确禁忌、不宜、慎用、肾功能调整等时填写。",
        "boundary": "不要写普通适用条件或方案名称。",
        "positive": "肾功能不全者根据肌酐清除率调整剂量",
        "negative": "每日1次",
    },
    "source_section": {
        "meaning": "关系或属性证据来源章节。",
        "fill_when": "上下文带有章节标题或原文明确章节时填写。",
        "boundary": "不要写整段证据。",
        "positive": "三、治疗",
        "negative": "尿培养提示...",
    },
    "plan_role": {
        "meaning": "Plan 在落实 Treatment 时的角色。",
        "fill_when": "原文能判断药物、减压、监测、预防、随访等执行角色时填写。",
        "boundary": "不要写治疗线或推荐强度。",
        "positive": "antibiotic",
        "negative": "first_line",
    },
    "causal_strength": {
        "meaning": "病因关系强度。",
        "fill_when": "原文明确主要病因、危险因素、相关、可能导致等强弱语义时填写。",
        "boundary": "不要写置信度或典型性。",
        "positive": "major_risk_factor",
        "negative": "0.8",
    },
}


@dataclass(frozen=True)
class CandidateRelationship:
    """One legal candidate relationship before LLM audit."""

    candidate_id: str
    document_id: str
    source_pdf: str
    chunk_id: str
    source_chunk_id: str
    target_chunk_id: str
    start_entity_id: str
    end_entity_id: str
    start_entity_type: str
    end_entity_type: str
    start_entity_name: str
    end_entity_name: str
    relation_type: str
    relation_name: str
    search_level: str
    chunk_distance: int

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "document_id": self.document_id,
            "source_pdf": self.source_pdf,
            "chunk_id": self.chunk_id,
            "source_chunk_id": self.source_chunk_id,
            "target_chunk_id": self.target_chunk_id,
            "start_entity_id": self.start_entity_id,
            "end_entity_id": self.end_entity_id,
            "start_entity_type": self.start_entity_type,
            "end_entity_type": self.end_entity_type,
            "start_entity_name": self.start_entity_name,
            "end_entity_name": self.end_entity_name,
            "relation_type": self.relation_type,
            "relation_name": self.relation_name,
            "search_level": self.search_level,
            "chunk_distance": self.chunk_distance,
        }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_jsonl(row: dict[str, Any], path: str | Path | None) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_chunks(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)], {}
    if not isinstance(payload, dict):
        raise ValueError(f"Chunk JSON must be an object or list: {path}")
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError(f"Chunk JSON field `chunks` must be a list: {path}")
    return [item for item in chunks if isinstance(item, dict)], payload


def _chunk_id(chunk: dict[str, Any]) -> str:
    return str(chunk.get("chunk_id") or chunk.get("id") or "")


def _document_id_for_entity(entity: dict[str, Any], fallback: str) -> str:
    return str(entity.get("document_id") or fallback or "")


def _source_pdf_for_entity(entity: dict[str, Any], fallback: str) -> str:
    return str(entity.get("source_pdf") or entity.get("pdf_path") or fallback or "")


def _entity_name(entity: dict[str, Any]) -> str:
    return str(entity.get("name") or entity.get("content") or "").strip()


def _entity_id(entity: dict[str, Any]) -> str:
    return str(entity.get("entity_id") or entity.get("id") or "").strip()


def _entity_evidence(entity: dict[str, Any]) -> str:
    return str(entity.get("evidence_text") or entity.get("evidence") or "").strip()


def _filter_final_entities(
    entities: list[dict[str, Any]],
    *,
    include_review_entities: bool,
) -> list[dict[str, Any]]:
    """Filter Lv2 entity_nodes while leaving legacy fixtures untouched."""

    has_lv2_status = any(
        str(entity.get("entity_status") or entity.get("status") or "")
        in {"accepted", "review", "rejected"}
        for entity in entities
    )
    if not has_lv2_status:
        return entities
    allowed = {"accepted", "review"} if include_review_entities else {"accepted"}
    return [
        entity
        for entity in entities
        if str(entity.get("entity_status") or entity.get("status") or "") in allowed
        or (
            not include_review_entities
            and str(entity.get("entity_type") or "") == "diseases"
            and str(entity.get("entity_status") or entity.get("status") or "") == "review"
        )
    ]


def _relation_name(relation_type: str) -> str:
    spec = RELATION_SPECS.get(relation_type)
    return spec.zh_name if spec else relation_type


def _candidate_id(
    document_id: str,
    start_entity_id: str,
    end_entity_id: str,
    relation_type: str,
) -> str:
    return f"REL_CAND_{sha1_12('|'.join([document_id, start_entity_id, relation_type, end_entity_id]))}"


def _relation_id(
    document_id: str,
    start_entity_id: str,
    end_entity_id: str,
    relation_type: str,
) -> str:
    return f"REL_{sha1_12('|'.join([document_id, start_entity_id, relation_type, end_entity_id]))}"


def _chunk_order(chunks: list[dict[str, Any]]) -> dict[str, int]:
    return {_chunk_id(chunk): index for index, chunk in enumerate(chunks)}


def _choose_candidate_chunk(source_chunk_id: str, target_chunk_id: str, order: dict[str, int]) -> str:
    if source_chunk_id == target_chunk_id:
        return source_chunk_id
    source_index = order.get(source_chunk_id, 10**9)
    target_index = order.get(target_chunk_id, 10**9)
    return source_chunk_id if source_index <= target_index else target_chunk_id


def _within_cross_chunk_limit(
    source_chunk_id: str,
    target_chunk_id: str,
    order: dict[str, int],
    max_distance: int,
) -> bool:
    if source_chunk_id == target_chunk_id:
        return True
    if source_chunk_id not in order or target_chunk_id not in order:
        return True
    return abs(order[source_chunk_id] - order[target_chunk_id]) <= max_distance


def _chunk_distance(source_chunk_id: str, target_chunk_id: str, order: dict[str, int]) -> int:
    if source_chunk_id == target_chunk_id:
        return 0
    if source_chunk_id not in order or target_chunk_id not in order:
        return DEFAULT_MAX_CROSS_CHUNK_DISTANCE + 1
    return abs(order[source_chunk_id] - order[target_chunk_id])


def _candidate_search_level(chunk_distance: int) -> str:
    if chunk_distance > DEFAULT_MAX_CROSS_CHUNK_DISTANCE:
        return "document"
    if chunk_distance == 0:
        return "same_chunk"
    if chunk_distance == 1:
        return "adjacent_chunk"
    return "near_chunk_window"


def _chunk_text_by_id(chunks: list[dict[str, Any]]) -> dict[str, str]:
    return {_chunk_id(chunk): str(chunk.get("text") or "") for chunk in chunks}


def _cross_chunk_trigger_supported(
    relation_type: str,
    source_chunk_id: str,
    target_chunk_id: str,
    text_by_chunk_id: dict[str, str],
) -> bool:
    if source_chunk_id == target_chunk_id:
        return True
    triggers = RELATION_CROSS_CHUNK_TRIGGERS.get(relation_type)
    if not triggers:
        return False
    text = f"{text_by_chunk_id.get(source_chunk_id, '')}\n{text_by_chunk_id.get(target_chunk_id, '')}"
    return any(trigger in text for trigger in triggers)


def build_candidate_relationships(
    entities: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    document_id: str = "",
    source_pdf: str = "",
    max_cross_chunk_distance: int = DEFAULT_MAX_CROSS_CHUNK_DISTANCE,
    same_chunk_only: bool = DEFAULT_SAME_CHUNK_ONLY,
) -> list[CandidateRelationship]:
    """Generate legal schema candidates in layered chunk-distance order."""

    order = _chunk_order(chunks)
    text_by_chunk_id = _chunk_text_by_id(chunks)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in entities:
        entity_type = str(entity.get("entity_type") or "")
        if entity_type:
            grouped[entity_type].append(entity)

    candidates: list[CandidateRelationship] = []
    seen: set[tuple[str, str, str, str]] = set()
    for start_type, relation_type, end_type in LEGAL_RELATIONS:
        for start in grouped.get(start_type, []):
            for end in grouped.get(end_type, []):
                start_id = _entity_id(start)
                end_id = _entity_id(end)
                if not start_id or not end_id or start_id == end_id:
                    continue
                start_doc = _document_id_for_entity(start, document_id)
                end_doc = _document_id_for_entity(end, document_id)
                if start_doc and end_doc and start_doc != end_doc:
                    continue
                source_chunk_id = str(start.get("chunk_id") or "")
                target_chunk_id = str(end.get("chunk_id") or "")
                if same_chunk_only and source_chunk_id != target_chunk_id:
                    continue
                is_document_has_sub_disease = relation_type == "has_sub_disease" and start_type == "diseases"
                if not _within_cross_chunk_limit(
                    source_chunk_id,
                    target_chunk_id,
                    order,
                    max_cross_chunk_distance,
                ) and not is_document_has_sub_disease:
                    continue
                chunk_distance = _chunk_distance(source_chunk_id, target_chunk_id, order)
                relation_max_distance = min(
                    max_cross_chunk_distance,
                    RELATION_MAX_CHUNK_DISTANCE.get(relation_type, max_cross_chunk_distance),
                )
                if chunk_distance > relation_max_distance and not is_document_has_sub_disease:
                    continue
                if not _cross_chunk_trigger_supported(
                    relation_type,
                    source_chunk_id,
                    target_chunk_id,
                    text_by_chunk_id,
                ) and not is_document_has_sub_disease:
                    continue
                search_level = "document" if is_document_has_sub_disease and chunk_distance else _candidate_search_level(chunk_distance)
                key = (start_id, relation_type, end_id, start_doc or document_id)
                if key in seen:
                    continue
                seen.add(key)
                effective_doc = start_doc or end_doc or document_id
                effective_pdf = _source_pdf_for_entity(start, source_pdf) or _source_pdf_for_entity(end, source_pdf)
                chunk_id = _choose_candidate_chunk(source_chunk_id, target_chunk_id, order)
                candidates.append(
                    CandidateRelationship(
                        candidate_id=_candidate_id(effective_doc, start_id, end_id, relation_type),
                        document_id=effective_doc,
                        source_pdf=effective_pdf,
                        chunk_id=chunk_id,
                        source_chunk_id=source_chunk_id,
                        target_chunk_id=target_chunk_id,
                        start_entity_id=start_id,
                        end_entity_id=end_id,
                        start_entity_type=start_type,
                        end_entity_type=end_type,
                        start_entity_name=_entity_name(start),
                        end_entity_name=_entity_name(end),
                        relation_type=relation_type,
                        relation_name=_relation_name(relation_type),
                        search_level=search_level,
                        chunk_distance=chunk_distance,
                    )
                )
    candidates.sort(
        key=lambda candidate: (
            candidate.chunk_distance,
            candidate.relation_type,
            candidate.source_chunk_id,
            candidate.target_chunk_id,
            candidate.start_entity_id,
            candidate.end_entity_id,
        )
    )
    return candidates


def _extract_json_object(text: str) -> dict[str, Any] | None:
    clean = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.S)
    if fence_match:
        clean = fence_match.group(1)
    else:
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            clean = clean[start : end + 1]
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _bounded_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _chunk_text(chunk: dict[str, Any]) -> str:
    text = str(chunk.get("text") or "")
    title = str(chunk.get("section_title") or "")
    path = chunk.get("section_path", [])
    path_text = " > ".join(str(item) for item in path) if isinstance(path, list) else str(path or "")
    header = f"[{_chunk_id(chunk)}] {path_text} {title}".strip()
    return f"{header}\n{text}".strip()


def _find_window(text: str, needles: Iterable[str], *, window: int = 180) -> str:
    for needle in needles:
        clean = str(needle or "").strip()
        if not clean:
            continue
        index = text.find(clean)
        if index < 0:
            continue
        start = max(0, index - window)
        end = min(len(text), index + len(clean) + window)
        return text[start:end]
    return ""


def _context_map(
    candidate: CandidateRelationship,
    entity_by_id: dict[str, dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    max_document_chars: int,
) -> dict[str, str]:
    chunk_by_id = {_chunk_id(chunk): chunk for chunk in chunks}
    order = _chunk_order(chunks)
    source_chunk = chunk_by_id.get(candidate.source_chunk_id, {})
    target_chunk = chunk_by_id.get(candidate.target_chunk_id, {})
    start_entity = entity_by_id.get(candidate.start_entity_id, {})
    end_entity = entity_by_id.get(candidate.end_entity_id, {})

    joined_entity_text = "\n".join(
        part
        for part in [
            _chunk_text(source_chunk),
            _chunk_text(target_chunk) if candidate.target_chunk_id != candidate.source_chunk_id else "",
        ]
        if part
    )
    entity_window = _find_window(
        joined_entity_text,
        [
            _entity_evidence(start_entity),
            _entity_name(start_entity),
            _entity_evidence(end_entity),
            _entity_name(end_entity),
        ],
    )
    if not entity_window:
        entity_window = "\n".join(
            part
            for part in [
                f"start_evidence: {_entity_evidence(start_entity)}",
                f"end_evidence: {_entity_evidence(end_entity)}",
            ]
            if part.strip()
        )

    chunk_context = "\n\n".join(
        part
        for part in [
            _chunk_text(source_chunk),
            _chunk_text(target_chunk) if candidate.target_chunk_id != candidate.source_chunk_id else "",
        ]
        if part
    )

    neighbor_indexes: set[int] = set()
    for chunk_id in [candidate.source_chunk_id, candidate.target_chunk_id]:
        if chunk_id not in order:
            continue
        index = order[chunk_id]
        neighbor_indexes.update({index - 1, index, index + 1})
    neighbor_context = "\n\n".join(
        _chunk_text(chunks[index])
        for index in sorted(neighbor_indexes)
        if 0 <= index < len(chunks)
    )

    document_context = "\n\n".join(_chunk_text(chunk) for chunk in chunks)
    if len(document_context) > max_document_chars:
        document_context = document_context[:max_document_chars]

    return {
        "entity_window": entity_window,
        "chunk": chunk_context or entity_window,
        "neighbor_chunks": neighbor_context or chunk_context or entity_window,
        "document": document_context or neighbor_context or chunk_context or entity_window,
    }


def _audit_prompt(candidate: CandidateRelationship, context_level: str, context: str) -> str:
    contract = RELATION_SEMANTIC_CONTRACTS.get(candidate.relation_type, {})
    return f"""Decide whether the candidate relationship is supported by the source context. Output strict JSON only.
Use the relation definition, direction, positive examples, negative examples, and reasoning role below. Do not approve a relation just because two entities co-occur.

Relation semantic contract:
{json.dumps(contract, ensure_ascii=False, indent=2)}

Candidate relationship:
{json.dumps(candidate.to_record(), ensure_ascii=False, indent=2)}

Context level: {context_level}
Context:
{context}

Approval rules:
- approved=true only when the text explicitly supports this directed relation, or when the relation follows rigorously from the same sentence/paragraph.
- evidence must be the minimal source span supporting this exact relation.
- approved=false when the entities are merely listed together, the direction is wrong, the context is insufficient, or the entity-type semantics do not match.

Output JSON:
{{
  "approved": true,
  "evidence": "minimal source span supporting the relation; empty string if unsupported",
  "confidence": 0.0
}}
"""


def _batch_audit_prompt(candidates: list[CandidateRelationship], context_level: str, contexts: dict[str, str]) -> str:
    relation_contracts = {
        relation_type: RELATION_SEMANTIC_CONTRACTS.get(relation_type, {})
        for relation_type in sorted({candidate.relation_type for candidate in candidates})
    }
    candidate_payload = []
    for candidate in candidates:
        candidate_payload.append(
            {
                "candidate": candidate.to_record(),
                "context": contexts.get(candidate.candidate_id, ""),
            }
        )
    return f"""Decide whether each candidate relationship is supported by its source context. Output strict JSON only.
Use the relation definitions, directions, positive examples, negative examples, and reasoning roles below. Do not approve a relation just because two entities co-occur.

Relation semantic contracts:
{json.dumps(relation_contracts, ensure_ascii=False, indent=2)}

Context level: {context_level}

Candidate relationships with context:
{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}

Approval rules:
- approved=true only when the text explicitly supports this directed relation, or when the relation follows rigorously from the same sentence/paragraph.
- evidence must be the minimal source span supporting this exact relation.
- approved=false when the entities are merely listed together, the direction is wrong, the context is insufficient, or the entity-type semantics do not match.
- Return exactly one result object for every candidate_id.

Output JSON:
{{
  "results": [
    {{
      "candidate_id": "REL_CAND_xxx",
      "approved": true,
      "evidence": "minimal source span supporting the relation; empty string if unsupported",
      "confidence": 0.0
    }}
  ]
}}
"""


def _chunks_of(items: list[CandidateRelationship], size: int) -> Iterable[list[CandidateRelationship]]:
    step = max(1, size)
    for index in range(0, len(items), step):
        yield items[index : index + step]


def _parse_batch_audit_results(
    parsed: dict[str, Any] | None,
    candidates: list[CandidateRelationship],
) -> dict[str, tuple[bool, str, float]]:
    if not parsed:
        return {candidate.candidate_id: (False, "", 0.0) for candidate in candidates}
    if "results" not in parsed and len(candidates) == 1:
        candidate = candidates[0]
        return {
            candidate.candidate_id: (
                bool(parsed.get("approved")),
                str(parsed.get("evidence") or "").strip(),
                _bounded_confidence(parsed.get("confidence")),
            )
        }
    rows = parsed.get("results")
    if not isinstance(rows, list):
        return {candidate.candidate_id: (False, "", 0.0) for candidate in candidates}
    by_id: dict[str, tuple[bool, str, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            continue
        by_id[candidate_id] = (
            bool(row.get("approved")),
            str(row.get("evidence") or "").strip(),
            _bounded_confidence(row.get("confidence")),
        )
    return {
        candidate.candidate_id: by_id.get(candidate.candidate_id, (False, "", 0.0))
        for candidate in candidates
    }


def _audit_candidate_batch(
    candidates: list[CandidateRelationship],
    context_maps: dict[str, dict[str, str]],
    *,
    context_level: str,
    llm_func: LLMCallable,
    config_path: str | Path | None,
    raw_rows: list[dict[str, Any]],
    raw_output_path: str | Path | None = None,
    progress_label: str = "",
) -> dict[str, tuple[bool, str, float]]:
    contexts = {
        candidate.candidate_id: context_maps.get(candidate.candidate_id, {}).get(context_level, "")
        for candidate in candidates
    }
    prompt = _batch_audit_prompt(candidates, context_level, contexts)
    parsed, raw = _call_llm_json(
        llm_func,
        prompt,
        config_path=config_path,
        progress_label=progress_label,
    )
    results = _parse_batch_audit_results(parsed, candidates)
    for candidate in candidates:
        approved, evidence, confidence = results[candidate.candidate_id]
        raw_row = {
            "phase": "audit",
            "mode": "batch",
            "candidate_id": candidate.candidate_id,
            "relation_type": candidate.relation_type,
            "search_level": candidate.search_level,
            "chunk_distance": candidate.chunk_distance,
            "context_level": context_level,
            "approved": approved,
            "confidence": confidence,
            "raw_response": raw,
            "parsed": parsed,
        }
        raw_rows.append(raw_row)
        _append_jsonl(raw_row, raw_output_path)
    return results


def _property_defaults(relation_type: str) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for field_name in RELATION_PROPERTY_MAP.get(relation_type, []):
        if field_name in {"evidence_ids", "decision_basis"}:
            defaults[field_name] = []
        elif field_name in {
            "confidence",
            "value_min",
            "value_max",
            "weight",
            "required_count",
            "section_priority",
        }:
            defaults[field_name] = None
        elif field_name == "gold_standard":
            defaults[field_name] = None
        else:
            defaults[field_name] = ""
    return defaults


def _sanitize_relation_properties(
    relation_type: str,
    raw_properties: Any,
    *,
    relation_id: str,
    relation_name: str,
) -> dict[str, Any]:
    if not isinstance(raw_properties, dict):
        raw_properties = {}
    allowed_fields = set(RELATION_PROPERTY_MAP.get(relation_type, []))
    properties = _property_defaults(relation_type)
    for field_name in RELATION_PROPERTY_MAP.get(relation_type, []):
        if field_name in raw_properties:
            properties[field_name] = raw_properties[field_name]
    if "relation_id" in allowed_fields:
        properties["relation_id"] = relation_id
    if "relation_name" in allowed_fields:
        properties["relation_name"] = relation_name
    if "relation_type" in allowed_fields:
        properties["relation_type"] = relation_type
    if "confidence" in allowed_fields and properties.get("confidence") is not None:
        properties["confidence"] = _bounded_confidence(properties["confidence"])
    if "evidence_ids" in allowed_fields and properties.get("evidence_ids") in ("", None):
        properties["evidence_ids"] = []
    return properties


def _properties_have_signal(properties: dict[str, Any], relation_type: str) -> bool:
    ignored = {"relation_id", "relation_name", "relation_type", "confidence", "evidence_ids"}
    defaults = _property_defaults(relation_type)
    for key, value in properties.items():
        if key in ignored:
            continue
        default = defaults.get(key)
        if value not in ("", None) and value != [] and value != default:
            return True
    return False


def _relation_property_contract_block(field_names: list[str]) -> str:
    rows: list[str] = []
    for field_name in field_names:
        contract = RELATION_PROPERTY_CONTRACTS.get(
            field_name,
            {
                "meaning": "Schema allowed relation property.",
                "fill_when": "Fill only when supported by source text.",
                "boundary": "Leave empty/null/list default when unsupported.",
                "positive": "",
                "negative": "",
            },
        )
        rows.append(
            "\n".join(
                [
                    f"- property: {field_name}",
                    f"  meaning: {contract['meaning']}",
                    f"  fill_when: {contract['fill_when']}",
                    f"  semantic_boundary: {contract['boundary']}",
                    f"  positive_example: {contract['positive']}",
                    f"  negative_example: {contract['negative']}",
                ]
            )
        )
    return "\n".join(rows)


def _property_prompt(
    candidate: CandidateRelationship,
    relation_id: str,
    context_level: str,
    context: str,
) -> str:
    fields = RELATION_PROPERTY_MAP.get(candidate.relation_type, [])
    template = _sanitize_relation_properties(
        candidate.relation_type,
        {},
        relation_id=relation_id,
        relation_name=candidate.relation_name,
    )
    contract = RELATION_SEMANTIC_CONTRACTS.get(candidate.relation_type, {})
    return f"""Extract properties for an already approved relationship. Output strict JSON only.
Properties must be supported by the context; do not omit a supported property, and do not invent unsupported values.

Relation semantic contract:
{json.dumps(contract, ensure_ascii=False, indent=2)}

Approved relationship:
{json.dumps(candidate.to_record(), ensure_ascii=False, indent=2)}

Allowed property fields:
{json.dumps(fields, ensure_ascii=False)}

Allowed property semantic contracts:
{_relation_property_contract_block(fields)}

Property template:
{json.dumps(template, ensure_ascii=False, indent=2)}

Property guidance:
- diagnostic_role: role of symptom/test/etiology in diagnosis, such as required, supportive, exclusion, risk_factor, gold_standard.
- type_info: relation subtype or clinical category when stated.
- value_min/value_max/operator/unit/result_text: numeric or textual diagnostic criterion for requires_test.
- weight/typicality/gold_standard/criterion_group_id/required_count/polarity: diagnostic strength and rule logic when stated.
- clinical_stage/treatment_line/applicable_condition/recommendation_polarity/recommendation_strength/decision_basis/contraindication_note: treatment reasoning attributes when stated.
- plan_role/causal_strength/evidence_level/source_section: fill only when supported by text.
- relation_id, relation_name, relation_type must stay consistent with the template.

Context level: {context_level}
Context:
{context}

Output JSON:
{{
  "properties": {{}},
  "evidence": "minimal source span supporting the extracted properties; empty string when no property is supported",
  "confidence": 0.0
}}
"""

def _call_llm_json(
    llm_func: LLMCallable,
    prompt: str,
    *,
    config_path: str | Path | None,
    progress_label: str = "",
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if progress_label:
        result = _call_with_heartbeat(
            lambda: llm_func(prompt, system_prompt=RELATION_SYSTEM_PROMPT, config_path=config_path),
            label=progress_label,
        )
    else:
        result = llm_func(prompt, system_prompt=RELATION_SYSTEM_PROMPT, config_path=config_path)
    if result.get("status") != "ok":
        return None, result
    parsed = _extract_json_object(str(result.get("text") or ""))
    return parsed, result


def _audit_candidate(
    candidate: CandidateRelationship,
    contexts: dict[str, str],
    *,
    llm_func: LLMCallable,
    config_path: str | Path | None,
    raw_rows: list[dict[str, Any]],
    raw_output_path: str | Path | None = None,
    progress_label: str = "",
    detail_output: bool = True,
) -> tuple[bool, str, float, str]:
    for context_level in CONTEXT_LEVELS:
        if progress_label and detail_output:
            _progress(f"{progress_label} | audit | context={context_level} | start")
        prompt = _audit_prompt(candidate, context_level, contexts.get(context_level, ""))
        parsed, raw = _call_llm_json(
            llm_func,
            prompt,
            config_path=config_path,
            progress_label=f"{progress_label} | audit | context={context_level}",
        )
        approved = bool(parsed.get("approved")) if parsed else False
        evidence = str(parsed.get("evidence") or "").strip() if parsed else ""
        confidence = _bounded_confidence(parsed.get("confidence")) if parsed else 0.0
        if progress_label and detail_output:
            _progress(
                f"{progress_label} | audit | context={context_level} | "
                f"status={raw.get('status')} approved={approved} confidence={confidence:.3f}"
            )
        raw_row = {
            "phase": "audit",
            "candidate_id": candidate.candidate_id,
            "relation_type": candidate.relation_type,
            "search_level": candidate.search_level,
            "chunk_distance": candidate.chunk_distance,
            "context_level": context_level,
            "approved": approved,
            "raw_response": raw,
            "parsed": parsed,
        }
        raw_rows.append(raw_row)
        _append_jsonl(raw_row, raw_output_path)
        if approved and evidence:
            return True, evidence, confidence, context_level
    return False, "", 0.0, CONTEXT_LEVELS[-1]


def _extract_properties(
    candidate: CandidateRelationship,
    relation_id: str,
    contexts: dict[str, str],
    *,
    llm_func: LLMCallable,
    config_path: str | Path | None,
    raw_rows: list[dict[str, Any]],
    raw_output_path: str | Path | None = None,
    progress_label: str = "",
    detail_output: bool = True,
) -> tuple[dict[str, Any], str, float, str]:
    fallback_properties = _sanitize_relation_properties(
        candidate.relation_type,
        {},
        relation_id=relation_id,
        relation_name=candidate.relation_name,
    )
    last_evidence = ""
    last_confidence = 0.0
    last_level = CONTEXT_LEVELS[-1]
    for context_level in CONTEXT_LEVELS:
        if progress_label and detail_output:
            _progress(f"{progress_label} | properties | context={context_level} | start")
        prompt = _property_prompt(candidate, relation_id, context_level, contexts.get(context_level, ""))
        parsed, raw = _call_llm_json(
            llm_func,
            prompt,
            config_path=config_path,
            progress_label=f"{progress_label} | properties | context={context_level}",
        )
        raw_properties = parsed.get("properties", {}) if parsed else {}
        properties = _sanitize_relation_properties(
            candidate.relation_type,
            raw_properties,
            relation_id=relation_id,
            relation_name=candidate.relation_name,
        )
        evidence = str(parsed.get("evidence") or "").strip() if parsed else ""
        confidence = _bounded_confidence(parsed.get("confidence")) if parsed else 0.0
        if progress_label and detail_output:
            signal = _properties_have_signal(properties, candidate.relation_type)
            _progress(
                f"{progress_label} | properties | context={context_level} | "
                f"status={raw.get('status')} signal={signal} confidence={confidence:.3f}"
            )
        raw_row = {
            "phase": "properties",
            "candidate_id": candidate.candidate_id,
            "relation_id": relation_id,
            "relation_type": candidate.relation_type,
            "search_level": candidate.search_level,
            "chunk_distance": candidate.chunk_distance,
            "context_level": context_level,
            "raw_response": raw,
            "parsed": parsed,
            "kept_properties": properties,
        }
        raw_rows.append(raw_row)
        _append_jsonl(raw_row, raw_output_path)
        fallback_properties = properties
        last_evidence = evidence
        last_confidence = confidence
        last_level = context_level
        if _properties_have_signal(properties, candidate.relation_type):
            return properties, evidence, confidence, context_level
    return fallback_properties, last_evidence, last_confidence, last_level


def _relationship_record(
    candidate: CandidateRelationship,
    *,
    relation_id: str,
    properties: dict[str, Any],
    evidence_text: str,
    confidence: float,
    context_level: str,
) -> dict[str, Any]:
    if "confidence" in properties:
        properties["confidence"] = confidence
    return {
        "relation_id": relation_id,
        "document_id": candidate.document_id,
        "source_pdf": candidate.source_pdf,
        "chunk_id": candidate.chunk_id,
        "source_chunk_id": candidate.source_chunk_id,
        "target_chunk_id": candidate.target_chunk_id,
        "start_entity_id": candidate.start_entity_id,
        "end_entity_id": candidate.end_entity_id,
        "start_entity_type": candidate.start_entity_type,
        "end_entity_type": candidate.end_entity_type,
        "relation_type": candidate.relation_type,
        "relation_name": candidate.relation_name,
        "properties": properties,
        "evidence_text": evidence_text,
        "confidence": confidence,
        "context_level": context_level,
        "status": "confirmed",
        "source": "llm_relationship_extraction",
    }


def _sub_disease_anchor_id(candidate: CandidateRelationship) -> str:
    if candidate.start_entity_type == "sub_diseases":
        return candidate.start_entity_id
    if candidate.end_entity_type == "sub_diseases":
        return candidate.end_entity_id
    return ""


def _candidate_allowed_by_coverage(
    candidate: CandidateRelationship,
    coverage: dict[str, Counter[str]],
    confirmed_treatment_ids: set[str],
    coverage_targets: dict[str, int],
) -> bool:
    if candidate.relation_type == "implements_by":
        return candidate.start_entity_id in confirmed_treatment_ids
    anchor_id = _sub_disease_anchor_id(candidate)
    if not anchor_id:
        return True
    target = coverage_targets.get(candidate.relation_type)
    if target is None:
        return True
    return coverage[anchor_id][candidate.relation_type] < target


def _update_coverage(
    candidate: CandidateRelationship,
    coverage: dict[str, Counter[str]],
    confirmed_treatment_ids: set[str],
) -> None:
    anchor_id = _sub_disease_anchor_id(candidate)
    if anchor_id:
        coverage[anchor_id][candidate.relation_type] += 1
    if candidate.relation_type == "follows_treatment":
        confirmed_treatment_ids.add(candidate.end_entity_id)


def extract_relationship_base_for_file(
    *,
    entities_path: str | Path,
    chunks_path: str | Path,
    output_path: str | Path,
    candidate_output_path: str | Path | None = None,
    raw_output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    config_path: str | Path | None = None,
    llm_func: LLMCallable = chat_completion_text,
    max_document_chars: int = DEFAULT_MAX_DOCUMENT_CHARS,
    max_cross_chunk_distance: int = DEFAULT_MAX_CROSS_CHUNK_DISTANCE,
    same_chunk_only: bool = DEFAULT_SAME_CHUNK_ONLY,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    audit_batch_size: int = DEFAULT_AUDIT_BATCH_SIZE,
    include_review_entities: bool = DEFAULT_INCLUDE_REVIEW_ENTITIES,
    coverage_targets: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build relationship_base for one document."""

    doc_name = _doc_label(output_path)
    _progress(f"[{doc_name}] stage 1/5 load inputs")
    _progress(f"[{doc_name}] entities={Path(entities_path).name} chunks={Path(chunks_path).name}")
    raw_entities = _read_jsonl(entities_path)
    entities = _filter_final_entities(
        raw_entities,
        include_review_entities=include_review_entities,
    )
    chunks, chunk_payload = _load_chunks(chunks_path)
    document_id = str(chunk_payload.get("doc_id") or chunk_payload.get("document_id") or "")
    source_pdf = str(
        chunk_payload.get("pdf_path")
        or chunk_payload.get("source_title")
        or Path(chunks_path).name.removesuffix(".chunk.json")
    )
    sub_disease_count = sum(
        1 for entity in entities if str(entity.get("entity_type") or "") == "sub_diseases"
    )
    if sub_disease_count == 0:
        raise ValueError(
            f"[{doc_name}] No sub_diseases found in entity_base; "
            "abort relationship extraction because Sub_disease is required as the graph reasoning hub."
        )
    _progress(f"[{doc_name}] stage 2/5 build legal candidate relationships")
    candidates = build_candidate_relationships(
        entities,
        chunks,
        document_id=document_id,
        source_pdf=source_pdf,
        max_cross_chunk_distance=max_cross_chunk_distance,
        same_chunk_only=same_chunk_only,
    )
    total_generated_candidates = len(candidates)
    if max_candidates and max_candidates > 0 and len(candidates) > max_candidates:
        _progress(
            f"[{doc_name}] candidate limit active: generated={len(candidates)} "
            f"processing_first={max_candidates}"
        )
        candidates = candidates[:max_candidates]
    _progress(
        f"[{doc_name}] inputs loaded: document_id={document_id} "
        f"entities={len(entities)} raw_entities={len(raw_entities)} sub_diseases={sub_disease_count} chunks={len(chunks)} "
        f"candidates={len(candidates)} generated_candidates={total_generated_candidates} "
        f"same_chunk_only={same_chunk_only} context_levels={','.join(CONTEXT_LEVELS)}"
    )
    relation_type_counts = Counter(candidate.relation_type for candidate in candidates)
    search_level_counts = Counter(candidate.search_level for candidate in candidates)
    _progress(
        f"[{doc_name}] candidate distribution: "
        + ", ".join(f"{key}={value}" for key, value in sorted(relation_type_counts.items()))
    )
    _progress(
        f"[{doc_name}] candidate search levels: "
        + ", ".join(f"{key}={value}" for key, value in sorted(search_level_counts.items()))
    )
    candidate_rows = [candidate.to_record() for candidate in candidates]
    if candidate_output_path is not None:
        write_jsonl(candidate_rows, candidate_output_path)
        _progress(f"[{doc_name}] candidates written: {Path(candidate_output_path).name}")

    entity_by_id = {_entity_id(entity): entity for entity in entities if _entity_id(entity)}
    raw_rows: list[dict[str, Any]] = [
        {"phase": "candidate", "candidate": candidate_row}
        for candidate_row in candidate_rows
    ]
    relationship_rows: list[dict[str, Any]] = []
    if raw_output_path is not None:
        write_jsonl(raw_rows, raw_output_path)
        _progress(f"[{doc_name}] raw trace initialized: {Path(raw_output_path).name} rows={len(raw_rows)}")
    write_jsonl([], output_path)
    _progress(f"[{doc_name}] relationship_base initialized for incremental writes: {Path(output_path).name}")
    rejected_count = 0
    total_candidates = len(candidates)
    processed_count = 0
    _progress(
        f"[{doc_name}] stage 3/5 audit candidates with batched layered LLM "
        f"batch_size={audit_batch_size}"
    )
    candidates_by_level: dict[str, list[CandidateRelationship]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_level[candidate.search_level].append(candidate)

    active_coverage_targets = dict(DEFAULT_COVERAGE_TARGETS)
    if coverage_targets:
        active_coverage_targets.update(coverage_targets)
    coverage: dict[str, Counter[str]] = defaultdict(Counter)
    confirmed_treatment_ids: set[str] = set()
    confirmed_relation_keys: set[tuple[str, str, str]] = set()

    for search_level in ("same_chunk", "adjacent_chunk", "near_chunk_window", "document"):
        for phase in ("primary", "implements_by"):
            raw_level_candidates = candidates_by_level.get(search_level, [])
            level_candidates = [
                candidate
                for candidate in raw_level_candidates
                if (
                    candidate.relation_type != "implements_by"
                    if phase == "primary"
                    else candidate.relation_type == "implements_by"
                )
                and _candidate_allowed_by_coverage(
                    candidate,
                    coverage,
                    confirmed_treatment_ids,
                    active_coverage_targets,
                )
            ]
            if not level_candidates:
                continue
            _progress(
                f"[{doc_name}] audit layer start: {search_level} phase={phase} "
                f"candidates={len(level_candidates)}"
            )
            context_maps = {
                candidate.candidate_id: _context_map(
                    candidate,
                    entity_by_id,
                    chunks,
                    max_document_chars=max_document_chars,
                )
                for candidate in level_candidates
            }
            pending = list(level_candidates)
            approved_payloads: list[tuple[CandidateRelationship, str, float, str]] = []
            for context_level in CONTEXT_LEVELS:
                pending = [
                    candidate
                    for candidate in pending
                    if _candidate_allowed_by_coverage(
                        candidate,
                        coverage,
                        confirmed_treatment_ids,
                        active_coverage_targets,
                    )
                ]
                if not pending:
                    break
                next_pending: list[CandidateRelationship] = []
                batch_count = 0
                _progress(
                    f"[{doc_name}] audit layer={search_level} phase={phase} context={context_level} "
                    f"pending={len(pending)}"
                )
                for batch in _chunks_of(pending, audit_batch_size):
                    filtered_batch = [
                        candidate
                        for candidate in batch
                        if _candidate_allowed_by_coverage(
                            candidate,
                            coverage,
                            confirmed_treatment_ids,
                            active_coverage_targets,
                        )
                    ]
                    if not filtered_batch:
                        continue
                    batch_count += 1
                    label = (
                        f"[{doc_name}] batch audit layer={search_level} phase={phase} "
                        f"context={context_level} batch={batch_count} size={len(filtered_batch)}"
                    )
                    results = _audit_candidate_batch(
                        filtered_batch,
                        context_maps,
                        context_level=context_level,
                        llm_func=llm_func,
                        config_path=config_path,
                        raw_rows=raw_rows,
                        raw_output_path=raw_output_path,
                        progress_label=label,
                    )
                    for candidate in filtered_batch:
                        approved, evidence, confidence = results[candidate.candidate_id]
                        if approved and evidence:
                            _update_coverage(candidate, coverage, confirmed_treatment_ids)
                            approved_payloads.append((candidate, evidence, confidence, context_level))
                        else:
                            next_pending.append(candidate)
                    processed_count += len(filtered_batch)
                    if batch_count == 1 or batch_count % 5 == 0:
                        _progress(
                            f"[{doc_name}] batch progress: checked={processed_count} "
                            f"confirmed={len(relationship_rows)} rejected={rejected_count} "
                            f"current_pending={len(next_pending)}"
                        )
                pending = next_pending
            rejected_count += len(pending)
            _progress(
                f"[{doc_name}] audit layer done: {search_level} phase={phase} "
                f"approved={len(approved_payloads)} rejected={len(pending)}"
            )

            for candidate, audit_evidence, audit_confidence, audit_level in approved_payloads:
                relation_key = (candidate.start_entity_id, candidate.relation_type, candidate.end_entity_id)
                if relation_key in confirmed_relation_keys:
                    continue
                confirmed_relation_keys.add(relation_key)
                index = candidates.index(candidate) + 1
                progress_label = f"[{doc_name}] {_candidate_label(candidate, index, total_candidates)}"
                detailed = _should_detail_candidate(index, total_candidates)
                _progress(f"{progress_label} | stage 4/5 extract relation properties")
                contexts = context_maps[candidate.candidate_id]
                relation_id = _relation_id(
                    candidate.document_id,
                    candidate.start_entity_id,
                    candidate.end_entity_id,
                    candidate.relation_type,
                )
                properties, property_evidence, property_confidence, property_level = _extract_properties(
                    candidate,
                    relation_id,
                    contexts,
                    llm_func=llm_func,
                    config_path=config_path,
                    raw_rows=raw_rows,
                    raw_output_path=raw_output_path,
                    progress_label=progress_label,
                    detail_output=detailed,
                )
                relationship_row = _relationship_record(
                    candidate,
                    relation_id=relation_id,
                    properties=properties,
                    evidence_text=property_evidence or audit_evidence,
                    confidence=max(property_confidence, audit_confidence),
                    context_level=property_level if property_evidence else audit_level,
                )
                relationship_rows.append(relationship_row)
                _append_jsonl(relationship_row, output_path)
                _progress(
                    f"{progress_label} | confirmed relation_id={relation_id} "
                    f"confidence={max(property_confidence, audit_confidence):.3f}"
                )

    _progress(f"[{doc_name}] stage 5/5 write outputs")
    write_jsonl(relationship_rows, output_path)
    _progress(f"[{doc_name}] relationship_base written: {Path(output_path).name} confirmed={len(relationship_rows)} rejected={rejected_count}")
    if raw_output_path is not None:
        write_jsonl(raw_rows, raw_output_path)
        _progress(f"[{doc_name}] raw trace written: {Path(raw_output_path).name} rows={len(raw_rows)}")

    summary = {
        "entities_path": str(entities_path),
        "chunks_path": str(chunks_path),
        "relationship_path": str(output_path),
        "candidate_output_path": str(candidate_output_path) if candidate_output_path else None,
        "raw_output_path": str(raw_output_path) if raw_output_path else None,
        "document_id": document_id,
        "source_pdf": source_pdf,
        "entities": len(entities),
        "raw_entities": len(raw_entities),
        "sub_diseases": sub_disease_count,
        "chunks": len(chunks),
        "candidates": len(candidates),
        "generated_candidates": total_generated_candidates,
        "max_candidates": max_candidates,
        "audit_batch_size": audit_batch_size,
        "candidate_search_levels": dict(sorted(search_level_counts.items())),
        "confirmed_relationships": len(relationship_rows),
        "rejected_candidates": rejected_count,
        "same_chunk_only": same_chunk_only,
        "include_review_entities": include_review_entities,
        "coverage_targets": active_coverage_targets,
        "coverage_by_sub_disease": {key: dict(value) for key, value in coverage.items()},
        "context_levels": list(CONTEXT_LEVELS),
    }
    if summary_path is not None:
        write_json(summary, summary_path)
        _progress(f"[{doc_name}] summary written: {Path(summary_path).name}")
    _progress(f"[{doc_name}] complete")
    return summary
