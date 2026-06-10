"""Schema-constrained LLM candidate entity extraction."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from src.entity_extraction.entity_schema import (
    ACTIVE_PRELABEL_TYPES,
    DEFAULT_PROPERTIES_BY_TYPE,
    LIST_FIELD_BY_TYPE,
    NAME_FIELD_BY_TYPE,
    PROPERTY_FIELDS_BY_TYPE,
    PrelabelEntity,
    empty_prelabel_response,
)
from src.entity_extraction.entity_normalizer import normalize_entity_text
from src.utils.llm_client import chat_completion_text

LLMCallable = Callable[..., dict[str, Any]]


PRELABEL_SYSTEM_PROMPT = """你是医学专家共识知识图谱的实体抽取器。
只允许基于当前 chunk 原文抽取实体。输出必须是严格 JSON，不要输出解释性正文。
所有实体必须提供非空 name、非空 evidence、0 到 1 之间的 confidence，以及符合本地 schema 的 properties。
无法从原文确认的实体不要输出；不得补齐原文没有支持的信息。
"""


LABEL_GUIDANCE: dict[str, str] = {
    "sub_diseases": (
        "抽取核心 Disease 下更具体的诊断名、疾病亚型、分型、分期、严重程度、解剖部位限定、"
        "病因限定或人群限定名称。实体名称应尽量使用原文完整诊断短语，例如“胆源性急性胰腺炎”、"
        "“重症急性胰腺炎”、“合并尿路结石的复杂性尿路感染”。"
    ),
    "symptoms": (
        "抽取患者主观症状、客观体征、临床表现、并发症表现、伴随临床问题或原文明确列为表现的异常状态。"
        "如果一个疾病名不是当前核心 Disease 的子类，但在原文中作为合并症、既往病史、鉴别诊断或伴随临床问题出现，"
        "实体阶段可放入 symptoms，并在 evidence 中保留其语义角色。"
    ),
    "tests": (
        "抽取诊断或评估所需的检查项目、检测指标、评分量表、影像学检查、病原学检查、实验室指标。"
        "优先抽取检查/指标本体名称；阈值、单位、比较符、阳性/阴性结果、诊断角色等不要作为独立 Test 实体，"
        "但必须保留在 evidence 中，供后续 requires_test 关系属性抽取。"
    ),
    "treatments": (
        "抽取治疗原则、治疗目标、原则性推荐、处理策略或决策框架。Treatment 是原则层，不是处方层；"
        "应保留能够表达原则适用条件、推荐强度、禁忌或决策依据的连续原文。"
    ),
    "plans": (
        "抽取具体治疗方案、药物类别、药名、剂量、给药途径、频次、疗程、复查、随访或操作计划。"
        "Plan 是 Treatment 的落地方案；如果原文同时包含原则和具体方案，原则放 treatments，具体执行内容放 plans。"
    ),
    "etiologies": (
        "抽取导致、诱发、促进或显著关联 Sub_disease 的病因、危险因素、病原体、结石因素、免疫因素、"
        "遗传因素、肿瘤因素、暴露因素等。必须有因果、诱因、危险因素、病因学或诊断支持语义。"
    ),
    "pathogeneses": (
        "抽取疾病发生发展机制、病理生理过程、免疫/炎症级联、结构改变、代谢改变、组织损伤路径或机制解释。"
        "Pathogenesis 关注“如何发生/为什么发展”的过程性解释，不等同于单个病因名称。"
    ),
}


NEGATIVE_GUIDANCE: dict[str, str] = {
    "sub_diseases": (
        "文档核心 Disease 本身、指南主题、泛称如“疾病/综合征/病变”、普通症状、检查项目、病因因素、"
        "机制描述、治疗原则或具体治疗方案。不要把只表达共病/伴随问题但非核心 Disease 子类的疾病名放入 sub_diseases。"
    ),
    "symptoms": (
        "检查项目、检验指标、阈值单位、病因因素、机制过程、治疗原则、具体治疗方案、药物剂量、单纯章节标题。"
    ),
    "tests": (
        "症状体征、病因或病原体名称本身、治疗药物、治疗原则、具体治疗方案、孤立阈值/单位/比较符/结果文本。"
    ),
    "treatments": (
        "单个药名、剂量、给药频次、疗程、具体处方组合、随访复查安排、检查建议、只有标题没有原则内容的短语。"
    ),
    "plans": (
        "治疗原则、治疗目标、决策依据、推荐强度、没有具体执行内容的泛化表达、单纯治疗章节标题。"
    ),
    "etiologies": (
        "普通症状、检查项目、治疗药物、机制过程、没有因果/诱因/危险因素/病因学语义的背景描述。"
    ),
    "pathogeneses": (
        "单个病因名称、危险因素名称、症状名称、检查项目、治疗措施、没有机制过程的泛化结论。"
    ),
}


EVIDENCE_GUIDANCE: dict[str, str] = {
    "sub_diseases": (
        "evidence 至少包含完整诊断名；若同句说明分型、分期、严重程度、部位、病因或适用人群，应一并保留。"
    ),
    "symptoms": (
        "evidence 应包含症状/体征原文和必要的上下文，如“表现为/常见症状/伴有/合并/出现”等触发词。"
    ),
    "tests": (
        "evidence 应包含检查名及其阈值、单位、比较符、结果、阳性/阴性或诊断意义；例如不要只保留“尿培养”，"
        "应保留“尿培养菌落计数≥10^5 cfu/ml 具有诊断意义”这类可支持关系属性的片段。"
    ),
    "treatments": (
        "evidence 应保留完整原则句或连续原则段，包含适用条件、推荐/不推荐、禁忌、决策依据等上下文。"
    ),
    "plans": (
        "evidence 应保留具体方案片段，包含药物/类别、剂量、途径、频次、疗程、复查或随访条件。"
    ),
    "etiologies": (
        "evidence 必须包含病因/诱因/危险因素/因果关联触发词，不能只保留一个孤立名词。"
    ),
    "pathogeneses": (
        "evidence 必须包含机制过程或解释性谓词，如“导致/引起/通过/造成/进展为/机制为”等。"
    ),
}


RELATION_HANDOFF_GUIDANCE: dict[str, str] = {
    "sub_diseases": "后续会作为 manifests_as/requires_test/follows_treatment/causes/explained_by 的起点，evidence 要利于判断该 subtype 与症状、检查、治疗、病因、机制的关系。",
    "symptoms": "后续通过 manifests_as 连接 Sub_disease，evidence 要能判断该表现是否属于某个 subtype。",
    "tests": "后续通过 requires_test 连接 Sub_disease，阈值、单位、operator、result_text、diagnostic_role 等保留在 evidence，不要拆成实体属性外的旧字段。",
    "treatments": "后续通过 follows_treatment 连接 Sub_disease，并通过 implements_by 连接 Plan，evidence 要能区分原则和具体方案。",
    "plans": "后续通过 implements_by 连接 Treatment，evidence 要能判断该方案落实的是哪条治疗原则。",
    "etiologies": "后续通过 causes 连接 Sub_disease，evidence 要能支持因果或危险因素关系。",
    "pathogeneses": "后续通过 explained_by 连接 Sub_disease，evidence 要能支持机制解释关系。",
}

ENTITY_SEMANTIC_CONTRACTS: dict[str, dict[str, Any]] = {
    "diseases": {
        "definition": "Disease 是疾病大类或上位疾病概念，必须来自当前 chunk 原文，不允许根据 PDF 标题、文件名或外部常识补造。它用于连接更具体的确诊名/分型。",
        "positive": ["尿路感染", "急性胰腺炎", "淋巴瘤"],
        "negative": ["诊治指南", "专家共识", "治疗原则", "尿频", "尿培养", "左氧氟沙星"],
        "property_rule": "必须同步填写 disease_name、normalized_name、aliases、confidence；没有依据的 aliases/source_document_ids/evidence_ids 用空数组。",
        "reasoning_role": "后续通过 has_sub_disease 连接 Sub_disease，是图谱推理的父级疾病节点。",
    },
    "sub_diseases": {
        "definition": "Sub_disease 是明确诊断名、疾病分型、疾病亚型、严重程度限定、部位限定、病因限定、人群限定或临床场景限定的疾病名称。只要当前 chunk 原文把它作为疾病/诊断/分型/感染类型/并发感染类型表达，就应抽取为 sub_diseases，不需要依赖标题截断出的核心疾病。",
        "positive": ["复杂性尿路感染", "非发热性尿路感染", "合并尿路结石的复杂性尿路感染", "梗阻性急性肾盂肾炎", "急性胆源性胰腺炎"],
        "negative": ["尿频", "发热", "尿培养", "药敏试验", "经验性治疗", "头孢呋辛"],
        "property_rule": "必须同步填写 sub_disease_name；如原文出现分型、分期、严重程度、人群、性别、年龄范围、父级疾病线索，分别写入 disease_subtype、clinical_stage、severity、population、gender、age_min、age_max、parent_disease_id。没有依据则留空或 null。",
        "reasoning_role": "Sub_disease 是推理中心节点，后续连接症状、检查、治疗原则、病因和发病机制。",
    },
    "symptoms": {
        "definition": "Symptom 是症状、体征、临床表现、异常状态或并发症表现。疾病名称只有在原文明确把它当作伴随问题/并发症表现而非确诊名时才放入 symptoms。",
        "positive": ["尿频", "尿急", "尿痛", "发热", "肋脊角压痛", "脓尿"],
        "negative": ["复杂性尿路感染", "尿培养", "C反应蛋白", "抗菌药物治疗", "尿路梗阻"],
        "property_rule": "必须同步填写 symptom_name；能从原文判断部位、类别、阳性/阴性、典型性时填写 body_site、symptom_category、polarity、typicality。",
        "reasoning_role": "后续通过 manifests_as 连接 Sub_disease，用于症状支持诊断推理。",
    },
    "tests": {
        "definition": "Test 是检查项目、检验指标、影像学检查、病原学检查、培养、药敏、评分或诊断标准中的检测项。检查结果、阈值、单位和阳性/阴性不是单独实体，但必须保留在 evidence 中。",
        "positive": ["尿培养", "药敏试验", "尿液分析", "血培养", "C反应蛋白", "降钙素原"],
        "negative": ["尿频", "复杂性尿路感染", "左氧氟沙星", "经验性治疗"],
        "property_rule": "必须同步填写 test_name；如原文给出数值上下限或是否数值型，填写 normal_range_min、normal_range_max、is_digital。阈值、单位、operator、result_text、diagnostic_role 保留在 evidence，供关系属性抽取。",
        "reasoning_role": "后续通过 requires_test 连接 Sub_disease，用于诊断标准和检查依据推理。",
    },
    "treatments": {
        "definition": "Treatment 是治疗原则、治疗目标、推荐策略、处理原则或决策框架，不是具体药名剂量。",
        "positive": ["根据尿培养和药敏试验结果选择敏感抗菌药物", "经验性治疗需根据临床反应和尿培养结果及时修正", "即刻的肾脏集合系统减压"],
        "negative": ["左氧氟沙星500mg每日1次", "尿培养", "复杂性尿路感染", "发热"],
        "property_rule": "必须同步填写 treatment_content；如原文给出推荐强度、来源章节，填写 recommendation_strength、source_section。",
        "reasoning_role": "后续通过 follows_treatment 连接 Sub_disease，再通过 implements_by 连接 Plan。",
    },
    "plans": {
        "definition": "Plan 是具体治疗方案、药物/剂量/途径/频次/疗程、操作方案、监测方案、随访方案或执行计划。",
        "positive": ["左氧氟沙星500mg静脉或口服每日1次", "经皮肾造瘘", "逆行输尿管插管", "术后1-2年低剂量预防性抗菌药物治疗"],
        "negative": ["根据药敏选择抗菌药物", "复杂性尿路感染", "尿频", "尿培养"],
        "property_rule": "必须同步填写 plan_content；能判断治疗线、适用条件、禁忌或注意事项时填写 plan_level、applicable_condition、contraindication_note。",
        "reasoning_role": "后续通过 implements_by 与 Treatment 相连，支撑从治疗原则推理到具体执行方案。",
    },
    "etiologies": {
        "definition": "Etiology 是病因、危险因素、诱因、病原体、解剖/功能异常、基础疾病、器械暴露等导致或增加疾病风险的因素。",
        "positive": ["糖尿病", "免疫缺陷", "尿路梗阻", "导尿管留置", "大肠埃希菌", "铜绿假单胞菌"],
        "negative": ["发热", "尿培养", "左氧氟沙星", "经验性治疗"],
        "property_rule": "必须同步填写 etiology_content；如能判断病因类型、典型性、诊断角色，填写 etiology_type、typicality、diagnostic_role。",
        "reasoning_role": "后续通过 causes 连接 Sub_disease，用于病因和风险因素推理。",
    },
    "pathogeneses": {
        "definition": "Pathogenesis 是发病机制、病理生理过程、免疫/炎症/代谢/结构改变、疾病进展机制或机制性解释。",
        "positive": ["产尿素酶细菌水解尿素升高尿液pH并促进结晶沉积", "细菌内毒素释放触发系统炎症应答反应"],
        "negative": ["大肠埃希菌", "尿频", "尿培养", "抗菌药物治疗"],
        "property_rule": "必须同步填写 pathogenesis_content；机制证据必须保留过程性表达，不能只抽一个名词。",
        "reasoning_role": "后续通过 explained_by 连接 Sub_disease，用于机制解释推理。",
    },
}

ENTITY_PROPERTY_CONTRACTS: dict[str, dict[str, str]] = {
    "disease_id": {
        "meaning": "Disease 节点唯一标识；抽取阶段可留空，由后处理回填。",
        "fill_when": "仅当原文或上游已有稳定 ID 时填写。",
        "boundary": "不要填写疾病名称、ICD 编码或随机解释。",
        "positive": "TEMP_xxx 或既有 disease_id",
        "negative": "尿路感染",
    },
    "disease_name": {
        "meaning": "疾病大类或上位疾病名称。",
        "fill_when": "当前 chunk 原文明确出现疾病大类时填写，通常与 name 一致。",
        "boundary": "不要写指南标题、章节标题、症状、检查、治疗原则。",
        "positive": "尿路感染",
        "negative": "尿路感染诊断与治疗中国专家共识",
    },
    "normalized_name": {
        "meaning": "规范化疾病名，用于同义合并。",
        "fill_when": "能从 disease_name 去除年份、指南词、空格、括号噪声后得到规范名时填写。",
        "boundary": "不要引入原文没有支持的更大疾病类或外部知识标准名。",
        "positive": "复杂性尿路感染 -> 复杂性尿路感染",
        "negative": "复杂性尿路感染 -> 感染性疾病",
    },
    "aliases": {
        "meaning": "原文同一实体的别名、缩写或中英文等价表达。",
        "fill_when": "同一 chunk 明确给出别名、括号缩写或等价名称时填写数组。",
        "boundary": "不要凭常识扩展同义词。",
        "positive": '["cUTI"]',
        "negative": '["泌尿系统疾病"]',
    },
    "source_document_ids": {
        "meaning": "实体来自的文档 ID 列表。",
        "fill_when": "上游已有明确 document_id 时填写；否则空数组。",
        "boundary": "不要写 PDF 文件名或标题文本。",
        "positive": '["DOC_001"]',
        "negative": '["尿路感染诊断与治疗中国专家共识.pdf"]',
    },
    "evidence_ids": {
        "meaning": "支撑该实体或关系的证据 ID 列表。",
        "fill_when": "上游已有证据 ID 时填写；没有则空数组，证据文本放 evidence。",
        "boundary": "不要把证据原文长文本塞进 evidence_ids。",
        "positive": '["EV_001"]',
        "negative": '["尿培养阳性支持诊断"]',
    },
    "confidence": {
        "meaning": "当前实体或关系及其属性的置信度，0 到 1。",
        "fill_when": "根据原文明确程度填写数值。",
        "boundary": "不要填写百分号字符串或超出 0-1 的数。",
        "positive": "0.86",
        "negative": "86%",
    },
    "sub_disease_id": {
        "meaning": "Sub_disease 节点唯一标识；抽取阶段可留空，由后处理回填。",
        "fill_when": "仅当已有稳定 ID 时填写。",
        "boundary": "不要填写确诊名文本。",
        "positive": "TEMP_xxx",
        "negative": "复杂性尿路感染",
    },
    "sub_disease_name": {
        "meaning": "明确诊断名、分型、亚型、感染类型或限定诊断名。",
        "fill_when": "当前 chunk 原文把该短语作为疾病/诊断/分型/感染类型表达时填写。",
        "boundary": "不要放症状、检查、药物、治疗原则；疾病名不要误归 symptoms。",
        "positive": "合并尿路结石的复杂性尿路感染",
        "negative": "尿频",
    },
    "parent_disease_id": {
        "meaning": "父级 Disease 的 ID。",
        "fill_when": "当前 chunk 或上游明确知道父级 Disease ID 时填写。",
        "boundary": "没有 ID 时不要把父级疾病名称写进此字段；父级名称可在 evidence 中保留。",
        "positive": "TEMP_DISEASE_001",
        "negative": "尿路感染",
    },
    "disease_subtype": {
        "meaning": "分型/亚型类别，例如复杂性、非发热性、梗阻性、胆源性。",
        "fill_when": "确诊名或上下文明确说明 subtype 时填写。",
        "boundary": "不要写治疗线、症状类别或检查类别。",
        "positive": "复杂性",
        "negative": "一线治疗",
    },
    "clinical_stage": {
        "meaning": "疾病分期、阶段或临床时期。",
        "fill_when": "原文明确出现 I期、急性期、复发期、术后等阶段时填写。",
        "boundary": "不要把严重程度、治疗线或章节名误写为分期。",
        "positive": "术后",
        "negative": "重症",
    },
    "severity": {
        "meaning": "严重程度或风险级别。",
        "fill_when": "原文明确描述轻中度、重症、危重、高危等程度时填写。",
        "boundary": "不要写疾病分期、治疗线或推荐强度。",
        "positive": "重症",
        "negative": "二线",
    },
    "population": {
        "meaning": "适用人群或患者亚群。",
        "fill_when": "原文明确限定成人、儿童、妊娠、导管相关、结石合并感染等人群/场景时填写。",
        "boundary": "不要写症状、检查结果或药物名。",
        "positive": "合并尿路结石患者",
        "negative": "尿培养阳性",
    },
    "gender": {
        "meaning": "性别限定。",
        "fill_when": "原文明确男/女/妊娠女性等性别适用条件时填写。",
        "boundary": "无性别限定时留空或 all，不要猜测。",
        "positive": "female",
        "negative": "老年",
    },
    "age_min": {
        "meaning": "适用年龄下限，数值。",
        "fill_when": "原文明确年龄下限时填写。",
        "boundary": "不要写年龄段文本；文本可留 evidence。",
        "positive": "18",
        "negative": "成人",
    },
    "age_max": {
        "meaning": "适用年龄上限，数值。",
        "fill_when": "原文明确年龄上限时填写。",
        "boundary": "不要写年龄段文本；文本可留 evidence。",
        "positive": "65",
        "negative": "老年",
    },
    "symptom_id": {
        "meaning": "Symptom 节点唯一标识；抽取阶段可留空，由后处理回填。",
        "fill_when": "仅当已有稳定 ID 时填写。",
        "boundary": "不要填写症状名称。",
        "positive": "TEMP_xxx",
        "negative": "发热",
    },
    "symptom_name": {
        "meaning": "症状、体征、临床表现或异常状态名称。",
        "fill_when": "当前 chunk 原文明确出现表现、症状、体征或异常状态时填写。",
        "boundary": "不要把明确诊断名、检查项目、治疗方案写成症状。",
        "positive": "肋脊角压痛",
        "negative": "复杂性尿路感染",
    },
    "body_site": {
        "meaning": "症状或体征发生部位。",
        "fill_when": "原文或症状名明确部位时填写。",
        "boundary": "不要填写疾病大类或检查科室。",
        "positive": "肾区",
        "negative": "感染性疾病",
    },
    "symptom_category": {
        "meaning": "症状类别，如下尿路刺激症状、全身感染表现、体征、并发症表现。",
        "fill_when": "原文或医学表达能直接支持分类时填写。",
        "boundary": "不要写病因类别、治疗类别或检查类别。",
        "positive": "下尿路刺激症状",
        "negative": "病原学检查",
    },
    "polarity": {
        "meaning": "症状/检查/关系的阳性、阴性、存在、排除等极性。",
        "fill_when": "原文明确有/无、阳性/阴性、排除/不支持时填写。",
        "boundary": "不要把典型性或推荐强度写进 polarity。",
        "positive": "positive",
        "negative": "common",
    },
    "typicality": {
        "meaning": "典型性或常见程度。",
        "fill_when": "原文出现常见、典型、少见、罕见、高危等语义时填写。",
        "boundary": "不要写阳性/阴性或推荐强度。",
        "positive": "common",
        "negative": "positive",
    },
    "test_id": {
        "meaning": "Test 节点唯一标识；抽取阶段可留空，由后处理回填。",
        "fill_when": "仅当已有稳定 ID 时填写。",
        "boundary": "不要填写检查名称。",
        "positive": "TEMP_xxx",
        "negative": "尿培养",
    },
    "test_name": {
        "meaning": "检查项目、检验指标、培养、药敏、影像学或评分名称。",
        "fill_when": "原文明确出现检查/指标本体时填写。",
        "boundary": "不要写检查结果、阈值、单位、疾病名或治疗名。",
        "positive": "尿培养",
        "negative": "菌落计数>=10^5 cfu/ml",
    },
    "normal_range_min": {
        "meaning": "检查指标正常或诊断范围下限，数值。",
        "fill_when": "原文明确给出下限时填写。",
        "boundary": "不要写单位、比较符或完整阈值文本。",
        "positive": "10",
        "negative": ">=10 cfu/ml",
    },
    "normal_range_max": {
        "meaning": "检查指标正常或诊断范围上限，数值。",
        "fill_when": "原文明确给出上限时填写。",
        "boundary": "不要写单位、比较符或完整阈值文本。",
        "positive": "100",
        "negative": "<100 mg/L",
    },
    "is_digital": {
        "meaning": "该检查/指标是否以数值结果为主。",
        "fill_when": "能判断是数值型指标时 true，培养/影像/定性结果可 false。",
        "boundary": "不确定则 null，不要把阳性/阴性写成 true/false。",
        "positive": "血白细胞计数 -> true",
        "negative": "尿培养阳性 -> true",
    },
    "treatment_id": {
        "meaning": "Treatment 节点唯一标识；抽取阶段可留空，由后处理回填。",
        "fill_when": "仅当已有稳定 ID 时填写。",
        "boundary": "不要填写治疗原则文本。",
        "positive": "TEMP_xxx",
        "negative": "经验性治疗",
    },
    "treatment_content": {
        "meaning": "治疗原则、治疗目标、推荐策略或处理原则的完整内容。",
        "fill_when": "原文表达原则层推荐、目标、决策或处理策略时填写。",
        "boundary": "不要只写单个药名剂量；具体执行方案放 plan_content。",
        "positive": "根据尿培养和药敏试验结果选择敏感抗菌药物",
        "negative": "左氧氟沙星500mg每日1次",
    },
    "recommendation_strength": {
        "meaning": "推荐强度或推荐极性。",
        "fill_when": "原文明确推荐、强推荐、不推荐、可考虑、建议等时填写。",
        "boundary": "不要写证据等级、治疗线或严重程度。",
        "positive": "推荐",
        "negative": "二线",
    },
    "source_section": {
        "meaning": "该实体或关系来自的章节/小节。",
        "fill_when": "chunk 中有 section_title 或原文明确章节时填写。",
        "boundary": "不要写整篇 PDF 标题或证据长文本。",
        "positive": "三、治疗",
        "negative": "尿培养提示...",
    },
    "plan_id": {
        "meaning": "Plan 节点唯一标识；抽取阶段可留空，由后处理回填。",
        "fill_when": "仅当已有稳定 ID 时填写。",
        "boundary": "不要填写方案文本。",
        "positive": "TEMP_xxx",
        "negative": "经皮肾造瘘",
    },
    "plan_content": {
        "meaning": "具体可执行方案，包括药物、剂量、途径、频次、疗程、操作、监测或随访。",
        "fill_when": "原文给出可执行方案时填写。",
        "boundary": "不要写抽象治疗原则或疾病名。",
        "positive": "左氧氟沙星500mg静脉或口服每日1次",
        "negative": "根据药敏选择抗菌药物",
    },
    "plan_level": {
        "meaning": "方案层级或角色，如一线、二线、目标治疗、预防、监测、干预。",
        "fill_when": "原文或上下文明确方案等级/用途时填写。",
        "boundary": "不要写推荐强度或疾病严重程度。",
        "positive": "first_line",
        "negative": "强推荐",
    },
    "applicable_condition": {
        "meaning": "实体、治疗原则、方案或关系适用条件。",
        "fill_when": "原文明确适用人群、病情、检查结果、阶段、失败条件等时填写。",
        "boundary": "不要写实体名称本身；不要猜测未说明条件。",
        "positive": "重症患者或初始经验性治疗失败患者",
        "negative": "复杂性尿路感染",
    },
    "contraindication_note": {
        "meaning": "禁忌、慎用、调整剂量、注意事项或不推荐原因。",
        "fill_when": "原文明确禁忌、肾功能不全调整、避免使用等时填写。",
        "boundary": "不要写普通适用条件或方案名称。",
        "positive": "肾功能不全者根据肌酐清除率调整剂量",
        "negative": "每日1次",
    },
    "etiology_id": {
        "meaning": "Etiology 节点唯一标识；抽取阶段可留空，由后处理回填。",
        "fill_when": "仅当已有稳定 ID 时填写。",
        "boundary": "不要填写病因名称。",
        "positive": "TEMP_xxx",
        "negative": "糖尿病",
    },
    "etiology_content": {
        "meaning": "病因、危险因素、诱因、病原体或暴露因素内容。",
        "fill_when": "原文表达导致、诱发、风险增加、病原学或基础因素时填写。",
        "boundary": "不要写症状、检查、治疗或机制过程。",
        "positive": "导尿管留置",
        "negative": "发热",
    },
    "etiology_type": {
        "meaning": "病因类别，如 pathogen、comorbidity、anatomic_factor、device_exposure。",
        "fill_when": "能从原文直接判断类别时填写。",
        "boundary": "不要写病因名称、典型性或诊断角色。",
        "positive": "pathogen",
        "negative": "大肠埃希菌",
    },
    "diagnostic_role": {
        "meaning": "该实体或关系在诊断中的角色，如 required、supportive、exclusion、risk_factor、gold_standard。",
        "fill_when": "原文说明诊断依据、支持、排除、危险因素、金标准等时填写。",
        "boundary": "不要写检查名称、数值阈值或治疗推荐。",
        "positive": "supportive",
        "negative": "尿培养",
    },
    "pathogenesis_id": {
        "meaning": "Pathogenesis 节点唯一标识；抽取阶段可留空，由后处理回填。",
        "fill_when": "仅当已有稳定 ID 时填写。",
        "boundary": "不要填写机制文本。",
        "positive": "TEMP_xxx",
        "negative": "细菌内毒素释放",
    },
    "pathogenesis_content": {
        "meaning": "发病机制或病理生理过程的完整过程性表达。",
        "fill_when": "原文解释如何发生、为什么进展、通过何种机制造成结果时填写。",
        "boundary": "不要只写单个病原体、症状、检查或治疗。",
        "positive": "产尿素酶细菌水解尿素升高尿液pH并促进结晶沉积",
        "negative": "变形杆菌",
    },
}


def _field_default(field_name: str) -> Any:
    if field_name in {"aliases", "source_document_ids", "evidence_ids", "decision_basis"}:
        return []
    if field_name in {
        "confidence",
        "age_min",
        "age_max",
        "normal_range_min",
        "normal_range_max",
        "plan_level",
        "is_digital",
        "gold_standard",
    }:
        return None
    return ""


def _entity_template(entity_type: str) -> dict[str, Any]:
    properties = {
        field_name: _field_default(field_name)
        for field_name in PROPERTY_FIELDS_BY_TYPE.get(entity_type, ())
    }
    return {
        "name": "",
        "properties": properties,
        "evidence": "",
        "confidence": 0.0,
        "override_lv1": False,
    }


def _property_contract_block(field_names: list[str]) -> str:
    rows: list[str] = []
    for field_name in field_names:
        contract = ENTITY_PROPERTY_CONTRACTS.get(
            field_name,
            {
                "meaning": "Schema allowed property.",
                "fill_when": "Fill only when supported by source text.",
                "boundary": "Leave empty/null/list default when unsupported.",
                "positive": "",
                "negative": "",
            },
        )
        rows.append(
            "\n".join(
                [
                    f"    - property: {field_name}",
                    f"      meaning: {contract['meaning']}",
                    f"      fill_when: {contract['fill_when']}",
                    f"      semantic_boundary: {contract['boundary']}",
                    f"      positive_example: {contract['positive']}",
                    f"      negative_example: {contract['negative']}",
                ]
            )
        )
    return "\n".join(rows)


OUTPUT_EXAMPLE = {
    LIST_FIELD_BY_TYPE[entity_type]: [_entity_template(entity_type)]
    for entity_type in ACTIVE_PRELABEL_TYPES
}
OUTPUT_EXAMPLE["lv1_overrides"] = []


def select_active_entity_prompts(
    lv1_results: list[dict[str, Any]],
    *,
    full_extraction: bool = False,
) -> dict[str, dict[str, Any]]:
    """Select extraction/check mode for each active entity type."""

    by_label = {str(item.get("label")): item for item in lv1_results}
    selected: dict[str, dict[str, Any]] = {}
    for label in ACTIVE_PRELABEL_TYPES:
        if full_extraction:
            selected[label] = {
                "mode": "extract",
                "present": True,
                "status": "full_extraction",
                "confidence": 0.0,
                "predicted_count": 0,
                "supporting_lfs": [],
                "evidence_texts": [],
            }
            continue
        result = by_label.get(label, {})
        present = bool(result.get("present", False))
        status = str(result.get("status", "rejected"))
        should_extract = present and status in {"accepted", "weak"}
        selected[label] = {
            "mode": "extract" if should_extract else "check_only",
            "present": present,
            "status": status,
            "confidence": float(result.get("confidence", 0.0) or 0.0),
            "predicted_count": int(result.get("predicted_count", result.get("count", 0)) or 0),
            "supporting_lfs": list(result.get("supporting_lfs", [])),
            "evidence_texts": list(result.get("evidence_texts", []))[:12],
        }
    return selected


def _schema_block(entity_type: str, spec: dict[str, Any]) -> str:
    list_field = LIST_FIELD_BY_TYPE[entity_type]
    name_field = NAME_FIELD_BY_TYPE[entity_type]
    allowed_fields = list(PROPERTY_FIELDS_BY_TYPE[entity_type])
    template = _entity_template(entity_type)
    contract = ENTITY_SEMANTIC_CONTRACTS[entity_type]
    return "\n".join(
        [
            f"- entity_type: {entity_type}",
            f"  output_list: {list_field}",
            f"  mode: {spec['mode']}",
            f"  predicted_count: {spec['predicted_count']}",
            f"  lv1_confidence: {spec['confidence']:.4f}",
            f"  definition: {contract['definition']}",
            f"  extract: {LABEL_GUIDANCE.get(entity_type, '')}",
            f"  do_not_extract: {NEGATIVE_GUIDANCE.get(entity_type, '')}",
            f"  positive_examples: {json.dumps(contract['positive'], ensure_ascii=False)}",
            f"  negative_examples: {json.dumps(contract['negative'], ensure_ascii=False)}",
            f"  property_rule: {contract['property_rule']}",
            f"  evidence_rule: {EVIDENCE_GUIDANCE.get(entity_type, '')}",
            f"  relation_handoff: {contract['reasoning_role']} {RELATION_HANDOFF_GUIDANCE.get(entity_type, '')}",
            f"  name_rule: name 必须非空；properties.{name_field} 必须与 name 或原文完整实体内容一致。",
            "  required_item_fields: name, properties, evidence, confidence",
            "  property_extraction_is_required: 抽取实体时必须同时抽取该实体类型允许的属性；原文支持的属性不得省略，不确定的属性才允许留空、null 或空数组。",
            f"  allowed_properties: {json.dumps(allowed_fields, ensure_ascii=False)}",
            "  property_contracts:",
            _property_contract_block(allowed_fields),
            f"  item_template: {json.dumps(template, ensure_ascii=False)}",
            f"  lv1_evidence: {json.dumps(spec['evidence_texts'], ensure_ascii=False)}",
        ]
    )

def build_prelabel_prompt(
    chunk: dict[str, Any],
    lv1_results: list[dict[str, Any]],
    *,
    full_extraction: bool = False,
) -> str:
    """Build a detailed schema-driven Teacher LLM pre-label prompt."""

    selected = select_active_entity_prompts(lv1_results, full_extraction=full_extraction)
    label_blocks = [_schema_block(label, spec) for label, spec in selected.items()]
    output_schema = {LIST_FIELD_BY_TYPE[label]: [] for label in ACTIVE_PRELABEL_TYPES}
    output_schema["lv1_overrides"] = []
    mode_rule = (
        "Full_extraction=true：所有 active entity types 都按 extract 处理，不要输出 override_lv1。"
        if full_extraction
        else "Full_extraction=false：mode=extract 的类型重点抽取；mode=check_only 的类型只有原文明确存在时才可抽取，并标记 override_lv1。"
    )

    return f"""请对以下 chunk 做医学知识图谱实体抽取。抽取必须完全基于当前 chunk 原文，不允许根据 PDF 标题、文件名、目录标题或外部常识补造实体。
章节标题: {chunk.get("section_title") or ""}
章节路径: {json.dumps(chunk.get("section_path", []), ensure_ascii=False)}

{mode_rule}

实体抽取总目标:
- 每个 chunk 内同时抽取实体和实体属性；不能只输出实体名再把属性留到后续阶段。
- Disease、Sub_disease、Symptom、Test、Treatment、Plan、Etiology、Pathogenesis 都是当前可抽取实体类型。
- Disease 必须来自 chunk 原文中的疾病大类或上位疾病概念，不再使用标题截断规则自动生成。
- Sub_disease/确诊名必须来自 chunk 原文中的明确诊断名、疾病分型、感染类型、部位/病因/严重程度/人群限定诊断名；例如“复杂性尿路感染”应是 sub_diseases，不应放入 symptoms。
- 对每个实体，必须输出 name、properties、evidence、confidence；properties 只能包含该实体类型 allowed_properties。
- evidence 必须是能支持该实体及属性的最小原文片段；如果属性来自同一句或同段原文，应体现在 evidence 中。
- 不确定的属性可以留空、null 或空数组；原文明确支持的属性不得省略。
- 不要为了补齐数量而臆造实体；不要输出 schema 未允许的旧字段或额外字段。
- 治疗原则和具体方案必须区分：原则/目标/推荐策略放 treatments，药物剂量、操作、疗程、监测和随访放 plans。
- etiologies 与 pathogeneses 必须区分：etiologies 是病因/风险因素/病原体/诱因，pathogeneses 是发生发展机制或病理生理解释。
- 检查实体和检查结果属性必须区分：检查名/指标名是 Test 实体；阈值、单位、比较符、阳性/阴性结果、诊断角色保留在 evidence，供后续关系属性抽取。
- 输出必须能支持后续图谱推理：Disease->Sub_disease、Sub_disease->Symptom/Test/Treatment/Etiology/Pathogenesis、Treatment->Plan。

动态实体任务:
{chr(10).join(label_blocks)}

输出严格 JSON，顶层字段必须包含:
{json.dumps(output_schema, ensure_ascii=False, indent=2)}

输出形状示例，仅用于字段格式，不代表当前 chunk 存在这些实体:
{json.dumps(OUTPUT_EXAMPLE, ensure_ascii=False, indent=2)}

chunk:
{chunk.get("text") or ""}
"""

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


def _item_name(item: Any, entity_type: str) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return ""
    return str(
        item.get("name")
        or item.get(NAME_FIELD_BY_TYPE[entity_type])
        or item.get("content")
        or ""
    )


def _sanitize_entity_item(
    item: Any,
    entity_type: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    item_dict = item if isinstance(item, dict) else {"name": str(item)}
    name = normalize_entity_text(_item_name(item_dict, entity_type))
    evidence = str(item_dict.get("evidence") or item_dict.get("evidence_text") or "").strip()
    if not name:
        warnings.append(f"{entity_type}: dropped item with empty name")
        return None, warnings
    if not evidence:
        warnings.append(f"{entity_type}:{name}: dropped item with empty evidence")
        return None, warnings

    raw_properties = item_dict.get("properties", {})
    if not isinstance(raw_properties, dict):
        raw_properties = {}
        warnings.append(f"{entity_type}:{name}: properties was not an object")

    allowed_fields = set(PROPERTY_FIELDS_BY_TYPE[entity_type])
    extra_fields = sorted(str(field) for field in raw_properties if field not in allowed_fields)
    if extra_fields:
        warnings.append(f"{entity_type}:{name}: removed unsupported properties {extra_fields}")

    properties = dict(DEFAULT_PROPERTIES_BY_TYPE.get(entity_type, {}))
    for field_name in PROPERTY_FIELDS_BY_TYPE[entity_type]:
        if field_name in raw_properties:
            properties[field_name] = raw_properties[field_name]

    name_field = NAME_FIELD_BY_TYPE[entity_type]
    if name_field in allowed_fields and not properties.get(name_field):
        properties[name_field] = name
    if "confidence" in allowed_fields:
        properties["confidence"] = _bounded_confidence(
            raw_properties.get("confidence", item_dict.get("confidence", properties.get("confidence")))
        )
    if "evidence_ids" in allowed_fields and properties.get("evidence_ids") in ("", None):
        properties["evidence_ids"] = []

    sanitized = {
        "name": name,
        "properties": properties,
        "evidence": evidence,
        "confidence": _bounded_confidence(item_dict.get("confidence")),
        "override_lv1": bool(item_dict.get("override_lv1", False)),
    }
    return sanitized, warnings


def normalize_prelabeled_entities(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten sanitized list fields into normalized candidate entity records."""

    entities: list[PrelabelEntity] = []
    seen: set[tuple[str, str]] = set()
    warnings = list(response.get("schema_warnings", []))
    for entity_type, field in LIST_FIELD_BY_TYPE.items():
        sanitized_items: list[dict[str, Any]] = []
        for item in response.get(field, []):
            sanitized, item_warnings = _sanitize_entity_item(item, entity_type)
            warnings.extend(item_warnings)
            if sanitized is None:
                continue
            sanitized_items.append(sanitized)
            key = (entity_type, sanitized["name"])
            if key in seen:
                warnings.append(f"{entity_type}:{sanitized['name']}: dropped duplicate entity")
                continue
            seen.add(key)
            entities.append(
                PrelabelEntity(
                    entity_type=entity_type,
                    name=sanitized["name"],
                    properties=sanitized["properties"],
                    evidence=sanitized["evidence"],
                    confidence=sanitized["confidence"],
                    override_lv1=sanitized["override_lv1"],
                )
            )
        response[field] = sanitized_items
    response["schema_warnings"] = warnings
    return [entity.to_record() for entity in entities]


def parse_teacher_prelabel_response(text: str) -> dict[str, Any]:
    """Parse Teacher LLM pre-label JSON into the normalized response shape."""

    parsed = _extract_json_object(text)
    if parsed is None:
        response = empty_prelabel_response()
        response["schema_warnings"] = ["failed to parse JSON object"]
        return response

    response = empty_prelabel_response()
    response["schema_warnings"] = []
    for field in LIST_FIELD_BY_TYPE.values():
        items = parsed.get(field, [])
        response[field] = items if isinstance(items, list) else []
        if field in parsed and not isinstance(items, list):
            response["schema_warnings"].append(f"{field}: expected list, got {type(items).__name__}")
    overrides = parsed.get("lv1_overrides", [])
    response["lv1_overrides"] = overrides if isinstance(overrides, list) else []
    response["entities"] = normalize_prelabeled_entities(response)
    return response


def enforce_core_disease_subtype_boundary(response: dict[str, Any], core_disease: str) -> dict[str, Any]:
    """Keep LLM subtype decisions based on chunk evidence, not title-derived anchors."""

    response["entities"] = normalize_prelabeled_entities(response)
    return response


def lv1_override_labels(lv1_results: list[dict[str, Any]]) -> set[str]:
    """Return labels where Teacher extraction would override a negative Lv1 decision."""

    by_label = {str(item.get("label")): item for item in lv1_results}
    overrides: set[str] = set()
    for label in ACTIVE_PRELABEL_TYPES:
        result = by_label.get(label, {})
        present = bool(result.get("present", False))
        status = str(result.get("status", "rejected"))
        if not present or status == "rejected":
            overrides.add(label)
    return overrides


def enforce_lv1_label_overrides(
    response: dict[str, Any],
    lv1_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute override flags from Lv1 label decisions instead of trusting LLM."""

    override_labels = lv1_override_labels(lv1_results)
    for entity_type, field in LIST_FIELD_BY_TYPE.items():
        should_override = entity_type in override_labels
        normalized_items: list[Any] = []
        for item in response.get(field, []):
            if isinstance(item, dict):
                fixed = dict(item)
                fixed["override_lv1"] = should_override
                normalized_items.append(fixed)
            else:
                normalized_items.append(
                    {
                        "name": str(item),
                        "properties": {},
                        "evidence": "",
                        "confidence": 0.0,
                        "override_lv1": should_override,
                    }
                )
        response[field] = normalized_items

    response["entities"] = normalize_prelabeled_entities(response)
    response["lv1_overrides"] = [
        {
            "entity_type": entity["entity_type"],
            "name": entity["name"],
            "reason": "teacher_extracted_despite_lv1_negative_label",
        }
        for entity in response["entities"]
        if entity.get("override_lv1")
    ]
    return response


def extract_prelabeled_entities(
    chunk: dict[str, Any],
    lv1_results: list[dict[str, Any]],
    *,
    config_path: str | Path | None = None,
    llm_func: LLMCallable = chat_completion_text,
    full_extraction: bool = False,
) -> dict[str, Any]:
    """Call Teacher LLM and return normalized pre-labeled entity candidates."""

    prompt = build_prelabel_prompt(chunk, lv1_results, full_extraction=full_extraction)
    result = llm_func(prompt, system_prompt=PRELABEL_SYSTEM_PROMPT, config_path=config_path)
    if result.get("status") != "ok":
        response = empty_prelabel_response()
        response["status"] = str(result.get("status") or "disabled")
        response["reason"] = str(result.get("reason") or "teacher_llm_unavailable")
        response["schema_warnings"] = []
        response["prompt"] = prompt
        return response

    response = parse_teacher_prelabel_response(str(result.get("text") or ""))
    response = enforce_core_disease_subtype_boundary(
        response,
        str(chunk.get("document_core_disease") or ""),
    )
    if full_extraction:
        for field in LIST_FIELD_BY_TYPE.values():
            for item in response.get(field, []):
                if isinstance(item, dict):
                    item["override_lv1"] = False
        response["entities"] = normalize_prelabeled_entities(response)
        response["lv1_overrides"] = []
    else:
        response = enforce_lv1_label_overrides(response, lv1_results)
    response["status"] = "ok"
    response["model"] = result.get("model")
    response["prompt"] = prompt
    response["Full_extraction"] = full_extraction
    return response
