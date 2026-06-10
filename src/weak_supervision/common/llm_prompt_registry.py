"""Central prompt registry for two-layer Snorkel LLM Labeling Functions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptLFSpec:
    """One prompted-LLM labeling-function prompt variant."""

    name: str
    focus: str
    instruction: str


LV1_CHUNK_PROMPT_SPECS: tuple[PromptLFSpec, ...] = (
    PromptLFSpec(
        name="semantic_presence",
        focus="语义召回视角：判断 chunk 是否在医学语义上出现某类实体，即使表达不是词典原词也可以给出正票。",
        instruction="适合发现同义表达、缩写、诊疗语义和跨句描述；必须给出原文 evidence，不确定时降低 confidence。",
    ),
    PromptLFSpec(
        name="evidence_anchor",
        focus="证据锚定视角：只在 chunk 中存在清晰、可逐字定位的实体证据时给出正票。",
        instruction="优先降低幻觉和误召回；evidence 必须是 chunk 原文连续片段，不能改写或概括。",
    ),
    PromptLFSpec(
        name="boundary_count",
        focus="边界与数量视角：判断实体边界是否独立，并估计每类不同实体的数量。",
        instruction="区分实体本体和关系/属性描述；不要把剂量、频率、阈值等属性误判为新的实体类型。",
    ),
)


LV2_ENTITY_PROMPT_SPECS: tuple[PromptLFSpec, ...] = (
    PromptLFSpec(
        name="type_boundary",
        focus="实体类型边界视角：判断候选名称和证据更符合哪一种最终实体类型。",
        instruction=(
            "重点区分 Sub_disease、Symptom、Test、Treatment、Plan、Etiology、Pathogenesis；"
            "不要把关系词、属性值、剂量、阈值、频率或诊疗建议片段误判为实体本体。"
        ),
    ),
    PromptLFSpec(
        name="evidence_support",
        focus="证据支持视角：判断候选 evidence 是否足以支持该候选属于某个实体类型。",
        instruction=(
            "只有 evidence 或 chunk 上下文明确支持该类型时才 present=true；"
            "如果证据只是提到名称但不能支持类型，或 evidence 无法回链，应降低 confidence 或 present=false。"
        ),
    ),
    PromptLFSpec(
        name="schema_contrast",
        focus="Schema 对比视角：把候选同时和所有实体类型边界做对比，给出最合适类型。",
        instruction=(
            "需要考虑该候选为什么不像其他类型；若多个类型都合理，应降低 confidence，"
            "让 Snorkel LabelModel 和 top2_gap 将其送入 review。"
        ),
    ),
)


def prompt_specs_from_config(raw_prompts: object, defaults: tuple[PromptLFSpec, ...]) -> list[PromptLFSpec]:
    """Read prompt specs from YAML-like config, falling back to defaults."""

    if not isinstance(raw_prompts, list):
        return list(defaults)
    specs: list[PromptLFSpec] = []
    for index, item in enumerate(raw_prompts, start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or f"prompt_{index}").strip()
        if not name:
            continue
        focus = str(item.get("focus") or item.get("prompt_focus") or "").strip()
        instruction = str(item.get("instruction") or item.get("prompt_instruction") or "").strip()
        specs.append(
            PromptLFSpec(
                name=name,
                focus=focus or f"Prompt {index} for Snorkel LLM LF.",
                instruction=instruction or "Return votes and abstentions from direct source evidence.",
            )
        )
    return specs or list(defaults)
