# weak_supervision_lv2/lfs 目录开发维护文档

## 目录职责

维护 Lv2 entity 级 Labeling Functions，用于把 `entity_base` 候选实体判定为最终图谱实体类型。

当前目录只有占位文件和模块说明，具体匹配逻辑尚未实现。

## 文件与功能说明

- `__init__.py`：Lv2 LF 包初始化文件。
- `lf_entity_dictionary.py`：实体词典类型投票 LF 占位。
- `lf_entity_context_window.py`：候选实体前后文窗口类型投票 LF 占位。
- `lf_entity_section_prior.py`：章节路径先验 LF 占位。
- `lf_entity_surface_pattern.py`：实体表面形式模式 LF 占位。
- `lf_entity_prompted_llm.py`：可选 Teacher LLM entity LF 占位；后续实现时禁用或失败必须 abstain。

## 当前状态

- Lv2 LF 文件已经创建，但尚未实现可执行的 `EntityLabelingFunction` 子类。
- `entity_lf_applier.py` 已经能批量应用符合协议的 LF；因此本目录后续只需要补充具体 LF 类即可接入。

## 维护约束

- 具体 LF 必须以 `entity_base` 候选实体为中心，不要重新从 chunk 中自由抽取新实体。
- 每个 LF 必须返回标准 `LFOutput`，并保留候选实体、标签、证据、置信度和 abstain 原因。
- 实体类型标签必须与 `common/graph_schema.py` 和配置文件保持一致。
- LLM LF 必须有配置开关、安全回退和不确定 abstain 行为。

## 开发修改日志

- 2026-05-19：创建 Lv2 entity 级 LF 骨架。
- 2026-05-21：将维护文档翻译并同步为中文，明确当前仍是占位阶段。
