# entity_extraction 目录开发维护文档

## 目录职责

维护 Lv1 与 Lv2 之间的候选实体抽取中间层：使用 Lv1 的 `present/status/predicted_count` 结果约束 Teacher LLM，从 chunk 文本中抽取高召回候选实体，并生成不含最终属性的 `entity_base.jsonl`。

## 文件与功能说明

- `__init__.py`：候选实体抽取包初始化文件。
- `llm_entity_extractor.py`：维护动态 prompt 构造、Teacher LLM 调用和响应解析。
- `entity_base_builder.py`：维护 `entity_base` 记录构建、span 校验、去重和 trace 字段写入。
- `entity_property_extractor.py`：维护 Lv2 后实体属性抽取、schema 字段白名单、默认核心字段兜底和 `entity_nodes` 构建。
- `core_disease.py`：从 PDF 标题/文件名推断文档级核心 `Disease` 实体，例如从《中国急性胰腺炎诊治指南（2021）》抽取“急性胰腺炎”。
- `entity_normalizer.py`：维护候选实体文本标准化。
- `entity_schema.py`：维护 `entity_base` 相关 Schema/字段辅助函数。

## 已实现功能

- 已实现受 Lv1 约束的 Teacher LLM 预标注 prompt 构造。
- 动态 prompt 会根据 Lv1 的 `present/status/predicted_count` 对每个标签选择 extract 或 check-only 模式。
- 已支持 `Full_extraction` 全量抽取模式：忽略 Lv1 阳性 chunk 过滤，所有接收 chunk 的 active entity types 都进入 extract 模式，且不按 Lv1 negative 生成 override 标记。
- Teacher 响应解析会规整五个列表字段，并展平为候选实体记录。
- `override_lv1` 会在解析后根据 Lv1 标签级决策重新计算，不信任 Teacher LLM 自己输出的该字段。
- `entity_base` 构建已支持基础去重和 trace 字段附加；Teacher LLM 初抽取属性只保留在 `candidate_properties`，最终 `properties` 保持空对象。
- Lv2 后属性抽取按最终实体类型选择 schema，属性失败时仍保留实体节点并记录 `property_incomplete` conflict。
- span 校验目前保持轻量；候选抽取质量稳定后再补充严格 evidence anchoring。
- 候选实体 schema 已同步 Graph_shema_v1.0：Disease 为文档级实体，chunk 级预抽取覆盖 Sub_Disease、Symptom、Test、Treatment、Plan、Etiology、Pathogenesis。

## 维护约束

- 本层初抽取只生成候选实体，不做最终实体类型融合；最终判定属于 Lv2。
- 实体属性抽取必须在 Lv2 定型之后运行，且不得改变实体类型。
- Teacher LLM 结果必须受 Lv1 约束，不能覆盖 Lv1 明确缺席的标签决策。
- 输出记录要保留 chunk/document、来源标签、证据、span、Teacher 原始字段和 Lv1 trace，方便后续排错。
- LLM 调用必须支持禁用、缺少 API Key、超时和解析失败时安全返回。

## 开发修改日志

- 2026-06-03：`llm_entity_extractor.py` 重构为本地 schema 驱动 prompt；新增实体后处理校验，只保留允许属性字段，丢弃空 name/evidence 实体，并在 raw response 中记录 `schema_warnings`。
- 2026-06-03：第二轮优化实体抽取 prompt，按实体类型补充抽取对象、禁止对象、evidence 粒度和关系构建交接约束，强调实体本体与后续关系属性信息分离。
- 2026-05-19：创建 Lv1 与 Lv2 之间的候选实体抽取包。
- 2026-05-20：实现受 Lv1 约束的动态 Teacher LLM 预标注抽取和 `entity_base` 记录构建。
- 2026-05-20：新增解析后 Lv1 标签覆盖校验，确保实体标记与 Lv1 标签决策一致。
- 2026-05-21：将维护文档翻译并同步为中文。
- 2026-05-21：补充预抽取提示词的实体属性 schema、症状/指标边界、治疗原则/治疗方案抽取粒度，并在实体构建时回填属性 ID。
- 2026-05-22：新增文档级核心 `Disease` 规则抽取，按标题生成父级疾病节点；同步收紧 `sub_diseases` prompt，要求只抽核心疾病下的部位、时期、病因、严重程度等更具体确诊名。
- 2026-05-22：增强 `sub_diseases` 边界控制，要求候选确诊名必须是文档核心 `Disease` 的子类；解析后会把不属于核心疾病子类的其他疾病名从 `sub_diseases` 迁移到 `symptoms`，例如 `Disease=急性胰腺炎` 时 `尿路感染` 作为伴随临床问题进入 symptoms。
- 2026-06-03：按 `code/data/Graph_Design/Graph_shema_v1.0.docx` 同步最终版实体属性，并将 Teacher LLM 预抽取扩展到 `etiologies`、`pathogeneses`。
- 2026-06-03：新增 `Full_extraction` 全量抽取开关，用于不依赖 Lv1 阳性 chunk 的 Teacher LLM 全量实体预抽取。
- 2026-06-04：`entity_base` 不再承载最终属性；新增 `entity_property_extractor.py`，在 Lv2 后按最终实体类型补属性并构建 `entity_nodes`。
