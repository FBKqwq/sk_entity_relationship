# relationship_extraction 目录开发维护文档

## 目录职责

维护从 Lv2 后 `entity_nodes + chunks + graph_schema` 构建图谱关系的逻辑；legacy `entity_base` 仅保留给测试和历史回填兼容。

## 文件与功能说明

- `relationship_base_builder.py`：读取最终实体节点和 chunk，上下文检索，按 `GRAPH_RELATIONS` 生成合法候选关系，调用 LLM 审核关系成立性并抽取关系属性，输出 `relationship_base`、候选关系、raw trace 和 summary。

## 维护约束

- 关系阶段不得直接使用 raw `entity_base` 作为正常主管线输入；默认输入应为 `entity_nodes.jsonl`。
- 默认包含 `accepted + review` 实体保护关系候选召回，可通过入口脚本 `--accepted-only` 关闭。
- `Sub_disease` 是动态分层锚点，按 `same_chunk -> adjacent_chunk -> near_chunk_window` 逐层审核并根据覆盖目标停止扩窗。
- `implements_by` 只能在已确认 `Sub_disease -> Treatment` 后，以 Treatment 为二级锚点寻找 Plan。
- 同文档同 `(start_entity_id, relation_type, end_entity_id)` 只允许输出一条 confirmed relation。

## 开发修改日志

- 2026-06-03：新增关系抽取模块，支持两阶段关系构建：先按 schema 生成候选并审核是否由原文支持，再对已确认关系抽取 schema 允许的关系属性。
- 2026-06-03：补充关系抽取进度日志，输出文件加载、候选关系数量、候选审核上下文层级、属性抽取上下文层级、确认/拒绝计数和写出路径，便于长时间 LLM 任务观察是否仍在推进。
- 2026-06-03：优化关系抽取终端输出为阶段化日志：按文档输出加载、候选生成、审核、属性抽取、写出五个阶段；候选详细日志仅对前 5 条、每 25 条和最后一条输出，长时间 LLM 调用每 30 秒打印等待心跳。
- 2026-06-04：关系阶段改为优先消费 `entity_nodes`；新增 review 实体召回开关、Sub_disease 覆盖式动态扩窗、`implements_by` 二级锚点限制和 confirmed 关系去重。
