# weak_supervision_lv2 目录开发维护文档

## 目录职责

维护 Snorkel Lv2：以 `entity_base` 候选实体为输入，结合实体表面文本、上下文、章节和可选 Teacher LLM 信号，判定候选实体的最终图谱实体类型，并生成后续 `entity_nodes.jsonl`。

当前 Lv2 已落地为可执行实体准入层：对 `entity_base` 候选应用实体级 LF，输出最终实体类型、概率、top2 gap、LF trace、conflict 和 `accepted/review/rejected` 状态。

## 文件与功能说明

- `__init__.py`：Lv2 包初始化文件。
- `entity_lf_base.py`：entity 级 LF 抽象基类，约定 `apply(entity, chunk, label) -> LFOutput`。
- `entity_lf_applier.py`：将所有 entity LF 应用到所有 entity-label 组合。
- `entity_label_matrix.py`：提供 LF 输出分析与矩阵辅助函数。
- `entity_vote_model.py`：当前提供本地 softmax 工具，作为 fallback/debug 对照；主融合由共享层官方 Snorkel runner 承担。
- `entity_signal_builder.py`：从 Lv2 LF 输出构建 `entity_label_result`、conflicts 和 recall report。
- `lfs/`：实体级词典、上下文窗口、章节先验、表面模式和 Teacher LLM Prompt LF。

## 当前同步状态

- 已实现：`EntityLabelingFunction` 协议、默认实体 LF、Lv2 多 prompt LLM LF、官方 Snorkel one-vs-rest LabelModel 融合、状态分层、conflict 输出、recall report 和 LFAnalysis 诊断。
- Lv2 默认 LLM Prompt LF 包括 `type_boundary`、`evidence_support`、`schema_contrast`，提示词由 `weak_supervision/common/llm_prompt_registry.py` 集中维护。
- Lv2 使用每个实体候选的 `entity x LF` 官方投票矩阵；为了保留同一候选跨类型冲突和 top2 gap，当前按 label 做 one-vs-rest LabelModel，再归一化比较各类型概率。
- 默认阈值：accepted >= 0.65；review 为 0.40-0.65 或 top2 gap < 0.15；rejected 只用于低概率或硬失败。
- 硬失败包括空名称、空证据、证据无法回链 chunk 和 schema 外类型。

## 维护约束

- Lv2 输入必须来自 `entity_extraction` 生成的 `entity_base`，不要绕过 Lv1 约束直接从 chunk 文本生成最终节点。
- Lv2 输出字段必须保留候选实体来源、chunk/document 信息、证据、LF trace、置信度和最终状态，便于追溯。
- 具体 LF 实现前，应先明确标签集合与 `common/graph_schema.py`、`configs/weak_supervision.yaml` 的一致性。
- LLM 型 LF 必须支持禁用、失败和不确定时 abstain；同一候选实体应优先使用 `apply_all` 一次性获取所有 label 的 LLM 输出，避免每个 label 重复调用。
- 为保护召回，低置信和类型冲突样本优先进入 review，不能静默丢弃。
- Lv2 主结果应以官方 Snorkel LabelModel 概率为准；本地 softmax 只作为 fallback/debug。

## 开发修改日志

- 2026-05-19：创建 Lv2 包，用于两层 Snorkel 结构中的候选实体类型判定。
- 2026-05-21：将维护文档翻译并同步为中文，修正当前状态为“基础骨架已建，具体 LF/融合/builder 待实现”。
- 2026-06-04：实现 Lv2 实体准入、状态分层、冲突记录和 recall report；属性抽取与 `entity_nodes` 构建改由 `entity_extraction/entity_property_extractor.py` 在 Lv2 后完成。
- 2026-06-04：Lv2 接入官方 Snorkel one-vs-rest LabelModel 融合，并在 recall report 中输出 `official_snorkel_fusion` 诊断。
- 2026-06-04：`EntityPromptedLLMLF` 从占位 abstain 改为实际多 prompt LLM LF，并接入 Lv2 默认 LF 列表参与官方 Snorkel 融合。
- 2026-06-05：`EntityPromptedLLMLF` 新增 `apply_batch`，默认一次请求 20 个 entity；`entity_signal_builder.py` 在逐实体融合前预批量运行 LLM LF，保持 LF trace 和官方 Snorkel 矩阵不变。
- 2026-06-06：增强 Lv2 确定性 LF 的结构型规则，覆盖病因/诱因、治疗方案、治疗原则和发病机制；新增 LLM 三 prompt 候选类型保护，三条 LLM LF 均支持候选类型且无 LLM 强冲突时保留候选类型并 accepted。
