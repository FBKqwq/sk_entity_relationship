# weak_supervision_lv1 目录开发维护文档

## 目录职责

维护 Snorkel Lv1：以 chunk 为单位判断各类实体标签是否存在，并预测对应标签在 chunk 中的候选实体数量。

Lv1 的输入是 PDF pipeline 生成的 chunk JSON；输出是面向后续 Teacher LLM 候选实体抽取的 `chunk_label_result.jsonl`，以及可追溯的 `lv1_lf_outputs.jsonl`。

## 文件与功能说明

- `__init__.py`：Lv1 包初始化文件。
- `chunk_lf_base.py`：chunk 级 LF 抽象基类，约定 `apply(chunk, label) -> LFOutput`。
- `chunk_lf_applier.py`：将所有 chunk LF 应用到所有 chunk-label 组合，并支持 `apply_all` 聚合调用。
- `chunk_label_matrix.py`：构建 `chunk x label x LF` 的 vote/count/confidence 矩阵。
- `chunk_vote_model.py`：维护本地手工加权二分类投票，当前作为 fallback/debug 对照和 count 辅助，不再是主融合路径。
- `chunk_count_regression.py`：维护数量特征、手工数量融合和缺席标签 count=0 的兜底规则。
- `chunk_signal_builder.py`：融合 Lv1 投票与数量预测，生成 `chunk_label_result.jsonl` 记录。
- `lfs/`：chunk 级章节先验、词典、指标正则、医学语义模式和可选 Teacher LLM LF。

## 已实现功能

- 已建立 Lv1 chunk-label 弱监督主结构。
- 已实现多标签 LF 输出，当前 active labels 为 `sub_diseases`、`symptoms`、`tests`、`treatments`、`plans`、`etiologies`、`pathogeneses`。
- `ChunkMedicalPatternLF` 是主力确定性多标签 LF，负责疾病亚型、症状、检查、治疗和方案等医学语义句式；病因和发病机制当前主要由词典/LLM 信号覆盖。
- `ChunkDictionaryLF` 使用配置词典和保守种子词做字符级匹配，并输出 count 与 evidence。
- `ChunkRegexIndicatorLF` 仅作为 `tests` 辅助信号，避免把时间、比例、病程等泛化数值误判为检查。
- `ChunkSectionPriorLF` 和 `ChunkRegexIndicatorLF` 都是低权重辅助信号。
- `ChunkPromptedLLMLF` 可选调用 Teacher LLM；禁用、失败、JSON 异常或无可信信号时必须 abstain。
- Lv1 Prompted LLM 已支持同一 label 的多 prompt LF 组合，默认包含 `semantic_presence`、`evidence_anchor`、`boundary_count` 三个互补视角，并作为独立 LF 名称进入融合。
- Lv1 默认 LLM prompt 由 `weak_supervision/common/llm_prompt_registry.py` 集中维护；YAML 可覆盖 prompt 列表，但不再是默认提示词唯一来源。
- 已接入官方 Snorkel one-vs-rest 主融合：每个 active label 构建 `chunk x LF` binary label matrix，使用 `PandasLFApplier`、`LFAnalysis`、`MajorityLabelVoter` 和 `LabelModel` 输出主概率；手工加权投票保留为 fallback/debug 对照。
- 已实现标签阈值配置、主力/辅助 LF 权重配置和数量融合模型；数量融合仍使用本地 `LFOutput.count/evidence` 旁路信息。
- `chunk_signal_builder.py` 会把 `present/status/confidence/source_lfs/evidence/predicted_count/count_confidence` 等字段合并进 chunk-label 结果。

## 维护约束

- Lv1 只负责“chunk 是否含有某类实体”和“预计数量”，不要直接生成最终实体节点。
- 新增标签前必须同步 `common/labels.py`、`configs/weak_supervision.yaml`、测试和后续 entity_extraction/Lv2 约束。
- 无信号、配置缺失、LLM 不确定、证据不可用时使用 `ABSTAIN`，不要为了覆盖率强行投票。
- 数量融合必须尊重 `present=False` 时 count=0 的规则。
- 多 prompt LLM LF 必须保持独立 `lf_name` 和 `prompt_name` trace，方便诊断不同 prompt 的覆盖、冲突和证据回链质量。
- Lv1 主结果应以官方 Snorkel LabelModel 概率为准；本地手工投票不得作为正式实验主结果。

## 开发修改日志

- 2026-05-19：创建 Lv1 包，用于两层 Snorkel 结构中的 chunk 级弱监督。
- 2026-05-20：新增多标签 chunk LF 应用能力和具体 Lv1 LF 实现。
- 2026-05-20：将 Lv1 active labels 限定为 `sub_diseases`、`symptoms`、`tests`、`treatments`、`plans`。
- 2026-05-20：新增主力医学语义模式 LF，并将章节/正则 LF 调整为辅助角色。
- 2026-05-20：实现可配置手工加权投票模型和 chunk-label 结果构建器。
- 2026-05-20：实现手工数量融合模型，并把预测数量写入 chunk-label 结果。
- 2026-05-21：补充中文维护文档，按当前 Lv1 实现同步文件职责、输出字段和维护约束。
- 2026-06-03：按 Graph_shema_v1.0 最终版将 Lv1 active labels 扩展到 `etiologies`、`pathogeneses`，Method 不再作为图谱抽取标签。
- 2026-06-04：`ChunkPromptedLLMLF` 支持多 prompt 组合；`02_snorkel_lv1_label_chunks.py` 会按 `lv1_prompted_llm.prompts` 生成多个互补 LLM LF。
- 2026-06-04：Lv1 主入口接入官方 Snorkel `PandasLFApplier/LFAnalysis/MajorityLabelVoter/LabelModel`；每个 label 使用 one-vs-rest 矩阵，输出官方融合概率和 LFAnalysis 诊断。
- 2026-06-04：Lv1 Prompt LF 默认提示词迁移到共享 Python registry，和 Lv2 Prompt LF 使用同一维护入口。
- 2026-06-05：`ChunkPromptedLLMLF` 新增 `apply_batch`，默认一次请求 10 个 chunk；`chunk_lf_applier.py` 优先批量调用后再按 chunk-label 顺序还原输出。
