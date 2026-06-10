# weak_supervision 目录开发维护文档

## 目录职责

本目录是两层 Snorkel 弱监督流水线的共享基础层，维护标签、证据、LF 输出、投票和图谱 Schema 等通用结构。这里不再承载具体的 chunk 级或 entity 级业务 Labeling Function。

具体业务逻辑应放在：

- `src/weak_supervision_lv1/`：chunk 级标签存在性与数量预测。
- `src/entity_extraction/`：基于 Lv1 结果约束 Teacher LLM，抽取候选实体 `entity_base`。
- `src/weak_supervision_lv2/`：候选实体到最终实体类型节点的 Lv2 判定。

## 文件与功能说明

- `__init__.py`：共享弱监督包初始化文件。
- `constants.py`：旧单层弱监督常量兼容文件。
- `lf_base.py`、`lf_applier.py`、`label_matrix.py`、`weak_signal_builder.py`、`weak_signal_analyzer.py`：旧单层弱监督骨架/兼容入口，不应继续扩展新业务。
- `lfs/`：旧单层 LF 兼容目录，新业务 LF 请放入 Lv1/Lv2 对应目录。
- `common/labels.py`：共享实体标签、投票标签常量和 Lv1 active labels。
- `common/evidence.py`：证据 span 构造辅助函数。
- `common/lf_output.py`：统一的 `EvidenceSpan` 与 `LFOutput` 数据结构。
- `common/voting.py`：可复用投票分数与 sigmoid 工具。
- `common/metrics.py`：共享弱监督指标占位模块。
- `common/model_selection.py`：投票模型选择占位模块。
- `common/graph_schema.py`：疾病知识图谱实体与关系的规范枚举。
- `common/official_snorkel_runner.py`：官方 Snorkel 适配层，使用 `LabelingFunction`、`PandasLFApplier`、`LFAnalysis`、`MajorityLabelVoter` 和 `LabelModel` 生成主融合概率，同时保留本地 `LFOutput` 旁路 trace。
- `common/llm_prompt_registry.py`：两层 Snorkel LLM Prompt LF 的集中维护模块，定义 Lv1 chunk prompt 与 Lv2 entity prompt 的默认多 prompt 组合。

## 已实现功能

- 已从旧单层弱监督骨架中拆出共享结构，供 Lv1、entity_extraction 和 Lv2 复用。
- 已统一 `LFOutput` 结构，用于记录 LF 名称、标签、投票、置信度、证据、计数与原因。
- 已接入官方 Snorkel 主融合适配层；Lv1 使用 one-vs-rest binary label matrix，Lv2 使用 one-vs-rest entity-label matrix 以保留 top2 gap 和冲突灰区。
- 已将 Lv1/Lv2 的 LLM Prompt LF 默认提示词集中到 Python registry，避免提示词分散在脚本或 YAML 中。
- 已维护 Graph_shema_v1.0 最终版图谱实体集合：`diseases`、`sub_diseases`、`symptoms`、`tests`、`treatments`、`plans`、`etiologies`、`pathogeneses`。
- 已将 Lv1 active labels 同步为 `sub_diseases`、`symptoms`、`tests`、`treatments`、`plans`、`etiologies`、`pathogeneses`。

## 维护约束

- 新增通用字段时，必须同步检查 Lv1、Lv2、Schema 和测试中对 `LFOutput` / 标签常量的使用。
- 新的业务 LF 不要加入本目录或 `weak_supervision/lfs/`，避免重新混成单层架构。
- 旧兼容文件只做必要的导入兼容和迁移辅助，不承载新的流水线状态。
- 官方 Snorkel label matrix 只承载投票；evidence、count、span 和 metadata 必须继续通过 `LFOutput` 旁路记录并写入 raw trace。

## 开发修改日志

- 2026-05-19：明确 `weak_supervision` 调整为共享基础层，并新增 `common` 子包。
- 2026-05-20：新增疾病图谱标准结构，覆盖 Disease/Sub_disease/Symptom/Test/Treatment/Plan/Method/Etiology/Pathogenesis 实体与关系。
- 2026-05-21：将维护文档翻译并补充为中文，补齐共享层职责、文件现状与维护约束。
- 2026-06-03：按 `code/data/Graph_Design/Graph_shema_v1.0.docx` 同步最终版实体、关系及属性，移除 Method 图谱节点并启用 Etiology/Pathogenesis 抽取标签。
- 2026-06-04：新增 `official_snorkel_runner.py`，将官方 Snorkel `PandasLFApplier/LFAnalysis/MajorityLabelVoter/LabelModel` 接为 Lv1/Lv2 主融合路径。
- 2026-06-04：新增 `llm_prompt_registry.py`，统一维护 `semantic_presence/evidence_anchor/boundary_count` 与 `type_boundary/evidence_support/schema_contrast` 两层 LLM Prompt LF。
- 2026-06-05：Lv1/Lv2 LLM Prompt LF 新增批量请求能力，默认 Lv1 每批 10 个 chunk、Lv2 每批 20 个 entity，批内按稳定 ID 回填 LFOutput，缺项可单条重试。
