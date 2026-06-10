# scripts 目录开发维护文档

## 目录职责

维护可直接执行的流水线入口脚本。

## 文件与功能说明

- `01_build_chunks.py`：从单个 PDF 或 PDF 目录构建 chunk JSON，在主流程中调用翻译和 OCR 模型增强页文本，并输出 validation JSON。
- `01_build_chunks.py` 在 PDF 正文读取完成后、semantic section 和 chunk 拆分前执行全文质量检测；若解析文本疑似乱码，则写出 skipped 空 chunk 标记并停止该 PDF 的后续处理。
- `02_snorkel_lv1_label_chunks.py`：Lv1 chunk-label 弱监督主入口，支持单文件和目录批量模式，应用 chunk 级 LF，并使用官方 Snorkel `PandasLFApplier/LFAnalysis/MajorityLabelVoter/LabelModel` 作为主融合路径，输出 `chunk_label_result.jsonl` 与 `lv1_lf_outputs.jsonl`。
- `03_llm_extract_entity_base.py`：Teacher LLM 候选实体抽取主入口，支持单文件和目录批量模式；默认读取 `llm.yaml` 中 `teacher_llm.Full_extraction`，缺省为 false；输出高召回 `entity_base.jsonl` 与原始追踪 JSONL，`entity_base` 不再承载最终属性。
- `04_snorkel_lv2_label_entities.py`：Lv2 entity 类型判定与 Lv2 后属性抽取入口，使用官方 Snorkel one-vs-rest LabelModel 作为实体类型主融合路径，输出 `entity_label_result.jsonl`、`entity_property_result.jsonl`、`entity_nodes.jsonl`、`entity_conflicts.jsonl` 和 `entity_recall_report.json`。
- `05_export_entity_metrics.py`：Lv1/Lv2/QC 指标与实验报告导出入口骨架；当前文件只有模块说明，具体 CLI 尚未实现。
- `06_llm_extract_relationship_base.py`：Teacher LLM 关系构建入口，支持单文件和目录批量模式；优先从 `entity_nodes + chunks + graph_schema` 生成合法候选关系，默认包含 review 实体保护召回，输出 `relationship_base.jsonl`、候选关系、raw trace 和 summary。
- `07_run_full_pipeline_parallel.py`：阶段级并发主管线入口；按 `01_chunk -> 02_lv1 -> 03_entity_base -> 04_lv2_nodes -> 06_relationships` 逐阶段推进，每个大阶段内允许多个 PDF 同时跑，默认 `--workers 5`，并写出每文档 pipeline log、stage status 与每阶段 summary，便于 `--skip-existing` 断点续跑。

## 已实现功能

- `01_build_chunks.py` 已支持 `--input/--output` 单文件模式。
- `01_build_chunks.py` 已支持 `--input_dir/--output_dir` 批量模式。
- `01_build_chunks.py` 已读取 `configs/pdf_pipeline.yaml` 并调用 PDF reader、翻译模型增强、OCR 模型增强、semantic chunker 和 chunk validator。
- `02_snorkel_lv1_label_chunks.py`、`03_llm_extract_entity_base.py`、`04_snorkel_lv2_label_entities.py` 与 `06_llm_extract_relationship_base.py` 已可执行；`05_export_entity_metrics.py` 目前仍是指标导出占位。

## 开发修改日志

- 2026-05-07：创建本目录 DEV.md，用于记录 scripts 目录职责、文件功能和后续修改历史。
- 2026-05-08：`01_build_chunks.py` 主流程接入 `TR_model_name` 翻译增强和 `OCR_model_name` 图片/图表识别增强。
- 2026-05-14：更新 `01_build_chunks.py`，在 semantic chunk 生成后、validation/output 之前启用版面感知的图表关联。
- 2026-05-19：新增两层 Snorkel 脚本入口骨架，旧 `02_generate_weak_signals.py` 保留为兼容入口。
- 2026-05-21：同步脚本维护文档，明确 `02` 到 `05` 目前仍为模块说明级骨架。
- 2026-05-21：实现 `03_llm_extract_entity_base.py` 主脚本 CLI，统一 entity_base 抽取入口。
- 2026-05-21：实现 `02_snorkel_lv1_label_chunks.py` 主脚本 CLI，统一 Lv1 chunk-label 打标入口。
- 2026-05-25：`01_build_chunks.py` 清理命令行路径参数首尾空白/换行，避免 PowerShell 续行或复制粘贴导致路径携带不可见字符。
- 2026-06-03：`03_llm_extract_entity_base.py` 新增 `Full_extraction` 布尔开关；默认从 `llm.yaml` 读取且当前为 true，为 true 时不要求 Lv1 文件、不按 Lv1 阳性 chunk 过滤，直接对所有输入 chunk 全量抽取。
- 2026-06-03：新增 `06_llm_extract_relationship_base.py`，实现 schema 合法候选关系生成、LLM 关系审核、关系属性抽取和关系输出目录默认路径。
- 2026-06-04：主管线改为 `entity_base -> Lv2 label -> Lv2 后属性抽取 -> entity_nodes -> relationships`；`06` 入口优先消费 `entity_nodes`，支持 `--accepted-only` 关闭 review 关系召回。
- 2026-06-04：`02` 与 `04` 接入官方 Snorkel 主融合路径；本地手工投票/softmax 降级为 fallback/debug，正式结果记录 `official_snorkel_fusion` 诊断。
- 2026-06-04：`01_build_chunks.py` 接入 `pdf.isTable` 总开关；`false` 时强制文本模式，跳过 OpenDataLoader 图表结构层、图表关联、关联对象 OCR 和 figure_ocr 输出。
- 2026-06-04：`01_build_chunks.py` 接入 `pdf.skip_if_garbled_text`，对疑似加密/编码导致的全文乱码解析结果，在 chunk 拆分前跳过并输出 skipped validation。
- 2026-06-09：新增 `07_run_full_pipeline_parallel.py`，支持一次并发处理 3 个 PDF 的完整非入库主管线，并按文档写出独立日志和汇总 summary。
- 2026-06-09：`07_run_full_pipeline_parallel.py` 将各子阶段 stdout/stderr 实时镜像到终端，按 `[document][stage]` 前缀展示，同时继续写入每文档 pipeline log。
- 2026-06-10：`07_run_full_pipeline_parallel.py` 从“单 PDF 完整链路并发”重构为“阶段级 barrier 并发”，先完成所有 PDF 的 chunk，再进入 Lv1、entity_base、Lv2/entity_nodes 和 relationship，每阶段内默认 5 个 PDF 并发，并输出阶段级 summary 与每文档 stage status 支持断点续跑。
