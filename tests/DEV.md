# tests 目录开发维护文档

## 目录职责

维护 PDF pipeline、Schema、LF、QC 与弱监督输出的自动化测试。

## 文件与功能说明

- `test_pdf_reader.py`：测试页文本清洗、双栏检测、翻译模型禁用回退、图片识别模型禁用回退和 OCR 结果注入。
- `test_opendataloader_reader.py`：测试 OpenDataLoader 适配层在调用外部 CLI 前使用 ASCII 临时 PDF 路径，并将输出复制回项目目录。
- `test_unit_exponent_ocr.py`：测试单位指数 OCR 校正层，覆盖 `cfu/ml` 指数恢复和导尿标本上下文兜底。
- `test_section_detector.py`：测试章节识别、正文回退与百分比文本不误识别为标题。
- `test_semantic_section_llm_prescreen.py`：测试语义章节 LLM 复筛的页分隔符 forbidden spans、超长输出拦截、`cleaned_text`、额外字段和不安全噪声删除边界拒绝逻辑。
- `test_semantic_chunker.py`：测试 chunk payload、字段、长文本切分、短父章节并入子章节和后置无效段落过滤，包括 `编写组成员` 到参考文献尾部的截断。
- `test_page_schema.py`、`test_chunk_schema.py`、`test_weak_signal_schema.py`：后续维护 Schema 测试。
- `test_lfs.py`、`test_lf_applier.py`、`test_evidence_qc.py`：旧单层弱监督和 QC 测试占位。
- `test_lv1_chunk_lfs.py`、`test_lv1_chunk_vote_model.py`、`test_lv1_count_regression.py`、`test_lv1_signal_builder.py`：维护 Lv1 测试，覆盖确定性 LF、Teacher LLM LF、多 prompt LLM LF 组合、官方 Snorkel 主融合和本地 fallback 输出。
- `test_entity_extraction_prompt.py`、`test_entity_base_builder.py`：后续维护 LLM 初抽取中间层测试。
- `test_relationship_extraction.py`：测试关系候选生成、LLM 审核删除、属性抽取白名单、上下文升级顺序、review 实体召回和 `implements_by` 二级锚点限制。
- `test_lv2_entity_pipeline.py`：测试 Lv2 accepted/review/rejected 状态分层、Lv2 后属性抽取、属性失败不删除实体和 review 属性抽取开关。
- `test_lv2_entity_lfs.py`、`test_lv2_entity_vote_model.py`、`test_lv2_signal_builder.py`：维护 Lv2 测试，覆盖实体 LF、Lv2 多 prompt LLM LF 和实体类型融合。

## 已实现功能

- 已补充 PDF reader、section detector、semantic chunker 的聚焦测试。
- 已补充百分比文本不应被当作数字章节标题的回归测试。
- 已补充翻译模型与图片识别模型在禁用配置下安全返回 stub 的测试。
- 已补充 OCR 结果注入到页文本的测试。
- 已补充 `执笔`、`诊疗规范撰写组名单`、`参考文献` 等后置无效段落不进入 chunk 的测试。
- 已补充 Lv1 多标签 LF 测试，覆盖章节先验、词典匹配、指标正则、Teacher LLM JSON 解析、多 prompt LLM LF 组合和 applier 的 `apply_all` 路径。
- 由于当前环境缺少 pytest，曾通过直接调用测试函数和 `compileall` 做基础验证。
- 其他测试文件仍是骨架，需随弱监督和 Schema 实现继续补充。

## 开发修改日志

- 2026-05-07：创建本目录 DEV.md，用于记录 tests 目录职责、文件功能和后续修改历史。
- 2026-05-08：补充 `test_section_detector.py` 回归测试，覆盖 `51.7%` 等百分比文本不应被识别为章节标题。
- 2026-05-08：补充 `test_pdf_reader.py` 模型禁用回退测试，避免单元测试触发真实 API 调用。
- 2026-05-08：补充 `test_pdf_reader.py` OCR 结果注入测试，覆盖 `[FIGURE_CONTENT]` 写入页文本。
- 2026-05-08：补充 `test_semantic_chunker.py` 后置无效段落过滤测试，覆盖执笔、撰写组名单和参考文献清理。
- 2026-05-14：补充版面行 offset、图表 caption 候选、table object 组装、显式图表关联、chunk page regions 和小图片过滤测试。
- 2026-05-19：补充两层 Snorkel 测试文件规划，区分 Lv1、entity_extraction 和 Lv2。
- 2026-05-20：补充 Lv1 手工加权投票和 chunk-label 结果构建测试。
- 2026-05-20：补充 Lv1 数量融合模型测试，覆盖缺席标签 count=0 和多 LF 计数融合。
- 2026-05-20：补充 entity_extraction 动态 prompt、Teacher prelabel 解析和 entity_base 记录构建测试。
- 2026-05-20：补充 Lv1 chunk LF 多标签行为测试，并通过直接函数调用验证。
- 2026-05-21：新增 `test_semantic_section_llm_prescreen.py`，覆盖页分隔符禁止区间格式化、模型超长输出契约校验、`cleaned_text` 与额外字段拒绝。
- 2026-05-21：补充语义复筛不安全 `noise_spans` 边界拒绝测试，并补充 `编写组成员` 后置尾部截断测试，避免作者名单和参考文献进入 chunk。
- 2026-05-21：补充短父章节标题并入子章节的 semantic chunker 回归测试，覆盖极短父级标题不应单独成块。
- 2026-05-25：新增 OpenDataLoader CLI 路径编码回归测试，覆盖中文 PDF 文件名和误带结尾换行的路径转临时 ASCII 文件名后再调用外部程序的行为。
- 2026-05-25：新增单位指数 OCR 校正测试，覆盖 OCR 证据恢复 `10^5/10^4` 和菌落计数上下文兜底。
- 2026-06-03：更新实体抽取 prompt 测试为 schema 驱动约束，并新增关系抽取测试，覆盖合法关系、拒绝删除、属性字段过滤和 `entity_window -> chunk -> neighbor_chunks -> document` fallback。
- 2026-06-04：补充主管线大改回归测试，覆盖 `entity_base` 不承载最终属性、Lv2 后属性抽取、review 关系召回、accepted-only 过滤和 `implements_by` 依赖已确认 Treatment。
- 2026-06-04：补充 Lv1 多 prompt LLM LF 测试，确认同一 label 的互补 prompt 会以独立 LF 名称进入融合。
- 2026-06-04：补充官方 Snorkel 主融合回归覆盖，当前聚焦测试通过 `PandasLFApplier/LFAnalysis/LabelModel` 的 Lv1/Lv2 接入路径。
- 2026-06-04：补充 Lv2 Prompt LLM LF 测试，确认默认三 prompt LF 已接入并可解析 LLM JSON 投票。
