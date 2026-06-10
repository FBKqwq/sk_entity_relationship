# configs 目录开发维护文档

## 目录职责

维护项目级配置，负责 PDF 解析、chunk 切分、弱监督、LLM 与 Schema 路径的参数入口。

## 文件与功能说明

- `llm.yaml`：维护 Teacher LLM 开关、通用模型、翻译模型、图表识别模型、API Base、API Key、API Key 环境变量、超时、重试、通用 deep-thinking 开关、OCR thinking 配置、`03_llm_extract_entity_base.py` 的 `Full_extraction` 默认开关，以及 Lv2 LLM LF 批处理参数。
- `pdf_pipeline.yaml`：维护 PDF 解析、semantic section LLM 复筛、图表关联和 chunk 切分参数。
- `pdf_pipeline.yaml` 中 `pdf.skip_if_garbled_text` 维护 PDF 解析后全文乱码检测阈值；默认开启，命中时跳过该 PDF 的后续 chunk 处理。
- `project.yaml`：维护项目名称、根目录、阶段和随机种子。
- `schema_paths.yaml`：维护 JSON Schema 文件路径。
- `weak_supervision.yaml`：维护实体标签、Lv1 多 prompt LLM LF 配置、LLM LF 批处理参数、融合方法、最低置信度和基础词典。

## 已实现功能

- 已建立项目、PDF pipeline、弱监督、LLM 和 Schema 路径配置。
- PDF pipeline 默认使用 pdfplumber，并启用双栏、表格、英文翻译模型和 OCR 模型增强开关。
- Teacher LLM 已配置通用模型 `model_name`、翻译模型 `TR_model_name`、图表识别模型 `OCR_model_name`；缺少 API Key 或禁用时功能模块会安全回退为 stub。
- `llm.yaml` 中 `teacher_llm.Full_extraction` 当前默认为 `false`，表示 `03_llm_extract_entity_base.py` 默认按 Lv1 阳性/灰区 chunk 抽取候选实体；命令行可用 `--Full_extraction` 临时开启全量抽取。
- `weak_supervision.yaml` 中 `lv1_prompted_llm.prompts` 当前配置了 `semantic_presence`、`evidence_anchor`、`boundary_count` 三个互补 LLM LF，并在 Lv1 vote/count 权重中分别配置。

## 开发修改日志

- 2026-05-07：创建本目录 DEV.md，用于记录配置目录职责、文件功能和后续修改历史。
- 2026-05-08：补充 `llm.yaml` 的三类模型、API Key 环境变量和 OCR thinking 参数说明。
- 2026-05-08：补充 `pdf_pipeline.yaml` 中翻译模型、OCR 模型、LLM 配置路径和每页最大图片识别数量配置。
- 2026-05-14：在 `pdf_pipeline.yaml` 中新增 `figure_table_linking` 与 `figure_table_ocr` 选项；默认启用图表关联，当只请求关联对象 OCR 时跳过整页 OCR 注入。
- 2026-05-20：同步新版医学图谱实体、关系与 active labels 到 `weak_supervision.yaml`，治疗方法节点暂保留但不启用抽取。
- 2026-05-20：补充 Lv1 手工加权投票模型配置，包括标签阈值、LF 权重和主力/辅助 LF 列表。
- 2026-05-20：补充 Lv1 数量预测模型配置，包括标签最大数量、计数 LF 权重和 evidence floor 来源。
- 2026-05-21：在 `pdf_pipeline.yaml` 的 `semantic_section_llm` 中新增 `max_output_chars`，用于限制区间式复筛模型输出长度，提前拦截正文回显、`cleaned_text` 或额外解释。
- 2026-05-25：在 `pdf_pipeline.yaml` 中新增 `pdf.unit_exponent_ocr_recovery`，用于控制单位指数局部 OCR 校正层。
- 2026-06-03：在 `llm.yaml` 中新增 `teacher_llm.Full_extraction`，用于控制 `03_llm_extract_entity_base.py` 是否绕过 Lv1 阳性 chunk 过滤做全量抽取。
- 2026-06-04：为新主管线将 `teacher_llm.Full_extraction` 默认改为 `false`，恢复 `chunk -> Lv1 -> entity_base` 主路径。
- 2026-06-04：`weak_supervision.yaml` 新增 Lv1 多 prompt LLM LF 配置，并为每个 prompt LF 增加独立 vote/count 权重。
- 2026-06-04：`pdf_pipeline.yaml` 新增 `pdf.isTable` 总开关，默认 `true`；设为 `false` 时 `pdf->chunk` 只做文本解析，跳过图表预解析、关联和 OCR。
- 2026-06-04：`pdf_pipeline.yaml` 新增 `pdf.skip_if_garbled_text` 配置，用于在 chunk 拆分前跳过疑似加密/编码乱码 PDF。
- 2026-06-05：`llm.yaml` 与 `weak_supervision.yaml` 新增 `llm_batching`，默认 Lv1 每批 10 个 chunk、Lv2 每批 20 个 entity，并设置批内字符上限与缺项重试开关。
- 2026-06-09：明确 `llm.yaml` 中 `teacher_llm.enable_thinking` 为 OpenAI-compatible `extra_body.enable_thinking` 深度思考开关，默认 `true`。
