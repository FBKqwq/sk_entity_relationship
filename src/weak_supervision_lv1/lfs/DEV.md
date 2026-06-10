# weak_supervision_lv1/lfs 目录开发维护文档

## 目录职责

维护 Lv1 chunk 级 Labeling Functions，为 chunk-label 组合提供实体类型存在性、数量和证据信号。

## 文件与功能说明

- `__init__.py`：Lv1 LF 包初始化文件。
- `_chunk_lf_utils.py`：chunk 文本/章节读取、证据构造、配置读取、active labels 解析和 evidence 合并工具。
- `lf_chunk_section.py`：基于章节标题和章节路径的低权重上下文先验。
- `lf_chunk_dictionary.py`：基于配置词典和保守种子词的字符匹配 LF，并输出命中数量。
- `lf_chunk_regex_indicator.py`：检查指标相关正则 LF，仅对 `tests` 标签投辅助票。
- `lf_chunk_medical_pattern.py`：主力确定性多标签医学语义模式 LF。
- `lf_chunk_prompted_llm.py`：可选 Teacher LLM chunk LF；禁用、失败或解析异常时必须 abstain。

## 已实现功能

- `ChunkSectionPriorLF` 根据 section title/path 产生低权重辅助上下文投票。
- `ChunkDictionaryLF` 使用配置词和保守种子词做字符级匹配，记录可定位 evidence span。
- `ChunkRegexIndicatorLF` 已降级为 `tests` 专用辅助正则信号，不再对非检查标签投票。
- `ChunkMedicalPatternLF` 是当前主力确定性 LF，覆盖 `sub_diseases`、`symptoms`、`tests`、`treatments`、`plans`；`etiologies`、`pathogeneses` 当前主要由词典和可选 Teacher LLM 信号覆盖。
- `ChunkPromptedLLMLF` 每个 chunk 可一次性请求 Teacher LLM 评估所有标签，校验 JSON，尽量记录可定位 evidence span；证据无法定位时不再阻断正票，但需要保留原因和低置信处理。
- `_chunk_lf_utils.py` 统一生成 `LFOutput`，并提供 active label 读取和 evidence 去重合并。

## 维护约束

- 新增 LF 时必须使用 `LFOutput`，并明确 `vote`、`confidence`、`count`、`evidence_spans` 与 `reason`。
- 无信号时统一 abstain，不要返回空 evidence 的强阳性。
- `tests` 正则信号应保持保守，避免把病程、比例、评分范围等非检查值误判为检查实体。
- LLM LF 必须遵守配置开关、API Key 缺失回退、超时/失败 abstain 和 JSON 校验。

## 开发修改日志

- 2026-05-19：将 chunk 级 LF 骨架拆入 Lv1 包。
- 2026-05-20：实现章节、词典、指标正则和 Teacher LLM 等具体多标签 Lv1 LF。
- 2026-05-20：将 indicator 命名对齐到图谱中的 `tests` 实体标签。
- 2026-05-20：放宽 Teacher LLM evidence 处理，不再因 evidence 无法定位而阻止正票。
- 2026-05-20：将 Lv1 限定为五个 active labels，并移除正则 LF 对非 tests 标签的旁路投票。
- 2026-05-20：移除通用单位/时间/比例匹配，避免把持续时间和患病率误认为检查。
- 2026-05-20：新增 `ChunkMedicalPatternLF` 作为主力多标签语义 LF，并将章节/正则 LF 降级为辅助信号。
- 2026-05-21：将维护文档翻译并同步为中文，补齐 `_chunk_lf_utils.py` 与当前 LF 分工。
- 2026-06-03：按 Graph_shema_v1.0 最终版扩展 active labels 到 `etiologies`、`pathogeneses`，保留 tests 正则 LF 的专用边界。
