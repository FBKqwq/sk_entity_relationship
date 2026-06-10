# pdf_pipeline 目录开发维护文档

## 目录职责

实现 PDF 到 chunk 的核心流水线，包括 PDF 读取、页清洗、章节识别、语义切分、表格与图片接口、chunk 校验。

## 文件与功能说明

- `__init__.py`：包初始化文件。
- `chunk_validator.py`：维护 chunk 必填字段、页码范围、text_span、长度、短 chunk 警告和噪声残留校验。
- `english_translator_stub.py`：维护英文段落检测、翻译模型调用和禁用时安全回退接口。
- `image_caption_stub.py`：维护图片/图表识别模型调用、本地图片 data URL 转换、PDF 页面图片 OCR、OCR 结果注入和禁用时安全回退接口。
- `page_cleaner.py`：维护页文本清洗、空白归一、医学期刊噪声过滤和后置无效段落截断；后置截断覆盖 `执笔`、`编写组成员`、撰写/制定组名单和参考文献标题的常见 OCR 变体。
- `pdf_reader.py`：维护 pdfplumber 读取、双栏检测、word 行聚类与页文本解析。
- `section_detector.py`：维护章节标题识别与章节范围生成。
- `semantic_chunker.py`：维护章节优先的 chunk 切分、短父章节并入子章节、span 映射和 chunk payload 构建。
- `semantic_section_llm.py`：在规则章节识别之前调用 LLM，仅返回 `noise_spans` / `translation_spans`（字符区间），本地套用删除与替换后写回 `pages[*].text`；摘要写入 chunk JSON 根字段 `semantic_section_llm_report`（`mode=span_ops_v1`）。维护页分隔符 forbidden spans、模型输出长度/字段契约校验和失败原因记录。
- `text_quality.py`：维护 PDF 解析后、chunk 拆分前的全文质量检测；当抽取文本极少或呈现低中文/低可读字符/高异常符号特征时，供主流程跳过该 PDF。
- `toc_llm_pipeline.py`：维护双层目录解析实验流程；规则目录解析和第一层 LLM 目录解析并列产生候选，第二层 LLM 可选仲裁，最终由本地原文定位、标题形态、密度和层级质量评分决定可写入 chunk 的目录结构。
- `linked_object_ocr.py`：在图/表与 chunk 关联之后，对已关联对象按 bbox 裁剪并调用视觉模型，生成解析记录；可将 `[FIGURE_CONTENT]` 追加到对象所在页的 `pages[*].text`（不重算 chunk）。
- `opendataloader_reader.py`：维护 OpenDataLoader PDF 预解析适配器，将元素级 JSON 转换为本项目兼容的结构对象；主流程只将其作为结构层使用，正文读取和双栏修复仍由 `pdf_reader.py` 完成。
- `table_extractor.py`：维护表格提取和文本化接口。
- `unit_exponent_ocr.py`：维护单位指数 OCR 校正层，对疑似缺失指数的单位表达进行局部裁剪 OCR、指数归一化和修正 trace 记录。

## 已实现功能

- 已复用旧 notebook 的 pdfplumber 读取、双栏检测、word 行聚类、页眉页脚区域过滤逻辑。
- 已实现零宽字符清理、空白归一、医学期刊页眉页脚与 DOI 噪声行过滤，并支持清理 `执笔`、`编写组成员`、`诊疗规范撰写组名单`、`参考文献` / `考文献` 等后置无效段落。
- 已实现中文编号、数字编号、括号编号与常见医学章节标题识别，并避免将小数百分比误识别为标题。
- 已在章节优先切分上支持段落回退与句末安全切分、全文 text_span 与 page_span 映射；未显式配置 `min_section_merge_chars` 时，短 chunk 按 `chunking.min_chars` 与同章节路径层级向相邻块合并（仍受 `max_chars` 约束），短父章节标题可并入紧随其后的子章节，避免生成标题孤块。
- 已支持 `semantic_section_llm`：在 `detect_sections` 之前对拼接页文本调用 LLM，仅取噪声删除区间与英文译文替换区间，由程序在本地拼接；禁用或失败时保持原文。prompt 会显式提供页分隔符禁止区间；若模型返回超长内容、`cleaned_text`、额外字段或噪声删除区间疑似从正文词中间开始/结束，会提前失败并记录原因，避免误套用回显正文、页分隔符冲突区间或局部截断正文。
- 已支持 `skip_if_garbled_text`：PDF 正文读取完成后先检查全文文本质量，若解析结果为空/极短或呈现乱码特征，则写出 `skipped=true` 的空 chunk 标记、validation 失败报告和 heading report，并跳过 semantic section、chunk、图表和 OCR 后续处理。
- 已支持 `figure_table_ocr.enabled`：在关联完成后对关联图/表按需 OCR，写出 `{stem}.figure_ocr.json`；可按 `inject_into_page_text`（或兼容旧键 `inject_into_chunk_text`）将 `[FIGURE_CONTENT]` 追加到对应页的 `pages[*].text`，**不**修改已生成的 chunk。
- 已实现 chunk 必填字段、页码范围、text_span、空文本、长度与残留噪声检查；短 chunk 记录为 warning，不阻塞合法结构输出。
- 已实现 chunk 级语言去噪：当 `chunking.chunk_language_filter.enabled=true` 时，长度达到阈值且中文字符占比低于阈值的 chunk 会被丢弃，并在 payload 的 `dropped_chunks` 中保留摘要。
- 已提供表格扩展接口；英文翻译和图片/图表识别已接入主流程，配置启用后调用模型，禁用或缺少 API Key 时安全回退为 stub。

## 待开发事项：图表定位、关联与内容理解分层

### 背景与目标

当前 `image_caption_stub.py` 已支持对 PDF 页面图片调用 OCR/VL 模型，并将识别结果以 `[FIGURE_CONTENT]` 形式注入页文本。但后续 Snorkel 弱监督实体抽取需要更细粒度地区分两件事：

1. **图表定位与 chunk 关联**：判断 PDF 中的图、表、表题、表注分别位于何处，以及它们应关联到哪个 chunk。
2. **图表内容理解与实体化**：对已经关联到 chunk 的图表调用 OCR/VL，将图表内容转成文本、结构化表格或可参与实体抽取的 evidence。

前者属于 PDF 版面结构问题，应优先用 `pdfplumber` / `PyMuPDF` 等 PDF 库完成；后者属于图表内容理解问题，才需要 OCR 或多模态大模型。

本阶段目标是：**不修改原始 PDF，不把所有图表内容无差别注入正文，而是先用 PDF 库发现并关联图表，再只对关联图表按需 OCR，并将结果作为 chunk 的附加证据字段保存。**

### 需要新增/调整的文件

建议新增：

- `figure_table_linker.py`：维护图表对象发现、caption/note 合并、chunk 与图表关联逻辑。
- `layout_regions.py`：维护页面文本行 bbox、chunk page region 计算、bbox 距离与重叠计算。
- `table_object_builder.py`：维护表格主体、表题、表注合并为完整 table object 的逻辑。

建议调整：

- `pdf_reader.py`：在读取文本时可选保留 word/line bbox，供 chunk 与图表位置关联使用。
- `table_extractor.py`：从单纯“提取表格文本”扩展为“提取表格主体 bbox 与二维表格内容”。
- `image_caption_stub.py`：从“全量页面图片 OCR”调整为“只对 linked figure/table 对象按需 OCR”。
- `semantic_chunker.py`：为 chunk payload 增加 `page_regions`、`linked_figures`、`linked_tables` 等字段。
- `chunk_validator.py`：补充 linked 图表字段的轻量校验。

### 核心数据结构

#### 页面文本行区域

`pdf_reader.py` 后续应支持输出页面行级布局信息：

```json
{
  "page_number": 3,
  "lines": [
    {
      "text": "白塞病评分系统见表1。",
      "bbox": [72.0, 120.5, 520.0, 138.2],
      "start_offset": 1024,
      "end_offset": 1036
    }
  ]
}
```

其中：

- `bbox` 使用 PDF 页面坐标 `[x0, top, x1, bottom]`。
- `start_offset` / `end_offset` 是该行在页文本中的字符范围，用于将 chunk 的 `page_span` 映射到页面区域。

#### 完整表格对象

不能只把 `page.extract_tables()` 返回的二维数组视为表格。医学 PDF 中常见表格由 **表题 + 表格主体 + 表下注释** 组成，例如：

```text
表 1 2014 年白塞病国际研究小组对白塞病诊断/分类标准修订后提出的白塞病评分系统
症状/体征 | 评分（分）
...
注：针刺试验是可选项...
```

因此应构建完整 `table_object`：

```json
{
  "object_id": "T_p03_001",
  "object_type": "table",
  "page": 3,
  "caption": "表1 2014年白塞病国际研究小组对白塞病诊断/分类标准修订后提出的白塞病评分系统",
  "caption_bbox": [70, 80, 520, 112],
  "body": [
    ["症状/体征", "评分（分）"],
    ["眼部病变（前葡萄膜炎，后葡萄膜炎，视网膜血管炎）", "2"]
  ],
  "body_bbox": [70, 116, 520, 330],
  "note": "注：针刺试验是可选项，主要评分系统不包括针刺试验...",
  "note_bbox": [70, 335, 520, 390],
  "bbox": [70, 80, 520, 390],
  "source": "pdfplumber",
  "ocr_status": "not_required"
}
```

#### 图片/图对象

图片对象来自 `page.images` 或 PyMuPDF image blocks：

```json
{
  "object_id": "F_p04_001",
  "object_type": "figure",
  "page": 4,
  "caption": "图1 白塞病发病机制示意图",
  "caption_bbox": [70, 420, 520, 450],
  "bbox": [70, 180, 520, 415],
  "source": "pdf_image",
  "ocr_status": "pending"
}
```

#### chunk 图表关联字段

`chunk` 应增加：

```json
{
  "page_regions": [
    {
      "page": 3,
      "bbox": [70, 120, 520, 260]
    }
  ],
  "linked_tables": [
    {
      "object_id": "T_p03_001",
      "page": 3,
      "caption": "表1 2014年白塞病国际研究小组...",
      "bbox": [70, 80, 520, 390],
      "link_reason": "explicit_reference",
      "link_confidence": 0.95,
      "ocr_status": "pending"
    }
  ],
  "linked_figures": []
}
```

### 具体实现步骤

#### 步骤 1：保留页面布局信息

在 `pdf_reader.py` 中新增可选参数：

```python
def read_pdf_pages(
    pdf_path: str | Path,
    *,
    enable_double_column: bool = True,
    clean_text: bool = True,
    return_layout: bool = False,
) -> list[dict[str, Any]]:
    """使用 pdfplumber 按页读取 PDF 文本，可选返回行级 bbox 布局信息。"""
```

当 `return_layout=True` 时，每页 `meta` 或新增字段中保存：

```json
{
  "layout_lines": [
    {"text": "...", "bbox": [x0, top, x1, bottom], "start_offset": 0, "end_offset": 12}
  ]
}
```

注意：

- 双栏重排后仍需保留行 bbox。
- 清洗文本会改变字符 offset，因此第一版可先基于清洗前行文本与 chunk 文本做近似映射；后续再优化严格 offset 对齐。

#### 步骤 2：从 chunk 映射到页面区域

新增 `layout_regions.py`：

```python
def build_chunk_page_regions(
    chunk: dict,
    page_layouts: dict[int, list[dict]],
) -> list[dict]:
    """根据 chunk 的 page_span/text_span 与页面行布局，估计 chunk 在页面上的 bbox 区域。"""
```

第一版策略：

- 找出 chunk 覆盖页。
- 在对应页中找到与 chunk 文本片段重叠的行。
- 将这些行 bbox 合并为该 chunk 在该页的 `bbox`。

辅助函数：

```python
def union_bboxes(bboxes: list[list[float]]) -> list[float]:
    """合并多个 bbox。"""


def bbox_vertical_distance(a: list[float], b: list[float]) -> float:
    """计算两个 bbox 的垂直距离。"""


def bbox_horizontal_overlap_ratio(a: list[float], b: list[float]) -> float:
    """计算两个 bbox 的横向重叠比例。"""
```

#### 步骤 3：提取表格主体

在 `table_extractor.py` 中补充：

```python
def extract_table_bodies(pdf_path: str | Path) -> list[dict]:
    """使用 pdfplumber find_tables/extract_tables 提取表格主体 bbox 与二维内容。"""
```

返回：

```json
{
  "page": 3,
  "body_bbox": [70, 116, 520, 330],
  "body": [["症状/体征", "评分（分）"], ["眼部病变", "2"]],
  "source": "pdfplumber_find_tables"
}
```

注意：

- 如果表格是扫描图片，`find_tables()` 可能失败。
- 失败时应继续尝试从 `page.images` 或文本 caption 推断图表对象。
- 不能把表格提取失败视为 pipeline hard error。

#### 步骤 4：识别 caption 与 note

新增 `table_object_builder.py`：

```python
def extract_caption_candidates(page_layout: list[dict]) -> list[dict]:
    """识别以 表1/表 1/图1/图 1/Table 1/Figure 1 开头的 caption 行。"""


def extract_note_candidates(page_layout: list[dict]) -> list[dict]:
    """识别以 注：/注:/Note: 开头的表注或图注说明行。"""
```

caption 规则：

```text
表\s*\d+
图\s*\d+
Table\s*\d+
Figure\s*\d+
```

note 规则：

```text
注：
注:
注释：
Note:
Notes:
```

需要支持多行 caption/note 合并：

```python
def merge_nearby_caption_lines(
    caption_line: dict,
    page_layout: list[dict],
    max_gap: float = 18.0,
) -> dict:
    """合并表题/图题的续行。"""


def merge_note_lines(
    note_line: dict,
    page_layout: list[dict],
    max_gap: float = 18.0,
) -> dict:
    """合并表注/图注续行。"""
```

#### 步骤 5：构建完整 table object

```python
def build_table_objects(
    table_bodies: list[dict],
    caption_candidates: list[dict],
    note_candidates: list[dict],
    config: dict,
) -> list[dict]:
    """将表格主体、表题和表注合并为完整 table object。"""
```

合并规则：

- 在 `body_bbox` 上方 `caption_search_distance` 内找最近 caption。
- 在 `body_bbox` 下方 `note_search_distance` 内找最近 note。
- caption、body、note 的外接矩形作为完整 `bbox`。
- caption 匹配成功时从 caption 中解析 `table_number`，如 `表1`。

默认阈值：

```yaml
figure_table_linking:
  caption_search_distance: 90
  note_search_distance: 120
```

#### 步骤 6：构建 figure object

新增：

```python
def extract_figure_objects(
    pdf_path: str | Path,
    page_layouts: dict[int, list[dict]],
    config: dict,
) -> list[dict]:
    """从 page.images 或 PyMuPDF image blocks 构建 figure object，并匹配图题。"""
```

处理逻辑：

- 从 `page.images` 获取图片 bbox。
- 在图片上方或下方一定距离内寻找 `图1` / `Figure 1` caption。
- 合并图片 bbox 与 caption bbox。

#### 步骤 7：显式引用关联

新增 `figure_table_linker.py`：

```python
def find_explicit_figure_table_refs(text: str) -> list[dict]:
    """从 chunk 文本中识别 表1、见表2、图3、如图1所示等显式引用。"""
```

规则：

```text
表\s*\d+
图\s*\d+
Table\s*\d+
Figure\s*\d+
```

关联函数：

```python
def link_by_explicit_reference(
    chunk: dict,
    objects: list[dict],
) -> list[dict]:
    """根据 chunk 文本中的显式图表编号关联对象。"""
```

置信度：

```text
explicit_reference = 0.95
```

#### 步骤 8：版面贴近关联

```python
def link_by_layout_proximity(
    chunk_regions: list[dict],
    objects: list[dict],
    config: dict,
) -> list[dict]:
    """根据 chunk bbox 与图表 object bbox 的版面距离建立弱关联。"""
```

默认规则：

```text
同页
垂直距离 <= 120 pt
横向重叠比例 >= 0.30
```

置信度：

```text
layout_proximity = 0.65
caption_nearby = 0.85
```

注意：

- 版面贴近关联只作为弱关联。
- 如果同页有多个对象，应选择距离最近且横向重叠最高的对象。
- 显式引用优先级高于版面贴近。

#### 步骤 9：写回 chunk payload

```python
def attach_figure_table_links(
    chunk_payload: dict,
    page_layouts: dict[int, list[dict]],
    figure_table_objects: list[dict],
    config: dict,
) -> dict:
    """为每个 chunk 添加 page_regions、linked_figures、linked_tables。"""
```

写回策略：

- `object_type=table` 写入 `linked_tables`。
- `object_type=figure` 写入 `linked_figures`。
- 同一对象因多个规则命中时去重，保留最高置信度和所有 `link_reasons`。

#### 步骤 10：仅对关联图表按需 OCR/VL

调整 `image_caption_stub.py` 的调用策略：

```python
def recognize_linked_objects(
    pdf_path: str | Path,
    linked_objects: list[dict],
    config_path: str | Path,
) -> dict[str, dict]:
    """只对 linked figure/table 对象裁剪并调用 OCR/VL。"""
```

原则：

- 不再默认对每页所有图片 OCR。
- 优先 OCR 与 chunk 已关联的图表。
- OCR 结果不直接混入 chunk 正文，而是写入 linked object 的 `ocr_text` / `table_struct` / `entities` 字段。

示例：

```json
{
  "object_id": "T_p03_001",
  "ocr_status": "done",
  "ocr_text": "症状/体征 评分（分）...",
  "table_struct": {
    "columns": ["症状/体征", "评分（分）"],
    "rows": []
  }
}
```

### 配置项建议

在 `configs/pdf_pipeline.yaml` 中新增：

```yaml
figure_table_linking:
  enabled: true
  return_layout: true
  link_by_explicit_reference: true
  link_by_layout_proximity: true
  caption_search_distance: 90
  note_search_distance: 120
  proximity_vertical_threshold: 120
  proximity_horizontal_overlap_threshold: 0.30
  max_linked_objects_per_chunk: 3

figure_table_ocr:
  enabled: false
  only_linked_objects: true
  inject_into_page_text: false
  store_as_evidence: true
```

### 验收标准

最小可用版本需满足：

- 能从 PDF 页面中提取 `page.images` 作为 figure object。
- 能使用 `pdfplumber.find_tables()` 提取文本表格主体 bbox 与二维内容。
- 能识别 `表1`、`图1`、`注：` 等 caption/note 行。
- 能将表题、表格主体、表注合并为完整 table object。
- 能根据 chunk 文本中的显式 `表1/图1` 关联图表。
- 能根据 bbox 距离将相邻图表弱关联到 chunk。
- 能将 `linked_tables` / `linked_figures` 写入 chunk JSON。
- 默认不修改原始 PDF，不将所有 OCR 内容直接注入正文。
- 参考文献、作者名单等无效文本段过滤逻辑继续复用，不因图表关联而回流进入 chunk。

### 测试计划

建议新增测试：

- `tests/test_figure_table_caption.py`：测试 `表1`、多行表题、`注：` 表注识别。
- `tests/test_table_object_builder.py`：测试表题 + 表体 + 表注合并为完整 table object。
- `tests/test_figure_table_linker.py`：测试显式引用关联和版面贴近关联。
- `tests/test_chunk_with_linked_objects.py`：测试 chunk payload 中 `linked_tables` / `linked_figures` 字段。

测试 fixture 应至少覆盖：

- chunk 文本明确包含“见表1”。
- chunk 文本不提表，但表格与 chunk 在同页且紧邻。
- 表格有多行标题。
- 表格有表下注释。
- `extract_tables()` 失败但 `page.images` 存在的扫描表格。

## 开发修改日志

- 2026-05-07：创建本目录 DEV.md，用于记录目录职责、文件功能、已实现能力与后续修改历史。
- 2026-05-08：修正数字章节识别规则，避免 `51.7%` 等百分比被误切为章节；调整短 chunk 校验为 warning，并用于跑通《白塞综合征诊疗规范》PDF 切分。
- 2026-05-08：升级英文翻译与图片/图表识别模块，分别接入 `TR_model_name` 和 `OCR_model_name`，并保留禁用时 stub 回退。
- 2026-05-08：新增 PDF 页面图片 OCR 与 `[FIGURE_CONTENT]` 注入能力，并在主流程中接入翻译和 OCR 增强。
- 2026-05-08：新增后置非知识段落截断逻辑，在章节识别前清理 `执笔`、`诊疗规范撰写组名单`、`参考文献` 等内容。
- 2026-05-12：补充图表定位、caption/note 合并、chunk 图表关联与按需 OCR/VL 的后续开发方案；明确图表定位由 PDF 库完成，图表内容理解由 OCR/VL 完成。
- 2026-05-14：实现第一版图表关联流程。新增 `layout_regions.py`、`table_object_builder.py` 和 `figure_table_linker.py`；扩展 `pdf_reader.py`，支持可选返回行级 bbox layout 和图片 bbox；扩展 `table_extractor.py`，支持返回表格主体 bbox；扩展 `chunk_validator.py`，校验 linked object 字段。
- 2026-05-21：强化 `semantic_section_llm` 区间式复筛契约，在 prompt 中注入页分隔符 forbidden spans，并新增超长输出、`cleaned_text`、额外字段的本地拒绝逻辑，便于定位模型未按“仅输出区间 JSON”返回的问题。
- 2026-05-21：增强后置非知识段落兜底过滤，新增 `编写组成员` 和 `考文献` 等参考文献 OCR 变体；同时要求 LLM `noise_spans` 删除区间落在安全空白边界，防止模型从有效正文词中间截断。
- 2026-05-21：调整 semantic chunk 合并规则，允许过短父章节标题并入紧随其后的子章节，避免 `四、...` 这类章节标题被单独切成极短 chunk。
- 2026-05-21：修复急性胰腺炎指南 PDF 中基线错位数字/英文 token 被拆到下一行的问题；增强《中国实用外科杂志》页眉页码过滤，并新增本地 Tesseract 兜底抽取缺失的主章节标题与推荐意见。
- 2026-05-22：新增无框文本表格兜底识别，在 reader 阶段从行级布局中提取文本表格 bbox/body 并过滤出 chunk 正文，随后作为 `figure_table_objects` 写入 chunk payload。
- 2026-05-22：调整 OCR 推荐意见补回策略，按推荐编号插入到对应小节边界前；`推荐1` 插入 `1.4` 前并统一中文标点，避免被追加到下一 chunk 且避免主章节标题混入推荐正文。
- 2026-05-22：新增首页题录 front matter 清理，检测英文题名、Keywords、中图分类号等噪声标记后，从正文定义句开始保留内容，避免标题、作者、英文题名和关键词进入 CH0001。
- 2026-05-22：增强后置非知识段落截断，支持行内断开的 `（按姓氏汉语拼音排序）：` 作者名单锚点，并补充利益冲突声明与 `［1］` 引用条目兜底，避免作者名单和参考文献生成 CH0014 以后的噪声 chunk。
- 2026-05-25：新增 OpenDataLoader PDF 预解析层，支持通过 `pdf.parser=opendataloader` 将元素级 JSON 接入现有 chunk、图表关联和校验流程。
- 2026-05-25：调整 OpenDataLoader 接入方式为结构层融合；chunk 正文继续使用 `pdfplumber` 双栏读取结果，避免 OpenDataLoader 阅读顺序错位导致 chunk 双栏混排。
- 2026-05-25：修复 OpenDataLoader CLI 在 Windows 中文 PDF 路径下找不到文件的问题；清理 CLI 路径首尾空白/换行，调用外部程序前复制为临时 ASCII 文件名，并将输出复制回项目目录。
- 2026-06-04：`pdf->chunk` 接入 `pdf.isTable` 总开关；`false` 时只读正文文本，跳过图表结构预解析、图表对象关联和关联对象 OCR。
- 2026-06-04：新增 `text_quality.py` 与 `pdf.skip_if_garbled_text`，在 chunk 拆分前拦截疑似加密/编码导致的全文乱码 PDF。
- 2026-06-05：新增 `toc_llm_pipeline.py` 双层目录解析实验开关；支持规则目录与第一层 LLM 目录并列候选、第二层 LLM 可选仲裁、本地硬校验和 TOC 质量评分，并在 chunk payload 写入 `toc_llm_report`。
- 2026-06-06：增强双层目录解析与后置清理的参考文献过滤，拒绝 `１０参考文献` / 编号 `References` 等带编号参考文献标题，并在 chunk 前截断这类非知识尾部。
- 2026-06-06：新增 chunk 级低中文占比过滤配置 `chunking.chunk_language_filter`，默认丢弃长度不少于 80 且中文字符占比低于 20% 的 chunk，并记录 `dropped_chunks` 审计摘要。
- 2026-06-06：修正短 chunk 合并策略；不再合并顶级兄弟章节，避免 `三、发病机制` 错合并 `四、病理改变` 后沿用错误章节路径；同一父章节下的短子章节可合并到共同父章节路径。
- 2026-06-06：增强后置尾部截断，支持从行内 `(收稿日期...)`、`(本文编辑...)` 截断期刊编辑信息与后续投稿广告，避免尾部噪声导致 chunk validation 失败。
- 2026-05-25：新增单位指数 OCR 校正层，针对 `10 cfu/ml` 等疑似丢失上标的单位表达裁剪局部图像 OCR，并将可信结果规范为 `10^n`。
