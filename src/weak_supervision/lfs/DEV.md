# weak_supervision/lfs 目录开发维护文档

## 目录职责

本目录仅保留旧单层弱监督骨架的兼容文件。两层 Snorkel 架构已经拆分完成，新业务 LF 不应继续放在这里。

请使用以下目录维护新逻辑：

- `src/weak_supervision_lv1/lfs/`：chunk 级 Lv1 Labeling Function。
- `src/weak_supervision_lv2/lfs/`：entity 级 Lv2 Labeling Function。

## 文件与功能说明

- `__init__.py`：旧 LF 包初始化文件。
- `lf_section.py`：旧 chunk 章节 LF 占位/兼容文件。
- `lf_dictionary.py`：旧词典 LF 占位/兼容文件。
- `lf_regex_indicator.py`：旧指标正则 LF 占位/兼容文件。
- `lf_medical_pattern.py`：旧医学模式 LF 占位/兼容文件。
- `lf_prompted_llm.py`：旧 prompted LLM LF 占位/兼容文件。

## 当前状态

- 正式 Lv1 LF 已迁移到 `weak_supervision_lv1/lfs/`。
- 正式 Lv2 LF 应在 `weak_supervision_lv2/lfs/` 实现。
- 本目录应保持轻量，直到旧导入路径完全退役。

## 维护约束

- 不要在本目录新增业务 LF 或配置读取逻辑。
- 如果必须保留旧导入路径，应只做兼容转发，并在最近的调用处逐步迁移到 Lv1/Lv2 新目录。

## 开发修改日志

- 2026-05-07：创建原始单层 LF 目录维护说明。
- 2026-05-19：两层 Snorkel 拆分后，将本目录降级为兼容层。
- 2026-05-21：将维护文档翻译并补充为中文，明确禁止新增业务 LF。
