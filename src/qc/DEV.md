# qc 目录开发维护文档

## 目录职责

维护 Schema QC、Evidence QC、Conflict QC 与 QC 报告输出。

## 文件与功能说明

- `__init__.py`：包初始化文件。
- `schema_qc.py`：维护 JSON Schema 校验逻辑。
- `evidence_qc.py`：维护 evidence 非空、原文匹配、span 对齐和自动修复逻辑。
- `conflict_qc.py`：维护多 LF 冲突、同名多类型、章节倾向冲突等检测逻辑。
- `qc_reporter.py`：维护 QC 报告聚合与导出。

## 已实现功能

- 已创建 QC 模块文件骨架。
- 当前 QC 完整逻辑尚未实现，chunk 级基础校验暂位于 `pdf_pipeline/chunk_validator.py`。

## 开发修改日志

- 2026-05-07：创建本目录 DEV.md，用于记录 QC 目录职责、文件功能和后续修改历史。
