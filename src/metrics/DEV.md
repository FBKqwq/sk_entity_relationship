# metrics 目录开发维护文档

## 目录职责

维护 LF 指标、弱监督信号指标与 metrics.json 导出逻辑。

## 文件与功能说明

- `__init__.py`：包初始化文件。
- `lf_metrics.py`：维护 LF Coverage、Abstain Rate、Overlap、Conflict Rate 等指标。
- `weak_signal_metrics.py`：维护实体分布、Evidence 命中率、Schema 通过率和章节维度指标。
- `metrics_exporter.py`：维护 metrics.json 导出逻辑。

## 已实现功能

- 已创建指标模块文件骨架。
- 当前指标统计逻辑尚未实现。

## 开发修改日志

- 2026-05-07：创建本目录 DEV.md，用于记录 metrics 目录职责、文件功能和后续修改历史。
