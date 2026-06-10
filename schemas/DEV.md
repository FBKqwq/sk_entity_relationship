# schemas 目录开发维护文档

## 目录职责

维护 page、chunk、LF 输出、weak signal 与 metrics 的 JSON Schema。

## 文件与功能说明

- `page_schema.json`：页级文本结构 Schema。
- `chunk_schema.json`：chunk JSON 结构 Schema。
- `lf_output_schema.json`：Labeling Function 输出结构 Schema。
- `weak_signal_schema.json`：弱监督信号结构 Schema。
- `weak_metrics_schema.json`：弱监督指标结构 Schema。

## 已实现功能

- 已建立所有要求的 Schema 文件。
- 当前 Schema 仍是最小占位结构，需要结合真实数据结构继续补充 `required`、`properties` 与字段约束。

## 开发修改日志

- 2026-05-07：创建本目录 DEV.md，用于记录 Schema 目录职责和后续修改历史。
- 2026-05-20：补充 LF 输出与 weak signal 的图谱实体枚举，和新版疾病图谱实体集合保持一致。
