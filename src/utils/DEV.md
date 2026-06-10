# utils 目录开发维护文档

## 目录职责

提供跨模块复用的 IO、JSON Schema、日志、span、哈希和异常工具。

## 文件与功能说明

- `__init__.py`：包初始化文件。
- `io.py`：维护 JSON、JSONL、YAML 读取写入和父目录创建工具。
- `llm_client.py`：维护 OpenAI-compatible Teacher LLM 配置读取、客户端创建、文本模型调用、深度思考 extra_body 参数和图片/图表识别模型调用。
- `json_schema.py`：维护 JSON Schema 工具，当前仍是骨架。
- `logging.py`：维护日志工具，当前仍是骨架。
- `text_span.py`：维护全文拼接、字符到页码映射和全局 span 到页内 span 转换。
- `hashing.py`：维护稳定短哈希、文档 ID 和实体 ID 工具。
- `exceptions.py`：维护项目自定义异常，当前仍是骨架。

## 已实现功能

- 已实现 JSON、JSONL、YAML 基础 IO。
- 已实现 Teacher LLM 配置读取、可用性判断、OpenAI-compatible 客户端创建、文本模型调用、OCR/VL 流式调用和本地图片 data URL 转换。
- 已实现 `sha1_12`、`stable_doc_id`、`stable_entity_id`。
- 已实现全文字符映射和 page span 转换。
- JSON Schema、日志和异常工具后续需要继续补充。

## 开发修改日志

- 2026-05-07：创建本目录 DEV.md，用于记录 utils 目录职责、文件功能和后续修改历史。
- 2026-05-08：新增 `llm_client.py`，封装通用模型、翻译模型和图表识别模型的 OpenAI-compatible 调用能力。
- 2026-06-09：`llm_client.py` 的 `enable_thinking` 缺省值改为 `true`，并在文本模型请求中传递 `extra_body.enable_thinking`。
