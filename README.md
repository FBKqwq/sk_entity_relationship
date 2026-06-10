# PDF 医学指南 Snorkel 弱监督项目

本项目用于把医学专家共识/诊疗指南 PDF 转换为可追溯的 chunk，并在此基础上构建两层 Snorkel 弱监督流程：

- PDF pipeline：读取 PDF、清洗页文本、识别章节、切分 semantic chunks，并关联图表证据。
- Lv1 弱监督：判断每个 chunk 是否包含目标实体类型，并预测候选实体数量。
- entity_extraction：用 Lv1 结果约束 Teacher LLM，抽取 `entity_base` 候选实体。
- Lv2 弱监督：后续将候选实体归类为最终图谱实体节点。

主要维护说明请优先查看各目录下的 `DEV.md`。
