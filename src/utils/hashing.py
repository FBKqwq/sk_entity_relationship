"""稳定 ID 与短哈希工具。"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha1_12(value: str) -> str:
    """返回 UTF-8 文本的 12 位 SHA1 摘要。"""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def stable_doc_id(source: str | Path) -> str:
    """基于文件名生成稳定文档 ID。"""
    name = Path(source).stem if not isinstance(source, str) or source.lower().endswith(".pdf") else source
    return f"DOC_{sha1_12(name)}"


def stable_entity_id(*parts: str) -> str:
    """基于实体关键字段生成稳定临时 ID。"""
    return f"TEMP_{sha1_12('|'.join(parts))}"
