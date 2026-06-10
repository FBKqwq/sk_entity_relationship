"""JSON、JSONL、YAML 与路径 IO 工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _json_default(value: Any) -> Any:
    """Convert numpy/pandas values before JSON serialization."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None

    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def ensure_parent(path: str | Path) -> Path:
    """确保输出文件的父目录存在。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def read_json(path: str | Path) -> Any:
    """读取 UTF-8 JSON 文件。"""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data: Any, path: str | Path, *, indent: int = 2) -> Path:
    """写入 UTF-8 JSON 文件，保留中文。"""
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent, default=_json_default)
        file.write("\n")
    return target


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """写入 JSONL 文件，逐行保留中文。"""
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    return target


def read_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置。"""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("读取 YAML 配置需要安装 PyYAML。") from exc

    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML 配置必须是对象: {path}")
    return data
