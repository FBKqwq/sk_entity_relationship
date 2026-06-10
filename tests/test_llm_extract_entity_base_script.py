"""Tests for the Lv1-constrained entity extraction main script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_script() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "03_llm_extract_entity_base.py"
    spec = importlib.util.spec_from_file_location("llm_extract_entity_base_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_extract_entity_base_for_file_calls_llm_per_positive_chunk(tmp_path: Path) -> None:
    script = _load_script()
    chunks_path = tmp_path / "sample.chunk.json"
    lv1_path = tmp_path / "sample.chunk_label_result.jsonl"
    output_path = tmp_path / "sample.entity_base.jsonl"
    raw_path = tmp_path / "sample.teacher_llm_raw.jsonl"
    chunks_path.write_text(
        json.dumps(
            {
                "pdf_path": "sample.pdf",
                "chunks": [
                    {"chunk_id": "CH1", "document_id": "DOC1", "text": "检查提示C反应蛋白升高。"},
                    {"chunk_id": "CH2", "document_id": "DOC1", "text": "无实体。"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    lv1_rows = [
        {"chunk_id": "CH1", "label": "tests", "present": True, "status": "accepted"},
        {"chunk_id": "CH2", "label": "tests", "present": False, "status": "rejected"},
    ]
    lv1_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in lv1_rows),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_extractor(chunk: dict[str, Any], lv1_results: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        calls.append(str(chunk["chunk_id"]))
        return {
            "status": "ok",
            "prompt": "large prompt omitted by default",
            "entities": [
                {
                    "entity_type": "tests",
                    "name": "C反应蛋白",
                    "properties": {"test_name": "C反应蛋白"},
                    "evidence": "C反应蛋白升高",
                    "confidence": 0.9,
                }
            ],
        }

    summary = script.extract_entity_base_for_file(
        chunks_path=chunks_path,
        lv1_path=lv1_path,
        output_path=output_path,
        raw_output_path=raw_path,
        full_extraction=False,
        extractor=fake_extractor,
    )

    assert calls == ["CH1"]
    assert summary["called_chunks"] == 1
    assert summary["skipped_chunks"] == 1
    assert summary["entity_records"] == 1
    records = _read_jsonl(output_path)
    assert records[0]["chunk_id"] == "CH1"
    assert records[0]["entity_type"] == "tests"
    assert records[0]["name"] == "C反应蛋白"
    raw_rows = _read_jsonl(raw_path)
    assert "prompt" not in raw_rows[0]["response"]


def test_extract_entity_base_for_file_full_extraction_calls_all_chunks_without_lv1(tmp_path: Path) -> None:
    script = _load_script()
    chunks_path = tmp_path / "sample.chunk.json"
    output_path = tmp_path / "sample.entity_base.jsonl"
    raw_path = tmp_path / "sample.teacher_llm_raw.jsonl"
    chunks_path.write_text(
        json.dumps(
            {
                "pdf_path": "sample.pdf",
                "chunks": [
                    {"chunk_id": "CH1", "document_id": "DOC1", "text": "检查提示C反应蛋白升高。"},
                    {"chunk_id": "CH2", "document_id": "DOC1", "text": "给予抗菌药物治疗。"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, bool, list[dict[str, Any]]]] = []

    def fake_extractor(
        chunk: dict[str, Any],
        lv1_results: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append((str(chunk["chunk_id"]), bool(kwargs.get("full_extraction")), lv1_results))
        return {
            "status": "ok",
            "entities": [
                {
                    "entity_type": "tests",
                    "name": f"entity_{chunk['chunk_id']}",
                    "properties": {"test_name": f"entity_{chunk['chunk_id']}"},
                    "evidence": str(chunk["text"]),
                    "confidence": 0.8,
                }
            ],
        }

    summary = script.extract_entity_base_for_file(
        chunks_path=chunks_path,
        lv1_path=None,
        output_path=output_path,
        raw_output_path=raw_path,
        full_extraction=True,
        extractor=fake_extractor,
    )

    assert calls == [("CH1", True, []), ("CH2", True, [])]
    assert summary["called_chunks"] == 2
    assert summary["skipped_chunks"] == 0
    assert summary["Full_extraction"] is True
    assert summary["lv1_path"] is None
    assert len(_read_jsonl(output_path)) == 2
    raw_rows = _read_jsonl(raw_path)
    assert [row["Full_extraction"] for row in raw_rows] == [True, True]


def test_full_extraction_default_is_read_from_llm_yaml(tmp_path: Path) -> None:
    script = _load_script()
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "teacher_llm:\n  Full_extraction: false\n",
        encoding="utf-8",
    )

    assert script._full_extraction_from_config(config_path) is False
    assert script._full_extraction_from_config(tmp_path / "missing.yaml") is False
