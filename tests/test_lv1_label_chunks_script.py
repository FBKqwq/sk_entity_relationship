"""Tests for the Lv1 chunk-label main script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_script() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "02_snorkel_lv1_label_chunks.py"
    spec = importlib.util.spec_from_file_location("lv1_label_chunks_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_run_lv1_for_file_writes_lf_trace_and_fused_results(tmp_path: Path) -> None:
    script = _load_script()
    chunks_path = tmp_path / "sample.chunk.json"
    output_path = tmp_path / "sample.chunk_label_result.jsonl"
    lf_output_path = tmp_path / "sample.lv1_lf_outputs.jsonl"
    config_path = tmp_path / "weak_supervision.yaml"

    chunks_path.write_text(
        json.dumps(
            {
                "pdf_path": "sample.pdf",
                "chunks": [
                    {
                        "chunk_id": "CH1",
                        "document_id": "DOC1",
                        "section_title": "临床表现",
                        "section_path": ["正文", "临床表现"],
                        "text": "患者常见发热和疼痛，治疗原则为控制炎症。",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        """
active_labels:
  - symptoms
  - treatments
lv1_vote_model:
  threshold:
    symptoms: 0.30
    treatments: 0.35
lv1_count_model:
  max_count:
    symptoms: 30
    treatments: 10
dictionary:
  symptom_terms:
    - 发热
    - 疼痛
  treatment_terms:
    - 治疗原则
""",
        encoding="utf-8",
    )

    summary = script.run_lv1_for_file(
        chunks_path=chunks_path,
        output_path=output_path,
        lf_output_path=lf_output_path,
        weak_config_path=config_path,
        llm_config_path=None,
        disable_prompted_llm=True,
    )

    assert summary["chunks"] == 1
    assert summary["labels"] == ["symptoms", "treatments"]
    assert summary["prompted_llm_enabled"] is False
    assert output_path.exists()
    assert lf_output_path.exists()

    fused = _read_jsonl(output_path)
    assert len(fused) == 2
    assert {row["label"] for row in fused} == {"symptoms", "treatments"}
    assert all(row["chunk_id"] == "CH1" for row in fused)
    assert all(row["source_pdf"] == "sample.pdf" for row in fused)

    raw = _read_jsonl(lf_output_path)
    assert len(raw) == 10
    assert {row["lf_name"] for row in raw} == {
        "lv1_chunk_medical_pattern",
        "lv1_chunk_dictionary",
        "lv1_chunk_regex_indicator",
        "lv1_chunk_section_prior",
        "lv1_chunk_prompted_llm",
    }
    assert all(row["chunk_id"] == "CH1" for row in raw)
