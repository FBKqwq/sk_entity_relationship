"""Tests for the pdf.isTable switch in the chunk builder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_build_chunks_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "01_build_chunks.py"
    spec = importlib.util.spec_from_file_location("build_chunks_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_is_table_false_forces_text_only_pipeline(tmp_path, monkeypatch) -> None:
    module = _load_build_chunks_module()
    pdf_path = tmp_path / "sample.pdf"
    output_path = tmp_path / "sample.chunk.json"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    calls: dict[str, object] = {}

    def fake_opendataloader(*args, **kwargs):
        raise AssertionError("OpenDataLoader should not run when pdf.isTable=false")

    def fake_read_pdf_pages(*args, **kwargs):
        calls["return_layout"] = kwargs.get("return_layout")
        return [{"page_number": 1, "text": "正文文本。", "meta": {}}]

    def fake_build_chunk_payload(*args, **kwargs):
        return {
            "doc_id": "DOC",
            "pdf_path": str(pdf_path),
            "total_chunks": 1,
            "chunks": [
                {
                    "chunk_id": "CH0001",
                    "text": "正文文本。",
                    "page_span": [1, 1],
                    "text_span": [0, 5],
                }
            ],
        }

    def fake_linked_ocr(*args, **kwargs):
        raise AssertionError("Linked object OCR should not run when pdf.isTable=false")

    monkeypatch.setattr(module, "read_pdf_pages_with_opendataloader", fake_opendataloader)
    monkeypatch.setattr(module, "read_pdf_pages", fake_read_pdf_pages)
    monkeypatch.setattr(module, "build_chunk_payload", fake_build_chunk_payload)
    monkeypatch.setattr(module, "run_linked_object_ocr_and_inject", fake_linked_ocr)
    monkeypatch.setattr(module, "validate_chunk_payload", lambda *args, **kwargs: {"pass": True})
    monkeypatch.setattr(
        module,
        "build_heading_hit_report",
        lambda payload: {"metrics": {"heading_hit_rate": 1.0, "hierarchical_hit_rate": 1.0, "weighted_hit_score": 1.0}},
    )

    module.build_one_pdf(
        pdf_path,
        output_path,
        {
            "pdf": {
                "parser": "opendataloader",
                "isTable": False,
                "enable_double_column": True,
                "enable_english_translation_model": False,
                "enable_english_translation_stub": False,
                "enable_ocr_model": True,
                "enable_image_caption_stub": True,
                "semantic_section_llm": {"enabled": False},
                "figure_table_linking": {"enabled": True, "return_layout": True},
                "figure_table_ocr": {"enabled": True},
            },
            "chunking": {"min_chars": 1, "max_chars": 100},
        },
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert calls["return_layout"] is False
    assert payload["figure_table_objects"] == []
    assert not output_path.with_name(f"{output_path.stem}.figure_ocr.json").exists()

