"""Tests for skipping garbled PDF text before chunking."""

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


def test_garbled_pdf_text_skips_chunking(tmp_path, monkeypatch) -> None:
    module = _load_build_chunks_module()
    pdf_path = tmp_path / "garbled.pdf"
    output_path = tmp_path / "garbled.chunk.json"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    def fake_read_pdf_pages(*args, **kwargs):
        return [{"page_number": 1, "text": "¤§¶▓▒▣★◆" * 20, "meta": {}}]

    def fake_build_chunk_payload(*args, **kwargs):
        raise AssertionError("Chunking should not run for garbled parsed text")

    def fake_prescreen(*args, **kwargs):
        raise AssertionError("Semantic prescreen should not run for garbled parsed text")

    monkeypatch.setattr(module, "read_pdf_pages", fake_read_pdf_pages)
    monkeypatch.setattr(module, "build_chunk_payload", fake_build_chunk_payload)
    monkeypatch.setattr(module, "apply_semantic_section_llm_prescreen", fake_prescreen)

    reports = module.build_one_pdf(
        pdf_path,
        output_path,
        {
            "pdf": {
                "parser": "pdfplumber",
                "enable_double_column": True,
                "enable_english_translation_model": False,
                "enable_english_translation_stub": False,
                "semantic_section_llm": {"enabled": True},
                "skip_if_garbled_text": {"enabled": True},
                "figure_table_linking": {"enabled": False},
                "figure_table_ocr": {"enabled": False},
            },
            "chunking": {"min_chars": 1, "max_chars": 100},
        },
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    validation = json.loads(output_path.with_suffix(".validation.json").read_text(encoding="utf-8"))
    assert payload["skipped"] is True
    assert payload["total_chunks"] == 0
    assert payload["text_quality_report"]["skip"] is True
    assert validation["pass"] is False
    assert validation["issues"][0]["type"] == "SKIPPED_GARBLED_TEXT"
    assert reports["validation"]["skipped"] is True

