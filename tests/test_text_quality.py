"""Tests for parsed PDF text quality checks."""

from src.pdf_pipeline.text_quality import analyze_parsed_pdf_text_quality


def test_text_quality_skips_symbol_garbled_text() -> None:
    pages = [{"page_number": 1, "text": "¤§¶▓▒▣★◆" * 20}]

    report = analyze_parsed_pdf_text_quality(pages)

    assert report["skip"] is True
    assert report["reasons"]


def test_text_quality_does_not_skip_readable_english_text() -> None:
    pages = [
        {
            "page_number": 1,
            "text": (
                "Perioperative infection prevention and management require "
                "risk assessment, antimicrobial timing, and postoperative monitoring."
            ),
        }
    ]

    report = analyze_parsed_pdf_text_quality(pages)

    assert report["skip"] is False
    assert report["ascii_letter_ratio"] >= 0.20


def test_text_quality_does_not_skip_readable_chinese_text() -> None:
    pages = [{"page_number": 1, "text": "感染性疾病诊疗专家共识建议结合临床表现、实验室指标和影像学结果进行综合判断。" * 3}]

    report = analyze_parsed_pdf_text_quality(pages)

    assert report["skip"] is False
    assert report["cjk_ratio"] > 0.02
