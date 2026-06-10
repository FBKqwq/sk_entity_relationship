from src.pdf_pipeline.semantic_section_llm import (
    _build_ops_from_parsed,
    _format_forbidden_spans,
    _looks_like_contract_violation,
    _page_break_spans,
    _validate_output_keys,
)


def test_page_break_spans_are_exposed_as_forbidden_ranges() -> None:
    marked = "第一页正文\n<<<PAGE_BREAK:2>>>\n第二页正文"

    spans = _page_break_spans(marked, [1, 2])

    assert spans == [(5, 25)]
    assert _format_forbidden_spans(spans) == '[{"start": 5, "end": 25}]'


def test_contract_violation_flags_overlong_model_output() -> None:
    err = _looks_like_contract_violation("x" * 22785, marked_len=8736, max_output_chars=12000)

    assert "模型输出过长" in err
    assert "22785" in err
    assert "12000" in err


def test_contract_violation_flags_cleaned_text_echo() -> None:
    err = _looks_like_contract_violation('{"cleaned_text":"正文"}', marked_len=20, max_output_chars=12000)

    assert "cleaned_text" in err


def test_validate_output_keys_rejects_extra_payload_fields() -> None:
    err = _validate_output_keys({"noise_spans": [], "translation_spans": [], "cleaned_text": "正文"})

    assert "cleaned_text" in err


def test_validate_output_keys_accepts_span_only_payload() -> None:
    err = _validate_output_keys({"noise_spans": [], "translation_spans": []})

    assert err == ""


def test_noise_span_rejects_mid_token_delete_boundaries() -> None:
    marked = "有效正文参考文献\n[1] Example"
    parsed = {"noise_spans": [{"start": 5, "end": 9}], "translation_spans": []}

    ops, err = _build_ops_from_parsed(parsed, marked, [])

    assert ops == []
    assert "截断正文字符" in err
