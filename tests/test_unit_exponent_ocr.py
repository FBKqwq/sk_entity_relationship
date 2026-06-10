"""Unit exponent OCR recovery tests."""

from src.pdf_pipeline.unit_exponent_ocr import recover_unit_exponents_in_page


def test_recover_cfu_exponents_from_ocr(monkeypatch) -> None:
    text = (
        "\u83cc\u843d\u8ba1\u6570\u5973\u6027>10cfu\uff0fml"
        "\u3001\u7537\u6027>10cfu\uff0fml\uff0c\u6216\u6240\u6709"
    )
    lines = [{"text": text, "bbox": [280.0, 188.0, 520.0, 200.0]}]

    monkeypatch.setattr(
        "src.pdf_pipeline.unit_exponent_ocr._ocr_bbox",
        lambda *args, **kwargs: [
            "\u83cc\u843d\u8ba1\u6570\u5973\u6027>105 cfu/ml \u7537\u6027>104 cfu/ml\uff0c\u6216\u6240\u6709"
        ],
    )

    corrected, corrections = recover_unit_exponents_in_page(
        "dummy.pdf",
        0,
        lines,
        {"enabled": True},
    )

    assert corrected[0]["text"] == (
        "\u83cc\u843d\u8ba1\u6570\u5973\u6027>10^5cfu\uff0fml"
        "\u3001\u7537\u6027>10^4cfu\uff0fml\uff0c\u6216\u6240\u6709"
    )
    assert corrections[0]["status"] == "corrected"
    assert corrections[0]["source"] == "ocr_crop"
    assert corrections[0]["exponent_sources"] == ["ocr_crop", "ocr_crop"]


def test_leave_gender_cfu_line_unresolved_when_ocr_is_ambiguous(monkeypatch) -> None:
    text = (
        "\u83cc\u843d\u8ba1\u6570\u5973\u6027>10cfu\uff0fml"
        "\u3001\u7537\u6027>10cfu\uff0fml\uff0c\u6216\u6240\u6709"
    )
    lines = [{"text": text, "bbox": [280.0, 188.0, 520.0, 200.0]}]

    monkeypatch.setattr(
        "src.pdf_pipeline.unit_exponent_ocr._ocr_bbox",
        lambda *args, **kwargs: [
            "\u83cc\u843d\u8ba1\u6570\u5973\u6027>105 cfu/ml \u7537\u6027>10? cfu/ml\uff0c\u6216\u6240\u6709",
            "\u83cc\u843d\u8ba1\u6570\u5973\u6027>105 cfu/ml A HE>10\u00b0 cfu/ml\uff0c\u6216\u6240\u6709",
        ],
    )

    corrected, corrections = recover_unit_exponents_in_page(
        "dummy.pdf",
        0,
        lines,
        {"enabled": True},
    )

    assert corrected[0]["text"] == text
    assert corrections[0]["status"] == "unresolved"


def test_leave_catheter_cfu_line_unresolved_without_ocr_evidence(monkeypatch) -> None:
    text = "\u60a3\u8005\u5bfc\u5c3f\u7559\u53d6\u7684\u5c3f\u6807\u672c\u7ec6\u83cc\u83cc\u843d\u8ba1\u6570>10cfu\uff0fml"
    lines = [{"text": text, "bbox": [280.0, 205.0, 520.0, 216.0]}]

    monkeypatch.setattr("src.pdf_pipeline.unit_exponent_ocr._ocr_bbox", lambda *args, **kwargs: [""])

    corrected, corrections = recover_unit_exponents_in_page(
        "dummy.pdf",
        0,
        lines,
        {"enabled": True},
    )

    assert corrected[0]["text"] == text
    assert corrections[0]["status"] == "unresolved"
