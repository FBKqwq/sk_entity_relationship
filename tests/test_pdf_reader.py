"""PDF reader 相关基础测试。"""

from src.pdf_pipeline.page_cleaner import clean_page_text
from src.pdf_pipeline.english_translator_stub import translate_english_text
from src.pdf_pipeline.image_caption_stub import append_ocr_results_to_page_text, recognize_image_url
from src.pdf_pipeline.pdf_reader import _cluster_rows, _detect_two_column_kmeans, _rows_to_layout_lines


def test_clean_page_text_removes_noise_lines() -> None:
    text = "中华内科杂志 2024 年第 63 卷\n临床表现包括发热和咳嗽。\nDOI: 10.3760/example"

    cleaned = clean_page_text(text)

    assert "中华内科杂志" not in cleaned
    assert "DOI" not in cleaned
    assert "发热" in cleaned


def test_clean_page_text_removes_front_matter_block() -> None:
    text = (
        "中国急性胰腺炎诊治指南（\n"
        "中华医学会外科学分会胰腺外科学组\n"
        "Guidelinesfordiagnosisandtreatmentofacute\n"
        "MedicalAssociation\n"
        "Keywords acutepancreatitis；diagnosis；treatment\n"
        "【 】急性胰腺炎；诊断；治疗；随访；指南\n"
        "R6 A\n"
        "急性胰腺炎指因胰酶异常激活对胰腺自身及周围器官产生消化作用。\n"
    )

    cleaned = clean_page_text(text)

    assert cleaned.startswith("急性胰腺炎指")
    assert "Guidelines" not in cleaned
    assert "Keywords" not in cleaned


def test_detect_two_column_kmeans() -> None:
    words = []
    for index in range(100):
        words.append({"x0": 40 + index % 10, "x1": 50 + index % 10, "top": index, "bottom": index + 1, "text": "左"})
        words.append({"x0": 340 + index % 10, "x1": 350 + index % 10, "top": index, "bottom": index + 1, "text": "右"})

    two_column, split_x = _detect_two_column_kmeans(words, 420)

    assert two_column is True
    assert split_x is not None
    assert 100 < split_x < 320


def test_rows_to_layout_lines_preserves_bbox_and_offsets() -> None:
    words = [
        {"x0": 10, "x1": 20, "top": 5, "bottom": 15, "text": "Alpha"},
        {"x0": 25, "x1": 35, "top": 5, "bottom": 15, "text": "Beta"},
        {"x0": 10, "x1": 20, "top": 25, "bottom": 35, "text": "Gamma"},
    ]

    lines = _rows_to_layout_lines(_cluster_rows(words))

    assert lines[0]["text"] == "AlphaBeta"
    assert lines[0]["bbox"] == [10.0, 5.0, 35.0, 15.0]
    assert lines[0]["start_offset"] == 0
    assert lines[0]["end_offset"] == 9
    assert lines[1]["text"] == "Gamma"


def test_translate_english_text_returns_stub_when_disabled(tmp_path) -> None:
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "teacher_llm:\n"
        "  enabled: false\n"
        "  provider: openai_compatible\n"
        "  model_name: qwen-plus\n"
        "  TR_model_name: qwen-plus\n"
        "  OCR_model_name: qwen3-vl-plus\n",
        encoding="utf-8",
    )

    result = translate_english_text("This is a medical paragraph.", config_path=config_path)

    assert result["status"] == "stub"
    assert result["text"] == "This is a medical paragraph."


def test_recognize_image_url_returns_stub_when_disabled(tmp_path) -> None:
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "teacher_llm:\n"
        "  enabled: false\n"
        "  provider: openai_compatible\n"
        "  model_name: qwen-plus\n"
        "  TR_model_name: qwen-plus\n"
        "  OCR_model_name: qwen3-vl-plus\n",
        encoding="utf-8",
    )

    result = recognize_image_url("https://example.com/a.png", page_number=1, config_path=config_path)

    assert result["status"] == "stub"
    assert result["text"] == "（图片内容待识别）"


def test_append_ocr_results_to_page_text() -> None:
    page_text = {1: "正文内容"}
    ocr_results = {1: [{"caption": "图1", "text": "图像识别结果", "status": "recognized"}]}

    output = append_ocr_results_to_page_text(page_text, ocr_results)

    assert "[FIGURE_CONTENT]" in output[1]
    assert "图像识别结果" in output[1]
