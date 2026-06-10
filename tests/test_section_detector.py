"""章节识别测试。"""

from src.pdf_pipeline.section_detector import detect_sections


def test_detect_chinese_numbered_sections_with_space_form() -> None:
    text = "摘要\n本文说明。\n\n一、 临床表现\n表现为发热。\n\n四 、 治疗\n推荐抗感染治疗。"

    sections = detect_sections(text)
    titles = [section.title for section in sections]

    assert any("临床表现" in title for title in titles)
    assert any("治疗" in title for title in titles)


def test_detect_sections_fallback_to_body() -> None:
    sections = detect_sections("没有明确标题的正文内容。")

    assert len(sections) == 1
    assert sections[0].title == "正文"


def test_detect_sections_does_not_treat_percentage_as_heading() -> None:
    text = "一、临床表现\n2.生殖器溃疡：很少为首发表现，发生率为\n51.7%~93%。生殖器溃疡在男性多见。"

    sections = detect_sections(text)
    titles = [section.title for section in sections]

    assert any("临床表现" in title for title in titles)
    assert not any(title.startswith("51.7%") for title in titles)


def test_detect_sections_builds_hierarchical_path_for_h2() -> None:
    text = (
        "一、临床表现\n"
        "总述内容。\n\n"
        "1.口腔溃疡\n"
        "描述一。\n\n"
        "2.生殖器溃疡\n"
        "描述二。"
    )

    sections = detect_sections(text)
    h2_sections = [section for section in sections if section.title.startswith(("1.", "2."))]

    assert len(h2_sections) == 2
    assert all(section.path[:2] == ["正文", "一、临床表现"] for section in h2_sections)


def test_detect_sections_supports_mixed_numbering_styles() -> None:
    text = (
        "第一章 总论\n"
        "章内容。\n\n"
        "（一）临床特征\n"
        "小节内容。\n\n"
        "1）病史要点\n"
        "要点内容。\n\n"
        "1.1 诊断路径\n"
        "路径内容。"
    )
    sections = detect_sections(text)
    paths = [section.path for section in sections if "病史要点" in section.title or "诊断路径" in section.title]

    assert any(path == ["正文", "第一章 总论", "（一）临床特征", "1）病史要点"] for path in paths)
    assert any(path == ["正文", "第一章 总论", "（一）临床特征", "1）病史要点", "1.1 诊断路径"] for path in paths)


def test_detect_sections_supports_common_english_headings() -> None:
    text = "Abstract\nSummary text.\n\nMethods\nMethod text.\n\nResults\nResult text."
    sections = detect_sections(text)
    titles = [section.title for section in sections]

    assert "Abstract" in titles
    assert "Methods" in titles
    assert "Results" in titles


def test_detect_sections_supports_numbered_heading_without_space() -> None:
    text = (
        "一、总则\n"
        "总则内容。\n\n"
        "2.1.2推荐药物管理\n"
        "药物内容。"
    )
    sections = detect_sections(text)
    target = [section for section in sections if section.title.startswith("2.1.2")]

    assert len(target) == 1
    assert target[0].path == ["正文", "一、总则", "2.1.2推荐药物管理"]


def test_detect_sections_supports_ocr_joined_numeric_headings() -> None:
    text = (
        "Abstract\n"
        "摘要内容。\n\n"
        "12023年《指南》和2018年《共识》治疗部分比较\n"
        "比较内容。"
    )
    sections = detect_sections(text)
    target = [section for section in sections if "比较" in section.title]

    assert len(target) == 1
    assert target[0].path == ["正文", "Abstract", "12023年《指南》和2018年《共识》治疗部分比较"]
