"""语义 chunk 切分测试。"""

from src.pdf_pipeline.semantic_chunker import build_chunk_payload


def test_build_chunk_payload_keeps_required_fields() -> None:
    pages = [
        {
            "page_number": 1,
            "text": "一、 临床表现\n患者可表现为发热、咳嗽。\n\n二、 治疗\n推荐抗感染治疗。",
            "meta": {},
        }
    ]

    payload = build_chunk_payload(
        pages,
        pdf_path="data/raw_pdfs/sample.pdf",
        chunking_config={"method": "section_semantic_v1", "max_chars": 1200, "min_chars": 10},
    )

    assert payload["doc_id"].startswith("DOC_")
    assert payload["total_chunks"] >= 2
    first = payload["chunks"][0]
    assert first["chunk_id"] == "CH0001"
    assert first["page_start"] == 1
    assert first["text_span"]["start"] < first["text_span"]["end"]
    assert first["text"]


def test_build_chunk_payload_splits_long_paragraphs() -> None:
    long_text = "一、 临床表现\n" + "患者表现为发热。" * 120
    pages = [{"page_number": 1, "text": long_text, "meta": {}}]

    payload = build_chunk_payload(
        pages,
        pdf_path="data/raw_pdfs/sample.pdf",
        chunking_config={"method": "section_semantic_v1", "max_chars": 120, "min_chars": 20},
    )

    assert payload["total_chunks"] > 1
    assert all(len(chunk["text"]) <= 120 for chunk in payload["chunks"])


def test_build_chunk_payload_drops_low_cjk_ratio_chunks() -> None:
    pages = [
        {
            "page_number": 1,
            "text": (
                "Abstract\n"
                "This section is written entirely in English and should be dropped because it has almost no CJK text. "
                "The content is long enough to pass the minimum length threshold.\n\n"
                "一、临床表现\n"
                "患者可表现为发热、皮疹和反复感染，需要结合病原学检查和临床表现进行诊断。"
            ),
            "meta": {},
        }
    ]

    payload = build_chunk_payload(
        pages,
        pdf_path="data/raw_pdfs/sample.pdf",
        chunking_config={
            "method": "section_semantic_v1",
            "max_chars": 1200,
            "min_chars": 10,
            "filter_non_knowledge_section": False,
            "chunk_language_filter": {"enabled": True, "min_chars": 80, "min_cjk_ratio": 0.20},
        },
    )

    merged_text = "\n".join(chunk["text"] for chunk in payload["chunks"])
    assert "This section is written entirely in English" not in merged_text
    assert "患者可表现为发热" in merged_text
    assert payload["dropped_chunks"]
    assert payload["dropped_chunks"][0]["reason"] == "low_cjk_ratio"


def test_build_chunk_payload_filters_non_knowledge_tail_sections() -> None:
    text = (
        "一、 临床表现\n"
        "患者可表现为反复口腔溃疡和发热。\n\n"
        "执笔：某某（某医院）\n"
        "诊疗规范撰写组名单（按姓氏排序）：甲、乙、丙。\n"
        "参考文献\n"
        "[1] Example reference."
    )
    pages = [{"page_number": 1, "text": text, "meta": {}}]

    payload = build_chunk_payload(
        pages,
        pdf_path="data/raw_pdfs/sample.pdf",
        chunking_config={"method": "section_semantic_v1", "max_chars": 1200, "min_chars": 10},
    )

    merged_text = "\n".join(chunk["text"] for chunk in payload["chunks"])
    assert "反复口腔溃疡" in merged_text
    assert "执笔" not in merged_text
    assert "诊疗规范撰写组名单" not in merged_text
    assert "参考文献" not in merged_text


def test_build_chunk_payload_filters_author_members_before_references() -> None:
    text = (
        "二、治疗\n"
        "抗菌药物的选择与一般的复杂性尿路感染相同。\n"
        "编写组成员(按单位汉语拼音排序)：北京大学第一医院抗感染科(郑波)。\n\n"
        "参考文献\n"
        "[1] Example reference."
    )
    pages = [{"page_number": 1, "text": text, "meta": {}}]

    payload = build_chunk_payload(
        pages,
        pdf_path="data/raw_pdfs/sample.pdf",
        chunking_config={"method": "section_semantic_v1", "max_chars": 1200, "min_chars": 10},
    )

    merged_text = "\n".join(chunk["text"] for chunk in payload["chunks"])
    assert "抗菌药物的选择" in merged_text
    assert "编写组成员" not in merged_text
    assert "参考文献" not in merged_text
    assert "Example reference" not in merged_text


def test_build_chunk_payload_filters_line_broken_pinyin_author_marker() -> None:
    text = (
        "7 急性胰腺炎的随访\n"
        "临床医生需根据具体情况采用个体化的诊疗措施，以获得最佳疗效。\n"
        "（按姓氏汉语拼音排\n"
        "序）：\n"
        "蔡守旺 陈其龙 陈汝福。\n"
        "所有作者均声明不存在利益冲突\n"
        "［1］ Example reference."
    )
    pages = [{"page_number": 1, "text": text, "meta": {}}]

    payload = build_chunk_payload(
        pages,
        pdf_path="data/raw_pdfs/sample.pdf",
        chunking_config={"method": "section_semantic_v1", "max_chars": 1200, "min_chars": 10},
    )

    merged_text = "\n".join(chunk["text"] for chunk in payload["chunks"])
    assert "最佳疗效" in merged_text
    assert "按姓氏汉语拼音" not in merged_text
    assert "蔡守旺" not in merged_text
    assert "利益冲突" not in merged_text
    assert "Example reference" not in merged_text


def test_build_chunk_payload_merges_short_parent_heading_into_child_section() -> None:
    text = (
        "四、特殊类型的复杂性尿路感染\n"
        "(一)合并尿路结石的复杂性尿路感染\n"
        "结石并发尿路感染时，需要依据症状、体征及相关实验室和影像学检查进行诊断。"
        "尿培养也是常规项目，应结合患者情况进行综合评估。"
    )
    pages = [{"page_number": 1, "text": text, "meta": {}}]

    payload = build_chunk_payload(
        pages,
        pdf_path="data/raw_pdfs/sample.pdf",
        chunking_config={"method": "section_semantic_v1", "max_chars": 1200, "min_chars": 500},
    )

    assert payload["total_chunks"] == 1
    chunk = payload["chunks"][0]
    assert chunk["section_title"] == "四、特殊类型的复杂性尿路感染"
    assert chunk["section_path"] == ["正文", "四、特殊类型的复杂性尿路感染"]
    assert chunk["text"].startswith("四、特殊类型的复杂性尿路感染")
    assert "(一)合并尿路结石的复杂性尿路感染" in chunk["text"]


def test_build_chunk_payload_does_not_merge_short_sibling_sections() -> None:
    text = (
        "一、临床表现\n"
        + "患者发热。"
        + "\n\n"
        + "二、辅助检查\n"
        + "血常规异常。"
    )
    pages = [{"page_number": 1, "text": text, "meta": {}}]

    payload = build_chunk_payload(
        pages,
        pdf_path="data/raw_pdfs/sample.pdf",
        chunking_config={
            "method": "section_semantic_v1",
            "max_chars": 2000,
            "min_chars": 50,
            "min_section_merge_chars": 500,
        },
    )

    assert payload["total_chunks"] == 2
    assert payload["chunks"][0]["section_path"] == ["正文", "一、临床表现"]
    assert payload["chunks"][1]["section_path"] == ["正文", "二、辅助检查"]


def test_build_chunk_payload_merges_short_child_siblings_to_parent_section() -> None:
    text = (
        "四、病理改变\n"
        "急性期主要病理改变为单核-吞噬细胞系统弥漫性增生。\n\n"
        "（一）肝、脾。\n"
        "有不同程度的细胞浸润。\n\n"
        "（二）淋巴结。\n"
        "感染早期几乎都会受累。\n\n"
        "（三）骨、关节。\n"
        "主要表现为关节周围软组织肿胀。"
    )
    pages = [{"page_number": 1, "text": text, "meta": {}}]

    payload = build_chunk_payload(
        pages,
        pdf_path="data/raw_pdfs/sample.pdf",
        chunking_config={
            "method": "section_semantic_v1",
            "max_chars": 2000,
            "min_chars": 50,
            "min_section_merge_chars": 500,
        },
    )

    assert payload["total_chunks"] == 1
    chunk = payload["chunks"][0]
    assert chunk["section_path"] == ["正文", "四、病理改变"]
    assert "（一）肝、脾。" in chunk["text"]
    assert "（二）淋巴结。" in chunk["text"]
    assert "（三）骨、关节。" in chunk["text"]
