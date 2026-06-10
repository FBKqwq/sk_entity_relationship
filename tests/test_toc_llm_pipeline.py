from src.pdf_pipeline.semantic_chunker import build_chunk_payload
from src.pdf_pipeline.toc_llm_pipeline import (
    TocCandidate,
    build_sections_with_dual_toc,
    collect_rule_toc_candidates,
)
from src.utils.text_span import build_fulltext_and_maps


def test_dual_toc_local_validation_rejects_measurement_noise() -> None:
    page_text = {
        1: (
            "Abstract\n"
            "Summary text.\n\n"
            "Methods\n"
            "Diagnosis body.\n\n"
            "51.7%~93% should stay in the body.\n\n"
            "Results\n"
            "Treatment body."
        )
    }
    full_text, char_to_page, _ = build_fulltext_and_maps(page_text)
    sections, report = build_sections_with_dual_toc(
        page_text,
        full_text,
        char_to_page,
        total_pages=1,
        llm_config_path="configs/llm.yaml",
        config={
            "first_layer_llm": {"enabled": False},
            "judge_llm": {"enabled": False},
        },
    )

    titles = [section.title for section in sections]

    assert "Methods" in titles
    assert "Results" in titles
    assert not any(title.startswith("51.7%") for title in titles)
    assert report["rule_candidate_count"] >= 2
    assert report["final_accepted_title_count"] >= 2
    assert report["final_metrics"]["title_match_rate"] == 1.0


def test_dual_toc_merges_rule_and_llm_candidate_without_real_llm() -> None:
    page_text = {1: "Abstract\nBody.\n\nMethods\nDiagnosis body.\n"}
    full_text, char_to_page, _ = build_fulltext_and_maps(page_text)
    rule_candidates = collect_rule_toc_candidates(full_text, char_to_page)
    llm_candidate = TocCandidate(
        title="Methods",
        level=1,
        start=full_text.find("Methods"),
        page=1,
        source="llm",
        evidence_text="Methods",
        confidence=0.70,
    )

    sections, report = build_sections_with_dual_toc(
        page_text,
        full_text,
        char_to_page,
        total_pages=1,
        llm_config_path="configs/llm.yaml",
        config={
            "first_layer_llm": {"enabled": False},
            "judge_llm": {"enabled": False},
        },
    )

    assert any(candidate.title == llm_candidate.title for candidate in rule_candidates)
    assert any(section.title == "Methods" for section in sections)
    assert report["llm_extract_report"]["status"] == "skipped"
    assert report["llm_judge_report"]["status"] == "skipped"


def test_build_chunk_payload_writes_toc_llm_report_when_enabled() -> None:
    pages = [
        {
            "page_number": 1,
            "text": "Abstract\nSummary text.\n\nMethods\nDiagnosis body.\n\nResults\nTreatment body.",
            "meta": {},
        }
    ]

    payload = build_chunk_payload(
        pages,
        pdf_path="sample.pdf",
        chunking_config={"min_chars": 1, "max_chars": 200, "filter_non_knowledge_section": False},
        pdf_config={
            "llm_config": "configs/llm.yaml",
            "toc_llm_pipeline": {
                "enabled": True,
                "first_layer_llm": {"enabled": False},
                "judge_llm": {"enabled": False},
            },
        },
    )

    assert payload["total_chunks"] >= 1
    assert payload["toc_llm_report"]["mode"] == "dual_toc_v1"
    assert payload["toc_llm_report"]["final_accepted_title_count"] >= 2
