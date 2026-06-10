from src.pdf_pipeline.figure_table_linker import attach_figure_table_links, find_explicit_figure_table_refs
from src.pdf_pipeline.layout_regions import build_chunk_page_regions
from src.pdf_pipeline.table_object_builder import (
    build_figure_objects_from_images,
    build_table_objects,
    extract_caption_candidates,
    extract_note_candidates,
)


def test_find_explicit_figure_table_refs() -> None:
    refs = find_explicit_figure_table_refs("See Table 1 and Figure 2 for details.")

    assert refs == [
        {"object_type": "table", "number": "1", "raw": "Table 1"},
        {"object_type": "figure", "number": "2", "raw": "Figure 2"},
    ]


def test_build_chunk_page_regions_from_layout_lines() -> None:
    chunk = {"page_start": 1, "page_end": 1, "text": "Diagnosis criteria are shown in Table 1."}
    page_layouts = {
        1: [
            {"text": "Intro line", "bbox": [50, 50, 300, 65]},
            {"text": "Diagnosis criteria are shown in Table 1.", "bbox": [50, 80, 300, 100]},
        ]
    }

    regions = build_chunk_page_regions(chunk, page_layouts)

    assert regions == [{"page": 1, "bbox": [50, 80, 300, 100]}]


def test_build_table_objects_merges_caption_and_note() -> None:
    page_layouts = {
        1: [
            {"text": "Table 1 Diagnostic score", "bbox": [50, 70, 350, 85]},
            {"text": "Note: optional item", "bbox": [50, 180, 350, 195]},
        ]
    }
    table_bodies = [
        {
            "object_id": "T_p001_001",
            "object_type": "table",
            "page": 1,
            "body": [["symptom", "score"], ["eye", "2"]],
            "body_bbox": [50, 100, 350, 170],
            "bbox": [50, 100, 350, 170],
        }
    ]

    tables = build_table_objects(table_bodies, page_layouts, {"caption_search_distance": 40, "note_search_distance": 40})

    assert tables[0]["caption"] == "Table 1 Diagnostic score"
    assert tables[0]["note"] == "Note: optional item"
    assert tables[0]["table_number"] == "1"
    assert tables[0]["bbox"] == [50, 70, 350, 195]


def test_caption_and_note_candidates() -> None:
    layout = [
        {"text": "Table 3 Example", "bbox": [0, 0, 1, 1]},
        {"text": "Note: details", "bbox": [0, 2, 1, 3]},
    ]

    assert len(extract_caption_candidates(layout)) == 1
    assert len(extract_note_candidates(layout)) == 1


def test_attach_figure_table_links_by_explicit_reference() -> None:
    payload = {
        "chunks": [
            {
                "chunk_id": "CH0001",
                "page_start": 1,
                "page_end": 1,
                "text": "Diagnosis criteria are shown in Table 1.",
            }
        ]
    }
    page_layouts = {1: [{"text": "Diagnosis criteria are shown in Table 1.", "bbox": [50, 80, 350, 100]}]}
    objects = [
        {
            "object_id": "T_p001_001",
            "object_type": "table",
            "page": 1,
            "caption": "Table 1 Diagnostic score",
            "table_number": "1",
            "bbox": [50, 120, 350, 200],
            "ocr_status": "not_required",
        }
    ]

    linked = attach_figure_table_links(
        payload,
        page_layouts,
        objects,
        {"max_linked_objects_per_chunk": 3, "link_by_layout_proximity": False},
    )

    chunk = linked["chunks"][0]
    assert chunk["page_regions"] == [{"page": 1, "bbox": [50, 80, 350, 100]}]
    assert chunk["linked_tables"][0]["object_id"] == "T_p001_001"
    assert chunk["linked_tables"][0]["link_reasons"] == ["explicit_reference"]
    assert chunk["linked_tables"][0]["link_confidence"] == 0.95


def test_build_figure_objects_filters_small_images() -> None:
    pages = [
        {
            "page_number": 1,
            "meta": {
                "images": [
                    {"bbox": [0, 0, 20, 20], "source": "pdfplumber_page_images"},
                    {"bbox": [0, 40, 120, 180], "source": "pdfplumber_page_images"},
                ]
            },
        }
    ]

    figures = build_figure_objects_from_images(pages, {}, {"min_figure_width": 80, "min_figure_height": 80})

    assert len(figures) == 1
    assert figures[0]["bbox"] == [0, 40, 120, 180]
