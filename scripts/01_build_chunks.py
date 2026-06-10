"""从 PDF 构建 chunk.json 的命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_pipeline.chunk_validator import validate_chunk_payload
from src.pdf_pipeline.english_translator_stub import translate_english_paragraphs
from src.pdf_pipeline.figure_table_linker import attach_figure_table_links
from src.pdf_pipeline.heading_report import build_heading_hit_report
from src.pdf_pipeline.image_caption_stub import append_ocr_results_to_page_text, recognize_pdf_images_by_page
from src.pdf_pipeline.layout_regions import build_page_layouts
from src.pdf_pipeline.linked_object_ocr import run_linked_object_ocr_and_inject
from src.pdf_pipeline.opendataloader_reader import read_pdf_pages_with_opendataloader
from src.pdf_pipeline.pdf_reader import read_pdf_pages
from src.pdf_pipeline.semantic_chunker import build_chunk_payload
from src.pdf_pipeline.semantic_section_llm import apply_semantic_section_llm_prescreen
from src.pdf_pipeline.table_extractor import extract_table_bodies
from src.pdf_pipeline.table_object_builder import build_figure_objects_from_images, build_table_objects
from src.pdf_pipeline.text_quality import analyze_parsed_pdf_text_quality
from src.utils.io import write_json, read_yaml


def _progress(msg: str) -> None:
    """终端进度输出（立即刷新，便于长任务观察）。"""
    print(f"[进度] {msg}", flush=True)


def _build_heading_summary_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """构建批量章节命中率汇总报告（中文展示）。"""
    total_docs = len(rows)
    if total_docs == 0:
        return {
            "阶段": "批量章节命中率汇总",
            "版本": "v1",
            "汇总说明": "分母均为 total_chunks；命中率统计粒度为 chunk。",
            "总文档数": 0,
            "通过校验文档数": 0,
            "校验通过率": 0.0,
            "平均章节命中率": 0.0,
            "平均层级命中率": 0.0,
            "平均加权得分": 0.0,
            "文档明细": [],
        }

    pass_docs = sum(1 for row in rows if row["validation_pass"])
    avg_heading = sum(float(row["heading_hit_rate"]) for row in rows) / total_docs
    avg_hier = sum(float(row["hierarchical_hit_rate"]) for row in rows) / total_docs
    avg_weighted = sum(float(row["weighted_hit_score"]) for row in rows) / total_docs
    details = sorted(rows, key=lambda item: float(item["weighted_hit_score"]), reverse=True)

    return {
        "阶段": "批量章节命中率汇总",
        "版本": "v1",
        "汇总说明": {
            "统计口径": "分母均为 total_chunks；命中率统计粒度为 chunk。",
            "章节命中率公式": "heading_hit_rate = non_generic_heading_chunks / total_chunks",
            "层级命中率公式": "hierarchical_hit_rate = hierarchical_chunks / total_chunks",
            "加权得分公式": "weighted_hit_score = 0.7 * heading_hit_rate + 0.3 * hierarchical_hit_rate",
        },
        "总文档数": total_docs,
        "通过校验文档数": pass_docs,
        "校验通过率": round(pass_docs / total_docs, 4),
        "平均章节命中率": round(avg_heading, 4),
        "平均层级命中率": round(avg_hier, 4),
        "平均加权得分": round(avg_weighted, 4),
        "文档明细": details,
    }


def _load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return read_yaml(path)


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _is_table_parsing_enabled(pdf_config: dict[str, Any]) -> bool:
    """Return whether figure/table parsing is enabled for pdf->chunk."""
    return _coerce_bool(pdf_config.get("isTable"), default=True)


def _return_layout_enabled(pdf_config: dict[str, Any]) -> bool:
    if not _is_table_parsing_enabled(pdf_config):
        return False
    return bool(pdf_config.get("figure_table_linking", {}).get("return_layout", False))


def enrich_pages_with_models(
    pages: list[dict[str, Any]],
    *,
    input_pdf: Path,
    pdf_config: dict[str, Any],
    skip_translation: bool = False,
    skip_ocr: bool = False,
) -> list[dict[str, Any]]:
    """按配置在主流程中调用翻译和 OCR 模型增强页文本。"""
    page_text = {int(page["page_number"]): str(page.get("text", "")) for page in pages}
    llm_config_path = pdf_config.get("llm_config", "configs/llm.yaml")
    figure_table_ocr_config = pdf_config.get("figure_table_ocr", {})
    table_parsing_enabled = _is_table_parsing_enabled(pdf_config)
    skip_full_page_ocr = bool(
        skip_ocr
        or not table_parsing_enabled
        or pdf_config.get("skip_full_page_ocr_before_chunk", False)
        or (
            bool(figure_table_ocr_config.get("enabled", False))
            and bool(figure_table_ocr_config.get("only_linked_objects", False))
        )
        or (
            figure_table_ocr_config.get("only_linked_objects", False)
            and not figure_table_ocr_config.get("inject_into_chunk_text", False)
        )
    )

    will_translate = bool(
        pdf_config.get("enable_english_translation_model", pdf_config.get("enable_english_translation_stub", False))
    ) and not skip_translation
    will_full_page_ocr = bool(
        pdf_config.get("enable_ocr_model", pdf_config.get("enable_image_caption_stub", False))
    ) and not skip_full_page_ocr
    if not will_translate and not will_full_page_ocr:
        _progress("页模型增强：跳过翻译与全页 OCR（未启用或已由上游步骤跳过）。")

    if will_translate:
        _progress("页模型增强：正在调用英文段落翻译…")
        translated_text, translated_pages, pending_pages = translate_english_paragraphs(
            page_text,
            config_path=llm_config_path,
        )
        page_text = translated_text
        for page in pages:
            meta = page.setdefault("meta", {})
            page_number = int(page["page_number"])
            meta["translation_model_enabled"] = True
            meta["translated"] = page_number in translated_pages
            meta["translation_pending"] = page_number in pending_pages
        _progress(f"页模型增强：英文翻译完成（已翻译页: {len(translated_pages)}，待处理页: {len(pending_pages)}）。")

    if (
        will_full_page_ocr
    ):
        _progress("页模型增强：正在对每页图片做全页 OCR（可能较慢）…")
        ocr_results = recognize_pdf_images_by_page(
            input_pdf,
            config_path=llm_config_path,
            max_images_per_page=int(pdf_config.get("max_images_per_page", 1)),
        )
        page_text = append_ocr_results_to_page_text(page_text, ocr_results)
        for page in pages:
            meta = page.setdefault("meta", {})
            page_number = int(page["page_number"])
            meta["ocr_model_enabled"] = True
            meta["ocr_images"] = len(ocr_results.get(page_number, []))
        total_imgs = sum(len(v) for v in ocr_results.values())
        _progress(f"页模型增强：全页 OCR 完成（共处理图片数: {total_imgs}）。")

    for page in pages:
        page["text"] = page_text.get(int(page["page_number"]), str(page.get("text", "")))
    _progress("页模型增强：页文本已写回 pages。")
    return pages


def enrich_payload_with_figure_table_links(
    payload: dict[str, Any],
    *,
    pages: list[dict[str, Any]],
    input_pdf: Path,
    pdf_config: dict[str, Any],
    preparse_objects: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not _is_table_parsing_enabled(pdf_config):
        payload["figure_table_objects"] = []
        _progress("figure/table parsing disabled by pdf.isTable=false; text-only chunking.")
        return payload

    link_config = pdf_config.get("figure_table_linking", {})
    if not bool(link_config.get("enabled", False)):
        _progress("图表关联：已关闭（figure_table_linking.enabled=false），跳过。")
        return payload

    _progress("图表关联：正在构建版面、提取表体与图对象并写入 chunk 关联字段…")
    page_layouts = build_page_layouts(pages)
    parser = str(pdf_config.get("parser", "pdfplumber")).lower()
    discovered_objects = list(preparse_objects or [])
    if parser != "opendataloader":
        table_bodies = extract_table_bodies(input_pdf)
        for page in pages:
            table_bodies.extend(((page.get("meta", {}) or {}).get("text_table_bodies", []) or []))
        discovered_objects.extend(build_table_objects(table_bodies, page_layouts, link_config))
        discovered_objects.extend(build_figure_objects_from_images(pages, page_layouts, link_config))
    merged = attach_figure_table_links(payload, page_layouts, discovered_objects, link_config)
    n_obj = len(merged.get("figure_table_objects") or [])
    _progress(f"图表关联：完成（发现图/表对象数: {n_obj}）。")
    return merged


def _write_skipped_pdf_outputs(
    *,
    input_pdf: Path,
    output_path: Path,
    text_quality_report: dict[str, Any],
    preparse_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "doc_id": input_pdf.stem,
        "pdf_path": str(input_pdf),
        "total_chunks": 0,
        "chunks": [],
        "skipped": True,
        "skip_reason": "garbled_or_unusable_parsed_text",
        "text_quality_report": text_quality_report,
    }
    if preparse_report is not None:
        payload["preparse_report"] = preparse_report
    write_json(payload, output_path)

    report = {
        "stage": "chunk_validation",
        "total_chunks": 0,
        "length_stats": {"min": 0, "max": 0, "p50": 0, "p90": 0, "avg": 0},
        "warnings": [],
        "issues": [
            {
                "type": "SKIPPED_GARBLED_TEXT",
                "reason": "garbled_or_unusable_parsed_text",
                "text_quality_status": text_quality_report.get("status"),
                "text_quality_reasons": text_quality_report.get("reasons", []),
            }
        ],
        "pass": False,
        "skipped": True,
        "text_quality_report": text_quality_report,
    }
    report_path = output_path.with_suffix(".validation.json")
    write_json(report, report_path)

    heading_report = build_heading_hit_report(payload)
    heading_report["skipped"] = True
    heading_report["skip_reason"] = "garbled_or_unusable_parsed_text"
    heading_report_path = output_path.with_suffix(".heading_report.json")
    write_json(heading_report, heading_report_path)
    return {"validation": report, "heading_report": heading_report}


def build_one_pdf(input_pdf: Path, output_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """构建单个 PDF 的 chunk JSON，并返回校验报告。"""
    pdf_config = config.get("pdf", {})
    chunking_config = config.get("chunking", {})
    llm_config_path = pdf_config.get("llm_config", "configs/llm.yaml")
    prescreen_cfg = pdf_config.get("semantic_section_llm") or {}
    prescreen_on = bool(prescreen_cfg.get("enabled", False))
    table_parsing_enabled = _is_table_parsing_enabled(pdf_config)
    parser = str(pdf_config.get("parser", "pdfplumber")).lower() if table_parsing_enabled else "pdfplumber"
    preparse_objects: list[dict[str, Any]] = []
    preparse_report: dict[str, Any] | None = None

    _progress(f"开始处理: {input_pdf.name} -> {output_path}")
    if parser == "opendataloader":
        _progress("PDF 预解析：正在调用 OpenDataLoader 生成结构层元素级 JSON…")
        preparse_pages, preparse_objects, preparse_report = read_pdf_pages_with_opendataloader(
            input_pdf,
            config=pdf_config.get("opendataloader") or {},
            clean_text=True,
        )
        if preparse_report is not None:
            preparse_report["text_source"] = "pdfplumber"
            preparse_report["structure_source"] = "opendataloader"
            preparse_report["opendataloader_page_count"] = len(preparse_pages)
        _progress(
            "PDF 预解析：OpenDataLoader 完成"
            f"（页数 {preparse_report.get('page_count', len(preparse_pages)) if preparse_report else len(preparse_pages)}，"
            f"元素数 {preparse_report.get('total_elements', 0) if preparse_report else 0}，"
            f"图表对象 {len(preparse_objects)}）。"
        )
        _progress(
            f"读取 PDF 正文：使用 pdfplumber 双栏修复"
            f"（双栏={pdf_config.get('enable_double_column', True)}，"
            f"布局={_return_layout_enabled(pdf_config)}）…"
        )
        pages = read_pdf_pages(
            input_pdf,
            enable_double_column=bool(pdf_config.get("enable_double_column", True)),
            clean_text=True,
            return_layout=_return_layout_enabled(pdf_config),
            ocr_missing_high_value_text=bool(pdf_config.get("ocr_missing_high_value_text", False)),
            unit_exponent_ocr_config=pdf_config.get("unit_exponent_ocr_recovery") or {},
        )
    else:
        _progress(f"读取 PDF（双栏={pdf_config.get('enable_double_column', True)}，布局={_return_layout_enabled(pdf_config)}）…")
        pages = read_pdf_pages(
            input_pdf,
            enable_double_column=bool(pdf_config.get("enable_double_column", True)),
            clean_text=True,
            return_layout=_return_layout_enabled(pdf_config),
            ocr_missing_high_value_text=bool(pdf_config.get("ocr_missing_high_value_text", False)),
            unit_exponent_ocr_config=pdf_config.get("unit_exponent_ocr_recovery") or {},
        )
    _progress(f"读取 PDF 完成（页数: {len(pages)}）。")

    text_quality_report = analyze_parsed_pdf_text_quality(
        pages,
        pdf_config.get("skip_if_garbled_text") or {},
    )
    if bool(text_quality_report.get("skip", False)):
        _progress(
            "PDF text quality: skipped before chunking "
            f"(status={text_quality_report.get('status')}, reasons={text_quality_report.get('reasons')})."
        )
        return _write_skipped_pdf_outputs(
            input_pdf=input_pdf,
            output_path=output_path,
            text_quality_report=text_quality_report,
            preparse_report=preparse_report,
        )

    prescreen_report: dict[str, Any] | None = None
    if prescreen_on:
        _progress("语义章节 LLM 复筛：正在调用模型（噪声剔除 + 英文译中文，可能较慢）…")
        prescreen_report = apply_semantic_section_llm_prescreen(
            pages,
            pdf_config=pdf_config,
            llm_config_path=llm_config_path,
        )
        status = str(prescreen_report.get("status", ""))
        noise_n = int(prescreen_report.get("noise_fragment_count", 0) or 0)
        _progress(f"语义章节 LLM 复筛：结束（status={status}，噪声片段记录数: {noise_n}）。")
    else:
        _progress("语义章节 LLM 复筛：已关闭（配置 semantic_section_llm.enabled=false）。")

    _progress("页模型增强：进入翻译 / 全页 OCR 子流程…")
    pages = enrich_pages_with_models(
        pages,
        input_pdf=input_pdf,
        pdf_config=pdf_config,
        skip_translation=prescreen_on,
        skip_ocr=prescreen_on,
    )

    _progress("语义 chunk 切分：正在构建 chunk 列表…")
    payload = build_chunk_payload(
        pages,
        pdf_path=input_pdf,
        chunking_config=chunking_config,
        pdf_config=pdf_config,
    )
    payload["text_quality_report"] = text_quality_report
    n_chunks = int(payload.get("total_chunks", 0) or 0)
    _progress(f"语义 chunk 切分：完成（chunk 数: {n_chunks}）。")
    if prescreen_report is not None:
        payload["semantic_section_llm_report"] = prescreen_report
    if preparse_report is not None:
        payload["preparse_report"] = preparse_report

    payload = enrich_payload_with_figure_table_links(
        payload,
        pages=pages,
        input_pdf=input_pdf,
        pdf_config=pdf_config,
        preparse_objects=preparse_objects,
    )

    ocr_cfg = pdf_config.get("figure_table_ocr") or {}
    if table_parsing_enabled and bool(ocr_cfg.get("enabled", False)):
        _progress("关联图/表 OCR：正在按需调用视觉模型（可能较慢）…")
    else:
        _progress("关联图/表 OCR：已关闭（figure_table_ocr.enabled=false），跳过。")
    if table_parsing_enabled:
        payload, figure_ocr_entries = run_linked_object_ocr_and_inject(
            payload,
            input_pdf,
            pdf_config,
            llm_config_path=llm_config_path,
            pages=pages,
        )
    else:
        figure_ocr_entries = []
    _progress(f"关联图/表 OCR：结束（解析记录条数: {len(figure_ocr_entries)}）。")

    _progress(f"写入 chunk JSON: {output_path}")
    write_json(payload, output_path)

    figure_ocr_cfg = pdf_config.get("figure_table_ocr") or {}
    if table_parsing_enabled and bool(figure_ocr_cfg.get("enabled", False)):
        figure_ocr_path = output_path.with_name(f"{output_path.stem}.figure_ocr.json")
        write_json(
            {
                "doc_id": payload.get("doc_id"),
                "pdf_path": str(payload.get("pdf_path", "")),
                "entries": figure_ocr_entries,
            },
            figure_ocr_path,
        )
        _progress(f"写入图表解析 JSON: {figure_ocr_path}")

    _progress("校验 chunk 并生成 validation / heading 报告…")
    report = validate_chunk_payload(
        payload,
        min_chars=int(chunking_config.get("min_chars", 200)),
        max_chars=int(chunking_config.get("max_chars", 1200)),
    )
    report_path = output_path.with_suffix(".validation.json")
    write_json(report, report_path)
    heading_report = build_heading_hit_report(payload)
    heading_report_path = output_path.with_suffix(".heading_report.json")
    write_json(heading_report, heading_report_path)
    _progress(
        f"全部完成: validation_pass={report['pass']} | "
        f"heading_hit_rate={heading_report['metrics']['heading_hit_rate']} | "
        f"validation={report_path.name} | heading={heading_report_path.name}"
    )
    return {"validation": report, "heading_report": heading_report}


def _iter_pdfs(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("*.pdf") if path.is_file())


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="构建 PDF chunk.json")
    parser.add_argument("--input", type=Path, help="单个 PDF 输入路径")
    parser.add_argument("--output", type=Path, help="单个 chunk JSON 输出路径")
    parser.add_argument("--input_dir", type=Path, help="批量 PDF 输入目录")
    parser.add_argument("--output_dir", type=Path, help="批量 chunk JSON 输出目录")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "pdf_pipeline.yaml", help="PDF pipeline 配置")
    return parser.parse_args()


def _clean_cli_path(path: Path | None) -> Path | None:
    """Remove accidental surrounding whitespace from CLI path arguments."""
    if path is None:
        return None
    cleaned = str(path).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return Path(cleaned)


def main() -> int:
    """执行 CLI。"""
    args = parse_args()
    args.input = _clean_cli_path(args.input)
    args.output = _clean_cli_path(args.output)
    args.input_dir = _clean_cli_path(args.input_dir)
    args.output_dir = _clean_cli_path(args.output_dir)
    args.config = _clean_cli_path(args.config)
    config = _load_config(args.config)

    if args.input:
        if not args.output:
            raise ValueError("使用 --input 时必须提供 --output。")
        _progress("单文件模式")
        reports = build_one_pdf(args.input, args.output, config)
        print(f"chunk_json={args.output}")
        print(f"validation_pass={reports['validation']['pass']}")
        print(f"heading_hit_rate={reports['heading_report']['metrics']['heading_hit_rate']}")
        return 0

    if args.input_dir:
        if not args.output_dir:
            raise ValueError("使用 --input_dir 时必须提供 --output_dir。")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pdfs = _iter_pdfs(args.input_dir)
        if not pdfs:
            raise FileNotFoundError(f"目录中未找到 PDF: {args.input_dir}")
        summary_rows: list[dict[str, Any]] = []
        total = len(pdfs)
        for index, pdf_path in enumerate(pdfs, start=1):
            _progress(f"批量模式 [{index}/{total}] 当前文件: {pdf_path.name}")
            output_path = args.output_dir / f"{pdf_path.stem}.chunk.json"
            reports = build_one_pdf(pdf_path, output_path, config)
            heading_metrics = reports["heading_report"]["metrics"]
            summary_rows.append(
                {
                    "文档名": pdf_path.name,
                    "输出文件": str(output_path),
                    "validation_pass": reports["validation"]["pass"],
                    "total_chunks": int(heading_metrics["total_chunks"]),
                    "heading_hit_rate": float(heading_metrics["heading_hit_rate"]),
                    "hierarchical_hit_rate": float(heading_metrics["hierarchical_hit_rate"]),
                    "weighted_hit_score": float(heading_metrics["weighted_hit_score"]),
                }
            )
            print(
                " ".join(
                    [
                        f"chunk_json={output_path}",
                        f"validation_pass={reports['validation']['pass']}",
                        f"heading_hit_rate={reports['heading_report']['metrics']['heading_hit_rate']}",
                    ]
                )
            )
        summary_payload = _build_heading_summary_report(summary_rows)
        summary_path = args.output_dir / "_heading_report_summary.json"
        write_json(summary_payload, summary_path)
        print(f"批量汇总文件={summary_path}")
        print(f"文档总数={summary_payload['总文档数']} 平均章节命中率={summary_payload['平均章节命中率']}")
        return 0

    raise ValueError("必须提供 --input 或 --input_dir。")


if __name__ == "__main__":
    raise SystemExit(main())
