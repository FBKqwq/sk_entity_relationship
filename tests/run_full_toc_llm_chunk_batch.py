"""Batch-run pdf->chunk with the dual TOC LLM pipeline and classified outputs.

Temporary operator script. Intended to be launched from the code/ directory.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import read_yaml, write_json

BUILD_SCRIPT = ROOT / "scripts" / "01_build_chunks.py"
DEFAULT_INPUT_DIR = ROOT / "data" / "raw_pdfs" / "F"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "chunks" / "S"
DEFAULT_CONFIG = ROOT / "configs" / "pdf_pipeline.yaml"


def _load_build_module() -> Any:
    spec = importlib.util.spec_from_file_location("build_chunks_script", BUILD_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load build script: {BUILD_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clean_stem(stem: str) -> str:
    bad = '<>:"/\\|?*'
    cleaned = "".join("_" if ch in bad else ch for ch in stem).strip()
    return cleaned or "document"


def _classify_outputs(work_path: Path, output_root: Path) -> dict[str, str | None]:
    folders = {
        "chunk_json": output_root / "chunk_json",
        "figure_ocr_json": output_root / "figure_ocr_json",
        "heading_report_json": output_root / "heading_report_json",
        "validation_json": output_root / "validation_json",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    stem = work_path.stem
    paths = {
        "chunk_json": work_path,
        "figure_ocr_json": work_path.with_name(f"{stem}.figure_ocr.json"),
        "heading_report_json": work_path.with_suffix(".heading_report.json"),
        "validation_json": work_path.with_suffix(".validation.json"),
    }

    moved: dict[str, str | None] = {}
    for key, src in paths.items():
        if not src.exists():
            moved[key] = None
            continue
        dst = folders[key] / src.name
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
        moved[key] = str(dst)
    return moved


def _prepare_config(
    config_path: Path,
    *,
    force_pdfplumber: bool,
    enable_unit_exponent_ocr: bool,
    enable_high_value_ocr: bool,
) -> dict[str, Any]:
    config = read_yaml(config_path)
    pdf_cfg = config.setdefault("pdf", {})
    toc_cfg = pdf_cfg.setdefault("toc_llm_pipeline", {})
    toc_cfg["enabled"] = True
    toc_cfg.setdefault("first_layer_llm", {})["enabled"] = True
    toc_cfg.setdefault("judge_llm", {})["enabled"] = True
    if not enable_unit_exponent_ocr:
        pdf_cfg.setdefault("unit_exponent_ocr_recovery", {})["enabled"] = False
    if not enable_high_value_ocr:
        pdf_cfg["ocr_missing_high_value_text"] = False
    if force_pdfplumber or shutil.which(str(pdf_cfg.get("opendataloader", {}).get("command", "opendataloader-pdf"))) is None:
        pdf_cfg["parser"] = "pdfplumber"
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch pdf->chunk with dual TOC LLM and classify outputs.")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--failed_from",
        type=Path,
        default=None,
        help="Optional _batch_summary.json path. When set, rerun only rows whose status is not ok.",
    )
    parser.add_argument("--force_pdfplumber", action="store_true", help="Force pdfplumber instead of OpenDataLoader.")
    parser.add_argument(
        "--enable_unit_exponent_ocr",
        action="store_true",
        help="Enable pypdfium-based local unit exponent OCR recovery. Disabled by default for batch stability.",
    )
    parser.add_argument(
        "--enable_high_value_ocr",
        action="store_true",
        help="Enable pypdfium/tesseract fallback for missing high-value headings. Disabled by default for batch stability.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N PDFs, 0 means all.")
    parser.add_argument("--workers", type=int, default=5, help="Concurrent PDF workers. Default: 5.")
    return parser.parse_args()


def _process_one(
    *,
    index: int,
    total: int,
    pdf_path: Path,
    work_dir: Path,
    output_root: Path,
    base_config: dict[str, Any],
    summary_path: Path,
    rows: list[dict[str, Any]],
    rows_lock: threading.Lock,
) -> dict[str, Any]:
    clean_stem = _clean_stem(pdf_path.stem)
    work_path = work_dir / f"{clean_stem}.chunk.json"
    print(f"[batch] [{index}/{total}] start {pdf_path}", flush=True)
    row: dict[str, Any] = {
        "index": index,
        "total": total,
        "pdf": str(pdf_path),
        "status": "running",
        "outputs": {},
        "error": "",
    }
    try:
        build_module = _load_build_module()
        reports = build_module.build_one_pdf(pdf_path, work_path, copy.deepcopy(base_config))
        row["validation_pass"] = reports.get("validation", {}).get("pass")
        row["heading_hit_rate"] = reports.get("heading_report", {}).get("metrics", {}).get("heading_hit_rate")
        row["outputs"] = _classify_outputs(work_path, output_root)
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - batch should continue and record failures.
        row["status"] = "failed"
        row["error"] = repr(exc)
        print(f"[batch] failed: {pdf_path} | {exc!r}", flush=True)

    with rows_lock:
        rows.append(row)
        rows.sort(key=lambda item: int(item.get("index", 0)))
        write_json({"rows": rows}, summary_path)
    print(f"[batch] [{index}/{total}] {row['status']} {pdf_path}", flush=True)
    return row


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_root = args.output_root.resolve()
    work_dir = output_root / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.failed_from:
        summary = read_yaml(args.failed_from.resolve()) if args.failed_from.suffix.lower() in {".yaml", ".yml"} else None
        if summary is None:
            import json

            with args.failed_from.resolve().open("r", encoding="utf-8") as file:
                summary = json.load(file)
        pdfs = [
            Path(row["pdf"])
            for row in summary.get("rows", [])
            if row.get("status") != "ok" and row.get("pdf")
        ]
    else:
        pdfs = sorted(input_dir.rglob("*.pdf"))
    if args.limit > 0:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found under: {input_dir}")

    config = _prepare_config(
        args.config.resolve(),
        force_pdfplumber=bool(args.force_pdfplumber),
        enable_unit_exponent_ocr=bool(args.enable_unit_exponent_ocr),
        enable_high_value_ocr=bool(args.enable_high_value_ocr),
    )

    rows: list[dict[str, Any]] = []
    total = len(pdfs)
    workers = max(1, int(args.workers))
    summary_path = output_root / "_batch_summary.json"
    rows_lock = threading.Lock()
    print(f"[batch] total={total} workers={workers} input={input_dir} output={output_root}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _process_one,
                index=index,
                total=total,
                pdf_path=pdf_path,
                work_dir=work_dir,
                output_root=output_root,
                base_config=config,
                summary_path=summary_path,
                rows=rows,
                rows_lock=rows_lock,
            )
            for index, pdf_path in enumerate(pdfs, start=1)
        ]
        for future in as_completed(futures):
            future.result()

    ok = sum(1 for row in rows if row["status"] == "ok")
    failed = len(rows) - ok
    print(f"[batch] done. ok={ok} failed={failed} summary={output_root / '_batch_summary.json'}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
