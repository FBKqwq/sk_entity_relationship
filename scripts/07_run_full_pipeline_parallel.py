"""Run the full graph pipeline with stage-level document parallelism."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import write_json  # noqa: E402


_PRINT_LOCK = threading.Lock()


def _console(message: str) -> None:
    with _PRINT_LOCK:
        print(message, flush=True)


def _safe_stem(path: Path, seen: dict[str, int]) -> str:
    stem = path.stem.strip() or hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    count = seen.get(stem, 0)
    seen[stem] = count + 1
    if count == 0:
        return stem
    suffix = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{stem}.{suffix}"


def _iter_pdfs(input_dir: Path, *, recursive: bool) -> list[Path]:
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def _ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "chunks": output_dir / "chunks",
        "lv1": output_dir / "lv1",
        "entity_base": output_dir / "entity_base",
        "entity_nodes": output_dir / "entity_nodes",
        "relationship_base": output_dir / "relationship_base",
        "summaries": output_dir / "summaries",
        "logs": output_dir / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _run_command(command: list[str], *, log_path: Path, cwd: Path, doc_stem: str, stage: str) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        log.write(f"\n[{stamp}] STAGE {stage}\n")
        log.write(f"[{stamp}] CMD {' '.join(command)}\n")
        log.flush()
        _console(f"[{doc_stem}][{stage}] start")
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip("\r\n")
            if not line:
                continue
            log.write(line + "\n")
            log.flush()
            _console(f"[{doc_stem}][{stage}] {line}")
        return_code = process.wait()
        if return_code != 0:
            _console(f"[{doc_stem}][{stage}] failed exit_code={return_code}")
            raise RuntimeError(f"command failed exit_code={return_code}: {' '.join(command)}")
        _console(f"[{doc_stem}][{stage}] done")


def _doc_paths(dirs: dict[str, Path], doc_stem: str) -> dict[str, Path]:
    return {
        "chunk": dirs["chunks"] / f"{doc_stem}.chunk.json",
        "lv1": dirs["lv1"] / f"{doc_stem}.chunk_label_result.jsonl",
        "lv1_lfs": dirs["lv1"] / f"{doc_stem}.lv1_lf_outputs.jsonl",
        "entity_base": dirs["entity_base"] / f"{doc_stem}.entity_base.jsonl",
        "teacher_raw": dirs["entity_base"] / f"{doc_stem}.teacher_llm_raw.jsonl",
        "entity_label": dirs["entity_nodes"] / f"{doc_stem}.entity_label_result.jsonl",
        "entity_property": dirs["entity_nodes"] / f"{doc_stem}.entity_property_result.jsonl",
        "entity_nodes": dirs["entity_nodes"] / f"{doc_stem}.entity_nodes.jsonl",
        "entity_conflicts": dirs["entity_nodes"] / f"{doc_stem}.entity_conflicts.jsonl",
        "entity_recall": dirs["entity_nodes"] / f"{doc_stem}.entity_recall_report.json",
        "entity_property_raw": dirs["entity_nodes"] / f"{doc_stem}.entity_property_raw.jsonl",
        "relationship": dirs["relationship_base"] / f"{doc_stem}.relationship_base.jsonl",
        "relationship_candidates": dirs["relationship_base"] / f"{doc_stem}.candidate_relationships.jsonl",
        "relationship_raw": dirs["relationship_base"] / f"{doc_stem}.relationship_llm_raw.jsonl",
        "log": dirs["logs"] / f"{doc_stem}.pipeline.log",
        "stage_status": dirs["logs"] / f"{doc_stem}.stage_status.json",
    }


STAGES = ("01_chunk", "02_lv1", "03_entity_base", "04_lv2_nodes", "06_relationships")


def _stage_outputs(paths: dict[str, Path], stage: str) -> list[Path]:
    if stage == "01_chunk":
        return [paths["chunk"]]
    if stage == "02_lv1":
        return [paths["lv1"], paths["lv1_lfs"]]
    if stage == "03_entity_base":
        return [paths["entity_base"]]
    if stage == "04_lv2_nodes":
        return [paths["entity_label"], paths["entity_property"], paths["entity_nodes"], paths["entity_recall"]]
    if stage == "06_relationships":
        return [paths["relationship"]]
    raise ValueError(f"unknown stage: {stage}")


def _stage_inputs(paths: dict[str, Path], stage: str) -> list[Path]:
    if stage == "01_chunk":
        return []
    if stage == "02_lv1":
        return [paths["chunk"]]
    if stage == "03_entity_base":
        return [paths["chunk"], paths["lv1"]]
    if stage == "04_lv2_nodes":
        return [paths["chunk"], paths["entity_base"]]
    if stage == "06_relationships":
        return [paths["chunk"], paths["entity_nodes"]]
    raise ValueError(f"unknown stage: {stage}")


def _stage_command(
    *,
    pdf_path: Path,
    paths: dict[str, Path],
    stage: str,
    python_exe: str,
    pdf_config: str,
    weak_config: str,
    llm_config: str,
    audit_batch_size: int,
    accepted_only: bool,
) -> list[str]:
    if stage == "01_chunk":
        return [
            python_exe,
            "scripts/01_build_chunks.py",
            "--input",
            str(pdf_path),
            "--output",
            str(paths["chunk"]),
            "--config",
            pdf_config,
        ]
    if stage == "02_lv1":
        return [
            python_exe,
            "scripts/02_snorkel_lv1_label_chunks.py",
            "--chunks",
            str(paths["chunk"]),
            "--output",
            str(paths["lv1"]),
            "--lf-output",
            str(paths["lv1_lfs"]),
            "--config",
            weak_config,
            "--llm-config",
            llm_config,
            "--enable-prompted-llm",
        ]
    if stage == "03_entity_base":
        return [
            python_exe,
            "scripts/03_llm_extract_entity_base.py",
            "--chunks",
            str(paths["chunk"]),
            "--lv1",
            str(paths["lv1"]),
            "--output",
            str(paths["entity_base"]),
            "--raw-output",
            str(paths["teacher_raw"]),
            "--config",
            llm_config,
            "--Full_extraction",
        ]
    if stage == "04_lv2_nodes":
        return [
            python_exe,
            "scripts/04_snorkel_lv2_label_entities.py",
            "--entities",
            str(paths["entity_base"]),
            "--chunks",
            str(paths["chunk"]),
            "--label-output",
            str(paths["entity_label"]),
            "--property-output",
            str(paths["entity_property"]),
            "--nodes-output",
            str(paths["entity_nodes"]),
            "--conflicts-output",
            str(paths["entity_conflicts"]),
            "--recall-report",
            str(paths["entity_recall"]),
            "--raw-output",
            str(paths["entity_property_raw"]),
            "--config",
            llm_config,
        ]
    if stage == "06_relationships":
        command = [
            python_exe,
            "scripts/06_llm_extract_relationship_base.py",
            "--entities",
            str(paths["entity_nodes"]),
            "--chunks",
            str(paths["chunk"]),
            "--output",
            str(paths["relationship"]),
            "--candidate-output",
            str(paths["relationship_candidates"]),
            "--raw-output",
            str(paths["relationship_raw"]),
            "--config",
            llm_config,
            "--audit-batch-size",
            str(audit_batch_size),
        ]
        if accepted_only:
            command.append("--accepted-only")
        return command
    raise ValueError(f"unknown stage: {stage}")


def _load_stage_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_stage_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(payload, path)


def _run_stage_for_document(
    *,
    pdf_path: Path,
    doc_stem: str,
    paths: dict[str, Path],
    stage: str,
    python_exe: str,
    pdf_config: str,
    weak_config: str,
    llm_config: str,
    audit_batch_size: int,
    accepted_only: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    started = time.time()
    status_payload = _load_stage_status(paths["stage_status"])
    stage_payload = status_payload.setdefault("stages", {})
    log_path = paths["log"]
    missing_inputs = [path for path in _stage_inputs(paths, stage) if not path.exists()]
    if missing_inputs:
        error = "missing inputs: " + ", ".join(str(path) for path in missing_inputs)
        row = {
            "pdf_path": str(pdf_path),
            "document_stem": doc_stem,
            "stage": stage,
            "status": "blocked",
            "error": error,
            "elapsed_seconds": 0.0,
        }
        stage_payload[stage] = {**row, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _write_stage_status(paths["stage_status"], status_payload)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S')}] BLOCKED {stage} {error}\n")
        _console(f"[{doc_stem}][{stage}] blocked {error}")
        return row

    expected_outputs = _stage_outputs(paths, stage)
    if skip_existing and all(path.exists() for path in expected_outputs):
        row = {
            "pdf_path": str(pdf_path),
            "document_stem": doc_stem,
            "stage": stage,
            "status": "skipped",
            "error": "",
            "elapsed_seconds": 0.0,
        }
        stage_payload[stage] = {**row, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _write_stage_status(paths["stage_status"], status_payload)
        _console(f"[{doc_stem}][{stage}] skip existing")
        return row

    status = "ok"
    error = ""
    command = _stage_command(
        pdf_path=pdf_path,
        paths=paths,
        stage=stage,
        python_exe=python_exe,
        pdf_config=pdf_config,
        weak_config=weak_config,
        llm_config=llm_config,
        audit_batch_size=audit_batch_size,
        accepted_only=accepted_only,
    )
    try:
        _run_command(command, log_path=log_path, cwd=ROOT, doc_stem=doc_stem, stage=stage)
    except Exception as exc:  # noqa: BLE001 - summarize per-stage failures.
        status = "failed"
        error = str(exc)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S')}] FAILED {stage} {error}\n")
        _console(f"[{doc_stem}][{stage}] failed {error}")

    elapsed = round(time.time() - started, 3)
    row = {
        "pdf_path": str(pdf_path),
        "document_stem": doc_stem,
        "stage": stage,
        "status": status,
        "error": error,
        "elapsed_seconds": elapsed,
    }
    stage_payload[stage] = {**row, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    status_payload["pdf_path"] = str(pdf_path)
    status_payload["document_stem"] = doc_stem
    status_payload["outputs"] = {
        key: str(value) for key, value in paths.items() if key not in {"log", "stage_status"}
    }
    _write_stage_status(paths["stage_status"], status_payload)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full PDF-to-relationship pipeline with stage-level document concurrency.")
    parser.add_argument("--input-dir", required=True, help="Directory containing input PDFs.")
    parser.add_argument("--output-dir", required=True, help="Root output directory for all pipeline stages.")
    parser.add_argument("--workers", type=int, default=5, help="Maximum PDFs processed concurrently within each stage.")
    parser.add_argument("--recursive", action="store_true", help="Recursively discover PDFs under input-dir.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of PDFs to process.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child stage scripts.")
    parser.add_argument("--pdf-config", default="configs/pdf_pipeline.yaml")
    parser.add_argument("--weak-config", default="configs/weak_supervision.yaml")
    parser.add_argument("--llm-config", default="configs/llm.yaml")
    parser.add_argument("--audit-batch-size", type=int, default=50)
    parser.add_argument("--include-review-relationships", action="store_true", help="Do not pass --accepted-only to relationship extraction.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip stages whose expected output already exists.")
    parser.add_argument(
        "--stop-on-stage-failure",
        action="store_true",
        help="Stop the whole run when any document fails in a stage. Default continues and blocks only dependent stages.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    dirs = _ensure_dirs(output_dir)
    pdfs = _iter_pdfs(input_dir, recursive=bool(args.recursive))
    if args.limit and args.limit > 0:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found under {input_dir}")
    workers = max(1, int(args.workers))
    seen: dict[str, int] = {}
    jobs = [(pdf_path, _safe_stem(pdf_path, seen)) for pdf_path in pdfs]
    doc_rows = [
        {
            "pdf_path": pdf_path,
            "document_stem": doc_stem,
            "paths": _doc_paths(dirs, doc_stem),
            "stages": {},
        }
        for pdf_path, doc_stem in jobs
    ]
    for row in doc_rows:
        log_path = row["paths"]["log"]
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S')}] RUN_START "
                f"stage_parallel_pipeline pdf={row['pdf_path']}\n"
            )

    stage_summaries: list[dict[str, Any]] = []
    _console(f"[parallel_pipeline] documents={len(jobs)} workers_per_stage={workers} stages={','.join(STAGES)}")
    for stage in STAGES:
        stage_started = time.time()
        _console(f"[parallel_pipeline][{stage}] START documents={len(doc_rows)} workers={workers}")
        stage_rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    _run_stage_for_document,
                    pdf_path=row["pdf_path"],
                    doc_stem=row["document_stem"],
                    paths=row["paths"],
                    stage=stage,
                    python_exe=str(args.python),
                    pdf_config=str(args.pdf_config),
                    weak_config=str(args.weak_config),
                    llm_config=str(args.llm_config),
                    audit_batch_size=int(args.audit_batch_size),
                    accepted_only=not bool(args.include_review_relationships),
                    skip_existing=bool(args.skip_existing),
                ): row
                for row in doc_rows
            }
            for future in as_completed(future_map):
                doc_row = future_map[future]
                try:
                    stage_row = future.result()
                except Exception as exc:  # noqa: BLE001 - top-level unexpected failure.
                    stage_row = {
                        "pdf_path": str(doc_row["pdf_path"]),
                        "document_stem": doc_row["document_stem"],
                        "stage": stage,
                        "status": "failed",
                        "error": str(exc),
                        "elapsed_seconds": 0.0,
                    }
                doc_row["stages"][stage] = stage_row
                stage_rows.append(stage_row)
                _console(
                    f"[parallel_pipeline][{stage}] {stage_row['status']} "
                    f"{stage_row['document_stem']} elapsed={stage_row.get('elapsed_seconds')}s"
                )
        stage_rows.sort(key=lambda item: item["document_stem"])
        stage_summary = {
            "stage": stage,
            "elapsed_seconds": round(time.time() - stage_started, 3),
            "documents": len(stage_rows),
            "ok": sum(1 for item in stage_rows if item["status"] == "ok"),
            "skipped": sum(1 for item in stage_rows if item["status"] == "skipped"),
            "blocked": sum(1 for item in stage_rows if item["status"] == "blocked"),
            "failed": sum(1 for item in stage_rows if item["status"] == "failed"),
            "rows": stage_rows,
        }
        stage_summaries.append(stage_summary)
        stage_summary_path = dirs["summaries"] / f"parallel_pipeline_{stage}_summary.json"
        write_json(stage_summary, stage_summary_path)
        _console(
            f"[parallel_pipeline][{stage}] DONE ok={stage_summary['ok']} skipped={stage_summary['skipped']} "
            f"blocked={stage_summary['blocked']} failed={stage_summary['failed']} summary={stage_summary_path}"
        )
        if args.stop_on_stage_failure and (stage_summary["failed"] or stage_summary["blocked"]):
            _console(f"[parallel_pipeline][{stage}] stop-on-stage-failure triggered")
            break

    summary_rows: list[dict[str, Any]] = []
    for row in doc_rows:
        stage_rows = row["stages"]
        terminal_status = "ok"
        errors: list[str] = []
        for stage in STAGES:
            stage_row = stage_rows.get(stage)
            if not stage_row:
                terminal_status = "not_run"
                continue
            if stage_row["status"] == "failed":
                terminal_status = "failed"
                errors.append(f"{stage}: {stage_row.get('error', '')}")
                break
            if stage_row["status"] == "blocked" and terminal_status == "ok":
                terminal_status = "blocked"
                errors.append(f"{stage}: {stage_row.get('error', '')}")
        summary_rows.append(
            {
                "pdf_path": str(row["pdf_path"]),
                "document_stem": row["document_stem"],
                "status": terminal_status,
                "error": " | ".join(errors),
                "stages": stage_rows,
                "outputs": {
                    key: str(value)
                    for key, value in row["paths"].items()
                    if key not in {"log", "stage_status"}
                },
                "log_path": str(row["paths"]["log"]),
                "stage_status_path": str(row["paths"]["stage_status"]),
            }
        )
    summary_rows.sort(key=lambda item: item["document_stem"])
    summary_path = dirs["summaries"] / "parallel_pipeline_summary.json"
    write_json(
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "workers": workers,
            "parallelism": "stage_level_document_parallel",
            "stages": list(STAGES),
            "stage_summaries": stage_summaries,
            "documents": len(summary_rows),
            "ok": sum(1 for row in summary_rows if row["status"] == "ok"),
            "failed": sum(1 for row in summary_rows if row["status"] != "ok"),
            "rows": summary_rows,
        },
        summary_path,
    )
    _console(f"[parallel_pipeline] summary={summary_path}")
    return 1 if any(row["status"] != "ok" for row in summary_rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
