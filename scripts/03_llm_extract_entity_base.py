"""Extract candidate entities with an LLM constrained by Lv1 outputs.

This is the main entry point for the Lv1 -> entity_base stage. It calls the
Teacher LLM one chunk at a time, using that chunk's Lv1 label decisions as the
extraction constraint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.entity_extraction.entity_base_builder import build_entity_base_records
from src.entity_extraction.llm_entity_extractor import extract_prelabeled_entities
from src.utils.io import _json_default, read_json, read_yaml, write_json, write_jsonl


Extractor = Callable[..., dict[str, Any]]


def _progress(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> Path:
    return write_jsonl(rows, path)


def _load_chunks(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, list):
        chunks = [item for item in payload if isinstance(item, dict)]
        return chunks, {}
    if not isinstance(payload, dict):
        raise ValueError(f"Chunk JSON must be an object or list: {path}")
    raw_chunks = payload.get("chunks", [])
    if not isinstance(raw_chunks, list):
        raise ValueError(f"Chunk JSON field `chunks` must be a list: {path}")
    chunks = [item for item in raw_chunks if isinstance(item, dict)]
    return chunks, payload


def _group_lv1_by_chunk(lv1_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in lv1_rows:
        chunk_id = str(row.get("chunk_id") or "")
        if not chunk_id:
            continue
        grouped.setdefault(chunk_id, []).append(row)
    return grouped


def _has_extractable_label(lv1_results: list[dict[str, Any]]) -> bool:
    for result in lv1_results:
        if bool(result.get("present")) and str(result.get("status")) in {"accepted", "weak"}:
            return True
    return False


def _chunk_id(chunk: dict[str, Any]) -> str:
    return str(chunk.get("chunk_id") or chunk.get("id") or "")


def _without_prompt(response: dict[str, Any], *, save_prompt: bool) -> dict[str, Any]:
    if save_prompt:
        return dict(response)
    return {key: value for key, value in response.items() if key != "prompt"}


def _full_extraction_from_config(config_path: str | Path | None) -> bool:
    """Read the default Full_extraction switch from llm.yaml."""

    path = Path(config_path or "configs/llm.yaml")
    if not path.exists():
        return False
    raw = read_yaml(path).get("teacher_llm", {})
    if not isinstance(raw, dict):
        return False
    return bool(raw.get("Full_extraction", False))


def extract_entity_base_for_file(
    *,
    chunks_path: str | Path,
    lv1_path: str | Path | None,
    output_path: str | Path,
    raw_output_path: str | Path | None = None,
    config_path: str | Path | None = None,
    include_check_only_chunks: bool = False,
    full_extraction: bool = False,
    save_prompts: bool = False,
    extractor: Extractor = extract_prelabeled_entities,
) -> dict[str, Any]:
    """Run chunk-by-chunk Teacher LLM extraction for one document."""

    chunks, chunk_payload = _load_chunks(chunks_path)
    if lv1_path is None:
        lv1_rows = []
    else:
        lv1_file = Path(lv1_path)
        if not lv1_file.exists():
            if not full_extraction:
                raise FileNotFoundError(f"Lv1 file does not exist: {lv1_path}")
            lv1_rows = []
        else:
            lv1_rows = _read_jsonl(lv1_file)
    lv1_by_chunk = _group_lv1_by_chunk(lv1_rows)
    document_id = chunk_payload.get("doc_id")
    source_pdf = (
        chunk_payload.get("pdf_path")
        or chunk_payload.get("source_title")
        or Path(chunks_path).name.removesuffix(".chunk.json")
    )
    core_disease = ""

    entity_records: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    called_chunks = 0
    skipped_chunks = 0
    failed_chunks = 0

    core_disease_entity = None
    if core_disease_entity is not None:
        first_chunk = chunks[0] if chunks else {}
        document_chunk = {
            **first_chunk,
            "chunk_id": "__DOC__",
            "document_id": document_id or first_chunk.get("document_id"),
            "section_title": "文档标题",
            "section_path": ["文档", "标题"],
        }
        entity_records.extend(
            build_entity_base_records(
                document_chunk,
                [core_disease_entity],
                source="document_title_rule",
            )
        )

    for index, chunk in enumerate(chunks, start=1):
        chunk_id = _chunk_id(chunk)
        enriched_chunk = dict(chunk)
        if document_id and not enriched_chunk.get("document_id"):
            enriched_chunk["document_id"] = document_id
        if core_disease:
            enriched_chunk["document_core_disease"] = core_disease
        lv1_results = lv1_by_chunk.get(chunk_id, [])
        if (
            not full_extraction
            and not include_check_only_chunks
            and not _has_extractable_label(lv1_results)
        ):
            skipped_chunks += 1
            continue

        called_chunks += 1
        _progress(f"extracting {Path(chunks_path).name} chunk {index}/{len(chunks)} ({chunk_id})")
        try:
            response = extractor(
                enriched_chunk,
                lv1_results,
                config_path=config_path,
                full_extraction=full_extraction,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch extraction moving.
            response = {
                "status": "error",
                "reason": str(exc),
                "entities": [],
            }

        raw_response = _without_prompt(response, save_prompt=save_prompts)
        raw_rows.append(
            {
                "chunk_id": chunk_id,
                "section_title": chunk.get("section_title"),
                "section_path": chunk.get("section_path", []),
                "lv1_results": lv1_results,
                "Full_extraction": full_extraction,
                "response": raw_response,
            }
        )
        if response.get("status") != "ok":
            if response.get("status") == "error":
                failed_chunks += 1
            continue

        entity_records.extend(
            build_entity_base_records(
                enriched_chunk,
                list(response.get("entities", [])),
                source="teacher_llm_prelabel",
            )
        )

    _write_jsonl(entity_records, output_path)
    if raw_output_path is not None:
        _write_jsonl(raw_rows, raw_output_path)

    summary = {
        "source_pdf": source_pdf,
        "core_disease": core_disease,
        "chunks_path": str(chunks_path),
        "lv1_path": str(lv1_path) if lv1_path is not None else None,
        "entity_path": str(output_path),
        "raw_output_path": str(raw_output_path) if raw_output_path else None,
        "chunks": len(chunks),
        "lv1_records": len(lv1_rows),
        "lv1_positive_chunks": sum(
            1 for results in lv1_by_chunk.values() if _has_extractable_label(results)
        ),
        "called_chunks": called_chunks,
        "skipped_chunks": skipped_chunks,
        "failed_chunks": failed_chunks,
        "entity_records": len(entity_records),
        "include_check_only_chunks": include_check_only_chunks,
        "Full_extraction": full_extraction,
    }
    return summary


def _default_lv1_path(chunks_path: Path, lv1_dir: Path) -> Path:
    return lv1_dir / f"{chunks_path.name.removesuffix('.chunk.json')}.chunk_label_result.jsonl"


def _default_entity_path(chunks_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{chunks_path.name.removesuffix('.chunk.json')}.entity_base.jsonl"


def _default_raw_path(chunks_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{chunks_path.name.removesuffix('.chunk.json')}.teacher_llm_raw.jsonl"


def extract_entity_base_for_directory(
    *,
    chunks_dir: str | Path,
    lv1_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    include_check_only_chunks: bool = False,
    full_extraction: bool = False,
    save_prompts: bool = False,
    write_raw: bool = True,
    summary_path: str | Path | None = None,
    extractor: Extractor = extract_prelabeled_entities,
) -> list[dict[str, Any]]:
    """Run extraction for every ``*.chunk.json`` file in a directory."""

    chunks_root = Path(chunks_dir)
    lv1_root = Path(lv1_dir)
    entity_root = Path(output_dir)
    summaries: list[dict[str, Any]] = []
    for chunks_path in sorted(chunks_root.glob("*.chunk.json")):
        lv1_path = _default_lv1_path(chunks_path, lv1_root)
        if not full_extraction and not lv1_path.exists():
            _progress(f"skipping {chunks_path.name}: missing Lv1 file {lv1_path}")
            continue
        effective_lv1_path = lv1_path if lv1_path.exists() else None
        output_path = _default_entity_path(chunks_path, entity_root)
        raw_path = _default_raw_path(chunks_path, entity_root) if write_raw else None
        summaries.append(
            extract_entity_base_for_file(
                chunks_path=chunks_path,
                lv1_path=effective_lv1_path,
                output_path=output_path,
                raw_output_path=raw_path,
                config_path=config_path,
                include_check_only_chunks=include_check_only_chunks,
                full_extraction=full_extraction,
                save_prompts=save_prompts,
                extractor=extractor,
            )
        )

    if summary_path is not None:
        write_json(summaries, summary_path)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract entity_base candidates from chunks using Lv1-constrained Teacher LLM calls."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--chunks", help="Single chunk JSON file.")
    input_group.add_argument("--chunks-dir", help="Directory containing *.chunk.json files.")
    parser.add_argument("--lv1", help="Single Lv1 *.chunk_label_result.jsonl file.")
    parser.add_argument("--lv1-dir", default="data/weak_signals", help="Directory containing Lv1 outputs.")
    parser.add_argument("--output", help="Single output *.entity_base.jsonl file.")
    parser.add_argument("--output-dir", default="data/entity_base", help="Directory for entity_base outputs.")
    parser.add_argument("--raw-output", help="Single raw trace *.teacher_llm_raw.jsonl file.")
    parser.add_argument("--no-raw", action="store_true", help="Do not write raw Teacher LLM trace JSONL.")
    parser.add_argument("--summary", help="Optional pipeline summary JSON path.")
    parser.add_argument("--config", default="configs/llm.yaml", help="Teacher LLM config YAML.")
    parser.add_argument(
        "--include-check-only-chunks",
        action="store_true",
        help="Call Teacher LLM even when no Lv1 label is accepted/weak for the chunk.",
    )
    parser.add_argument(
        "--Full_extraction",
        "--full-extraction",
        dest="full_extraction",
        action="store_true",
        default=None,
        help=(
            "Ignore Lv1 positive chunk filtering and extract all active entity types "
            "from every received chunk."
        ),
    )
    parser.add_argument(
        "--no-Full_extraction",
        "--no-full-extraction",
        dest="full_extraction",
        action="store_false",
        help="Disable Full_extraction and use Lv1 positive chunk constrained extraction.",
    )
    parser.add_argument("--save-prompts", action="store_true", help="Persist full prompts in raw trace output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_extraction = (
        _full_extraction_from_config(args.config)
        if args.full_extraction is None
        else bool(args.full_extraction)
    )
    if args.chunks:
        if not args.output:
            raise SystemExit("--chunks mode requires --output")
        if not full_extraction and not args.lv1:
            raise SystemExit("--chunks mode requires --lv1 unless --Full_extraction is set")
        raw_output = None if args.no_raw else args.raw_output
        if raw_output is None and not args.no_raw:
            raw_output = str(Path(args.output).with_suffix(".teacher_llm_raw.jsonl"))
        summary = extract_entity_base_for_file(
            chunks_path=args.chunks,
            lv1_path=args.lv1,
            output_path=args.output,
            raw_output_path=raw_output,
            config_path=args.config,
            include_check_only_chunks=args.include_check_only_chunks,
            full_extraction=full_extraction,
            save_prompts=args.save_prompts,
        )
        if args.summary:
            write_json(summary, args.summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
        return

    summaries = extract_entity_base_for_directory(
        chunks_dir=args.chunks_dir,
        lv1_dir=args.lv1_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        include_check_only_chunks=args.include_check_only_chunks,
        full_extraction=full_extraction,
        save_prompts=args.save_prompts,
        write_raw=not args.no_raw,
        summary_path=args.summary,
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
