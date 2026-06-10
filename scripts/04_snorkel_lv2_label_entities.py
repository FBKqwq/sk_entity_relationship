"""Run Snorkel Lv2 entity typing and post-Lv2 property extraction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.entity_extraction.entity_property_extractor import extract_entity_properties
from src.utils.io import _json_default, read_json, write_json, write_jsonl
from src.weak_supervision.common.snorkel_preflight import official_snorkel_preflight
from src.weak_supervision_lv2.entity_signal_builder import build_entity_label_results, chunk_by_id

PropertyExtractor = Callable[..., tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]]


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_chunks(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)], {}
    if not isinstance(payload, dict):
        raise ValueError(f"Chunk JSON must be an object or list: {path}")
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError(f"Chunk JSON field `chunks` must be a list: {path}")
    return [item for item in chunks if isinstance(item, dict)], payload


def _base_name(path: Path) -> str:
    return path.name.removesuffix(".entity_base.jsonl") if path.name.endswith(".entity_base.jsonl") else path.stem


def _default_chunks_path(entity_base_path: Path, chunks_dir: Path) -> Path:
    return chunks_dir / f"{_base_name(entity_base_path)}.chunk.json"


def _default_label_path(entity_base_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{_base_name(entity_base_path)}.entity_label_result.jsonl"


def _default_property_path(entity_base_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{_base_name(entity_base_path)}.entity_property_result.jsonl"


def _default_nodes_path(entity_base_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{_base_name(entity_base_path)}.entity_nodes.jsonl"


def _default_conflicts_path(entity_base_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{_base_name(entity_base_path)}.entity_conflicts.jsonl"


def _default_recall_path(entity_base_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{_base_name(entity_base_path)}.entity_recall_report.json"


def _default_raw_path(entity_base_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{_base_name(entity_base_path)}.entity_property_raw.jsonl"


def run_lv2_for_file(
    *,
    entities_path: str | Path,
    chunks_path: str | Path,
    label_output_path: str | Path,
    property_output_path: str | Path,
    nodes_output_path: str | Path,
    conflicts_output_path: str | Path,
    recall_report_path: str | Path,
    raw_output_path: str | Path | None = None,
    config_path: str | Path | None = None,
    include_review_properties: bool = False,
    accept_threshold: float = 0.65,
    review_threshold: float = 0.40,
    min_top2_gap: float = 0.15,
    property_extractor: PropertyExtractor = extract_entity_properties,
) -> dict[str, Any]:
    """Run Lv2 typing plus post-Lv2 entity property extraction for one file."""

    entities = _read_jsonl(entities_path)
    chunks, chunk_payload = _load_chunks(chunks_path)
    label_rows, lv2_conflicts, report = build_entity_label_results(
        entities,
        chunks,
        accept_threshold=accept_threshold,
        review_threshold=review_threshold,
        min_top2_gap=min_top2_gap,
        config_path=config_path,
    )
    base_by_id = {str(entity.get("entity_id") or ""): entity for entity in entities if entity.get("entity_id")}
    property_rows, entity_nodes, property_conflicts, raw_rows = property_extractor(
        label_rows,
        base_by_id,
        chunk_by_id(chunks),
        include_review=include_review_properties,
        config_path=config_path,
    )
    conflicts = [*lv2_conflicts, *property_conflicts]
    write_jsonl(label_rows, label_output_path)
    write_jsonl(property_rows, property_output_path)
    write_jsonl(entity_nodes, nodes_output_path)
    write_jsonl(conflicts, conflicts_output_path)
    write_json(report, recall_report_path)
    if raw_output_path is not None:
        write_jsonl(raw_rows, raw_output_path)
    summary = {
        "entities_path": str(entities_path),
        "chunks_path": str(chunks_path),
        "source_pdf": chunk_payload.get("pdf_path") or chunk_payload.get("source_title"),
        "entity_candidates": len(entities),
        "label_results": len(label_rows),
        "entity_nodes": len(entity_nodes),
        "conflicts": len(conflicts),
        "accepted": report["accepted"],
        "review": report["review"],
        "rejected": report["rejected"],
        "include_review_properties": include_review_properties,
        "official_snorkel": official_snorkel_preflight(),
        "label_output_path": str(label_output_path),
        "property_output_path": str(property_output_path),
        "nodes_output_path": str(nodes_output_path),
        "conflicts_output_path": str(conflicts_output_path),
        "recall_report_path": str(recall_report_path),
    }
    return summary


def run_lv2_for_directory(
    *,
    entities_dir: str | Path,
    chunks_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    include_review_properties: bool = False,
    accept_threshold: float = 0.65,
    review_threshold: float = 0.40,
    min_top2_gap: float = 0.15,
    summary_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    entity_root = Path(entities_dir)
    chunks_root = Path(chunks_dir)
    output_root = Path(output_dir)
    summaries: list[dict[str, Any]] = []
    for entities_path in sorted(entity_root.glob("*.entity_base.jsonl")):
        chunks_path = _default_chunks_path(entities_path, chunks_root)
        if not chunks_path.exists():
            continue
        summaries.append(
            run_lv2_for_file(
                entities_path=entities_path,
                chunks_path=chunks_path,
                label_output_path=_default_label_path(entities_path, output_root),
                property_output_path=_default_property_path(entities_path, output_root),
                nodes_output_path=_default_nodes_path(entities_path, output_root),
                conflicts_output_path=_default_conflicts_path(entities_path, output_root),
                recall_report_path=_default_recall_path(entities_path, output_root),
                raw_output_path=_default_raw_path(entities_path, output_root),
                config_path=config_path,
                include_review_properties=include_review_properties,
                accept_threshold=accept_threshold,
                review_threshold=review_threshold,
                min_top2_gap=min_top2_gap,
            )
        )
    if summary_path is not None:
        write_json(summaries, summary_path)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Lv2 entity typing and post-Lv2 entity property extraction."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--entities", help="Single *.entity_base.jsonl file.")
    input_group.add_argument("--entities-dir", help="Directory containing *.entity_base.jsonl files.")
    parser.add_argument("--chunks", help="Single *.chunk.json file.")
    parser.add_argument("--chunks-dir", default="data/chunks")
    parser.add_argument("--output-dir", default="data/entity_nodes")
    parser.add_argument("--label-output", help="Single output *.entity_label_result.jsonl file.")
    parser.add_argument("--property-output", help="Single output *.entity_property_result.jsonl file.")
    parser.add_argument("--nodes-output", help="Single output *.entity_nodes.jsonl file.")
    parser.add_argument("--conflicts-output", help="Single output *.entity_conflicts.jsonl file.")
    parser.add_argument("--recall-report", help="Single output *.entity_recall_report.json file.")
    parser.add_argument("--raw-output", help="Single output property raw trace JSONL file.")
    parser.add_argument("--summary", help="Optional summary JSON path.")
    parser.add_argument("--config", default="configs/llm.yaml")
    parser.add_argument("--include-review-properties", action="store_true")
    parser.add_argument("--accept-threshold", type=float, default=0.65)
    parser.add_argument("--review-threshold", type=float, default=0.40)
    parser.add_argument("--min-top2-gap", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.entities:
        if not args.chunks:
            raise SystemExit("--entities mode requires --chunks")
        output_root = Path(args.output_dir)
        entity_path = Path(args.entities)
        summary = run_lv2_for_file(
            entities_path=args.entities,
            chunks_path=args.chunks,
            label_output_path=args.label_output or _default_label_path(entity_path, output_root),
            property_output_path=args.property_output or _default_property_path(entity_path, output_root),
            nodes_output_path=args.nodes_output or _default_nodes_path(entity_path, output_root),
            conflicts_output_path=args.conflicts_output or _default_conflicts_path(entity_path, output_root),
            recall_report_path=args.recall_report or _default_recall_path(entity_path, output_root),
            raw_output_path=args.raw_output or _default_raw_path(entity_path, output_root),
            config_path=args.config,
            include_review_properties=args.include_review_properties,
            accept_threshold=args.accept_threshold,
            review_threshold=args.review_threshold,
            min_top2_gap=args.min_top2_gap,
        )
        if args.summary:
            write_json(summary, args.summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
        return
    summaries = run_lv2_for_directory(
        entities_dir=args.entities_dir,
        chunks_dir=args.chunks_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        include_review_properties=args.include_review_properties,
        accept_threshold=args.accept_threshold,
        review_threshold=args.review_threshold,
        min_top2_gap=args.min_top2_gap,
        summary_path=args.summary,
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
