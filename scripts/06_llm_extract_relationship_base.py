"""Extract relationship_base records from final entity_nodes and chunks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.relationship_extraction.relationship_base_builder import (  # noqa: E402
    extract_relationship_base_for_file,
)
from src.utils.io import _json_default, write_json  # noqa: E402

Extractor = Callable[..., dict[str, Any]]


def _progress(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def _base_name_from_entities(path: Path) -> str:
    name = path.name
    if name.endswith(".entity_nodes.jsonl"):
        return name.removesuffix(".entity_nodes.jsonl")
    if name.endswith(".entity_base.jsonl"):
        return name.removesuffix(".entity_base.jsonl")
    return path.stem


def _default_chunks_path(entities_path: Path, chunks_dir: Path) -> Path:
    return chunks_dir / f"{_base_name_from_entities(entities_path)}.chunk.json"


def _default_relationship_path(entities_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{_base_name_from_entities(entities_path)}.relationship_base.jsonl"


def _default_candidate_path(output_path: Path) -> Path:
    base = output_path.name.removesuffix(".relationship_base.jsonl")
    return output_path.with_name(f"{base}.candidate_relationships.jsonl")


def _default_raw_path(output_path: Path) -> Path:
    base = output_path.name.removesuffix(".relationship_base.jsonl")
    return output_path.with_name(f"{base}.relationship_llm_raw.jsonl")


def extract_relationship_base_for_directory(
    *,
    entities_dir: str | Path,
    chunks_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    write_raw: bool = True,
    write_candidates: bool = True,
    summary_path: str | Path | None = None,
    same_chunk_only: bool = False,
    max_candidates: int = 0,
    audit_batch_size: int = 20,
    include_review_entities: bool = True,
    extractor: Extractor = extract_relationship_base_for_file,
) -> list[dict[str, Any]]:
    """Run relationship extraction for every final entity_nodes file in a directory."""

    entity_root = Path(entities_dir)
    chunks_root = Path(chunks_dir)
    output_root = Path(output_dir)
    summaries: list[dict[str, Any]] = []
    entity_paths = sorted(entity_root.glob("*.entity_nodes.jsonl"))
    if not entity_paths:
        entity_paths = sorted(entity_root.glob("*.entity_base.jsonl"))
    for entities_path in entity_paths:
        chunks_path = _default_chunks_path(entities_path, chunks_root)
        if not chunks_path.exists():
            _progress(f"skipping {entities_path.name}: missing chunk file {chunks_path}")
            continue
        output_path = _default_relationship_path(entities_path, output_root)
        candidate_path = _default_candidate_path(output_path) if write_candidates else None
        raw_path = _default_raw_path(output_path) if write_raw else None
        _progress(f"extracting relationships for {entities_path.name}")
        summaries.append(
            extractor(
                entities_path=entities_path,
                chunks_path=chunks_path,
                output_path=output_path,
                candidate_output_path=candidate_path,
                raw_output_path=raw_path,
                config_path=config_path,
                same_chunk_only=same_chunk_only,
                max_candidates=max_candidates,
                audit_batch_size=audit_batch_size,
                include_review_entities=include_review_entities,
            )
        )

    if summary_path is not None:
        write_json(summaries, summary_path)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build relationship_base records from Lv2 entity_nodes and chunks using schema-constrained LLM audit."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--entities", help="Single *.entity_nodes.jsonl file; legacy *.entity_base.jsonl is accepted for tests/backfill.")
    input_group.add_argument("--entities-dir", default=None, help="Directory containing *.entity_nodes.jsonl files.")
    parser.add_argument("--chunks", help="Single *.chunk.json file.")
    parser.add_argument("--chunks-dir", default="data/chunks", help="Directory containing *.chunk.json files.")
    parser.add_argument("--output", help="Single output *.relationship_base.jsonl file.")
    parser.add_argument("--output-dir", default="data/relationship_base", help="Directory for relationship outputs.")
    parser.add_argument("--candidate-output", help="Single candidate_relationships JSONL path.")
    parser.add_argument("--raw-output", help="Single raw trace *.relationship_llm_raw.jsonl path.")
    parser.add_argument("--no-candidates", action="store_true", help="Do not write candidate_relationships JSONL.")
    parser.add_argument("--no-raw", action="store_true", help="Do not write raw LLM trace JSONL.")
    parser.add_argument("--summary", help="Optional relationship summary JSON path.")
    parser.add_argument("--config", default="configs/llm.yaml", help="Teacher LLM config YAML.")
    parser.add_argument(
        "--same-chunk-only",
        action="store_true",
        help="Only generate candidate relationships whose endpoints are in the same chunk. Default allows nearby cross-chunk candidates.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Maximum candidates to audit after layered sorting. 0 means no limit.",
    )
    parser.add_argument(
        "--audit-batch-size",
        type=int,
        default=20,
        help="Number of candidates to audit in one batched LLM call.",
    )
    parser.add_argument(
        "--accepted-only",
        action="store_true",
        help="Use only Lv2 accepted entities for relationships. Default also includes review entities to protect recall.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.entities:
        if not args.chunks:
            raise SystemExit("--entities mode requires --chunks")
        if not args.output:
            raise SystemExit("--entities mode requires --output")
        output_path = Path(args.output)
        candidate_output = None if args.no_candidates else args.candidate_output
        raw_output = None if args.no_raw else args.raw_output
        if candidate_output is None and not args.no_candidates:
            candidate_output = str(_default_candidate_path(output_path))
        if raw_output is None and not args.no_raw:
            raw_output = str(_default_raw_path(output_path))
        summary = extract_relationship_base_for_file(
            entities_path=args.entities,
            chunks_path=args.chunks,
            output_path=args.output,
            candidate_output_path=candidate_output,
            raw_output_path=raw_output,
            summary_path=args.summary,
            config_path=args.config,
            same_chunk_only=args.same_chunk_only,
            max_candidates=args.max_candidates,
            audit_batch_size=args.audit_batch_size,
            include_review_entities=not args.accepted_only,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
        return

    summaries = extract_relationship_base_for_directory(
        entities_dir=args.entities_dir,
        chunks_dir=args.chunks_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        write_raw=not args.no_raw,
        write_candidates=not args.no_candidates,
        summary_path=args.summary,
        same_chunk_only=args.same_chunk_only,
        max_candidates=args.max_candidates,
        audit_batch_size=args.audit_batch_size,
        include_review_entities=not args.accepted_only,
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
