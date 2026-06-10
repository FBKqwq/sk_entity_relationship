"""Run Snorkel Lv1 chunk-label weak supervision.

This is the main entry point for the first weak-supervision layer. It applies
chunk-level Labeling Functions to every ``(chunk, label)`` pair, writes raw LF
traces, and writes fused ``chunk_label_result`` records for the next stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.weak_supervision.common.lf_output import EvidenceSpan, LFOutput
from src.weak_supervision.common.llm_prompt_registry import (
    LV1_CHUNK_PROMPT_SPECS,
    prompt_specs_from_config,
)
from src.weak_supervision.common.official_snorkel_runner import (
    group_lf_outputs_by_row_label_lf,
    run_official_binary_label_models,
)
from src.weak_supervision.common.snorkel_preflight import official_snorkel_preflight
from src.weak_supervision_lv1.chunk_lf_applier import apply_chunk_lfs
from src.weak_supervision_lv1.chunk_signal_builder import build_chunk_label_results_for_chunks
from src.weak_supervision_lv1.lfs import (
    ChunkDictionaryLF,
    ChunkMedicalPatternLF,
    ChunkPromptedLLMLF,
    ChunkRegexIndicatorLF,
    ChunkSectionPriorLF,
)
from src.utils.io import _json_default, read_json, read_yaml, write_json, write_jsonl


def _progress(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def _load_chunks(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)], {}
    if not isinstance(payload, dict):
        raise ValueError(f"Chunk JSON must be an object or list: {path}")
    raw_chunks = payload.get("chunks", [])
    if not isinstance(raw_chunks, list):
        raise ValueError(f"Chunk JSON field `chunks` must be a list: {path}")
    return [item for item in raw_chunks if isinstance(item, dict)], payload


def _chunk_id(chunk: dict[str, Any]) -> str:
    return str(chunk.get("chunk_id") or chunk.get("id") or "")


def _source_pdf(chunk_payload: dict[str, Any], chunks_path: str | Path) -> str:
    value = chunk_payload.get("pdf_path") or chunk_payload.get("source_title")
    if value:
        return str(value)
    return Path(chunks_path).name.removesuffix(".chunk.json") + ".pdf"


def _active_labels(config: dict[str, Any]) -> list[str]:
    labels = config.get("active_labels") or config.get("labels")
    if isinstance(labels, list):
        return [str(label) for label in labels]
    return ["sub_diseases", "symptoms", "tests", "treatments", "plans"]


def _llm_enabled_from_config(config: dict[str, Any], *, force: bool = False, disable: bool = False) -> bool:
    if disable:
        return False
    if force:
        return True
    lv1 = config.get("lv1_prompted_llm", {})
    if isinstance(lv1, dict) and "enabled" in lv1:
        return bool(lv1["enabled"])
    lfs = config.get("lfs", {})
    if isinstance(lfs, dict):
        prompted = lfs.get("lv1_chunk_prompted_llm", {})
        if isinstance(prompted, dict) and "enabled" in prompted:
            return bool(prompted["enabled"])
    return False


def _prompted_llm_specs_from_config(config: dict[str, Any]) -> list[Any]:
    lv1 = config.get("lv1_prompted_llm", {})
    if not isinstance(lv1, dict):
        return list(LV1_CHUNK_PROMPT_SPECS)
    return prompt_specs_from_config(lv1.get("prompts"), LV1_CHUNK_PROMPT_SPECS)


def _llm_batching_config(config: dict[str, Any]) -> dict[str, Any]:
    batching = config.get("llm_batching", {})
    return batching if isinstance(batching, dict) else {}


def build_default_lfs(
    *,
    weak_config_path: str | Path | None,
    llm_config_path: str | Path | None,
    llm_enabled: bool,
    project_config: dict[str, Any] | None = None,
) -> list[Any]:
    """Create the configured Lv1 LF set in stable order."""

    config = project_config or {}
    batching = _llm_batching_config(config)
    prompted_llm_lfs = [
        ChunkPromptedLLMLF(
            enabled=llm_enabled,
            config_path=llm_config_path,
            name=f"lv1_chunk_prompted_llm_{spec.name}",
            prompt_name=spec.name,
            prompt_focus=spec.focus,
            prompt_instruction=spec.instruction,
            batch_size=int(batching.get("lv1_chunk_batch_size", batching.get("chunk_batch_size", 10))),
            max_batch_chars=int(batching.get("max_chars_per_batch", 24000)),
            retry_missing_items=bool(batching.get("retry_missing_items", True)),
        )
        for spec in _prompted_llm_specs_from_config(config)
    ]
    return [
        ChunkMedicalPatternLF(),
        ChunkDictionaryLF(config_path=weak_config_path),
        ChunkRegexIndicatorLF(),
        ChunkSectionPriorLF(),
        *prompted_llm_lfs,
    ]


def _evidence_to_record(span: EvidenceSpan) -> dict[str, Any]:
    return {
        "start": span.start,
        "end": span.end,
        "text": span.text,
        "source": span.source,
    }


def lf_output_to_record(output: LFOutput, chunk: dict[str, Any], *, source_pdf: str) -> dict[str, Any]:
    """Return a JSONL-safe raw LF trace record."""

    return {
        "chunk_id": _chunk_id(chunk),
        "document_id": chunk.get("document_id"),
        "source_pdf": source_pdf,
        "section_title": chunk.get("section_title"),
        "section_path": chunk.get("section_path", []),
        "lf_name": output.lf_name,
        "label": output.label,
        "vote": output.vote,
        "confidence": output.confidence,
        "count": output.count,
        "evidence": [_evidence_to_record(span) for span in output.evidence],
        "metadata": output.metadata,
    }


def _lv1_thresholds(config: dict[str, Any]) -> dict[str, float]:
    raw = config.get("lv1_vote_model", {})
    if not isinstance(raw, dict):
        return {}
    thresholds = raw.get("threshold", {})
    if not isinstance(thresholds, dict):
        return {}
    return {str(label): float(value) for label, value in thresholds.items()}


def _positive_count_floor(outputs: list[LFOutput], label: str) -> int:
    return max(
        [output.count for output in outputs if output.label == label and output.vote > 0],
        default=0,
    )


def _apply_official_results(
    records: list[dict[str, Any]],
    outputs_by_chunk: dict[str, list[LFOutput]],
    official_results: dict[tuple[str, str], dict[str, Any]],
) -> None:
    for record in records:
        chunk_id = str(record.get("chunk_id") or "")
        label = str(record.get("label") or "")
        result = official_results.get((chunk_id, label))
        if not result:
            continue
        local_metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        record.update(
            {
                "present": result["present"],
                "status": result["status"],
                "confidence": result["confidence"],
                "vote_score": result["vote_score"],
                "threshold": result["threshold"],
            }
        )
        record["metadata"] = {
            **local_metadata,
            "fusion_backend": "official_snorkel_label_model",
            "official_label_model_probability": result["official_label_model_probability"],
            "official_majority_probability": result["official_majority_probability"],
            "official_fit_status": result["official_fit_status"],
            "local_vote_score": local_metadata.get("local_vote_score", record.get("vote_score")),
        }
        if not result["present"]:
            record["predicted_count"] = 0
            record["count_confidence"] = 0.0
        elif int(record.get("predicted_count") or 0) <= 0:
            record["predicted_count"] = max(1, _positive_count_floor(outputs_by_chunk.get(chunk_id, []), label))


def _outputs_by_chunk(outputs: list[LFOutput]) -> dict[str, list[LFOutput]]:
    grouped: dict[str, list[LFOutput]] = {}
    for output in outputs:
        chunk_id = str(output.metadata.get("chunk_id") or "")
        if not chunk_id:
            continue
        grouped.setdefault(chunk_id, []).append(output)
    return grouped


def _attach_chunk_ids(outputs: list[LFOutput], chunks: list[dict[str, Any]], labels: list[str], lf_count: int) -> list[LFOutput]:
    """Attach chunk_id to LFOutput metadata for grouping without changing LF APIs."""

    expected_per_chunk = len(labels) * lf_count
    if expected_per_chunk <= 0:
        return outputs
    enriched: list[LFOutput] = []
    for index, output in enumerate(outputs):
        chunk_index = index // expected_per_chunk
        chunk = chunks[chunk_index] if chunk_index < len(chunks) else {}
        metadata = dict(output.metadata)
        metadata["chunk_id"] = _chunk_id(chunk)
        enriched.append(
            LFOutput(
                lf_name=output.lf_name,
                label=output.label,
                vote=output.vote,
                confidence=output.confidence,
                count=output.count,
                evidence=output.evidence,
                metadata=metadata,
            )
        )
    return enriched


def run_lv1_for_file(
    *,
    chunks_path: str | Path,
    output_path: str | Path,
    lf_output_path: str | Path,
    weak_config_path: str | Path | None = "configs/weak_supervision.yaml",
    llm_config_path: str | Path | None = "configs/llm.yaml",
    enable_prompted_llm: bool = False,
    disable_prompted_llm: bool = False,
) -> dict[str, Any]:
    """Run Lv1 chunk labeling for one chunk JSON file."""

    config = read_yaml(weak_config_path) if weak_config_path else {}
    labels = _active_labels(config)
    llm_enabled = _llm_enabled_from_config(
        config,
        force=enable_prompted_llm,
        disable=disable_prompted_llm,
    )
    chunks, chunk_payload = _load_chunks(chunks_path)
    source_pdf = _source_pdf(chunk_payload, chunks_path)
    lfs = build_default_lfs(
        weak_config_path=weak_config_path,
        llm_config_path=llm_config_path,
        llm_enabled=llm_enabled,
        project_config=config,
    )

    _progress(f"applying {len(lfs)} Lv1 LFs to {len(chunks)} chunks from {Path(chunks_path).name}")
    raw_outputs = apply_chunk_lfs(chunks, labels, lfs)
    raw_outputs = _attach_chunk_ids(raw_outputs, chunks, labels, len(lfs))
    outputs_by_chunk = _outputs_by_chunk(raw_outputs)
    records = build_chunk_label_results_for_chunks(
        chunks,
        labels,
        outputs_by_chunk,
        project_config=config,
    )
    official_results, official_diagnostics = run_official_binary_label_models(
        rows=chunks,
        labels=labels,
        outputs_by_row_label_lf=group_lf_outputs_by_row_label_lf(raw_outputs),
        row_id_field="chunk_id",
        lf_names=[lf.name for lf in lfs],
        thresholds=_lv1_thresholds(config),
    )
    if official_results:
        _apply_official_results(records, outputs_by_chunk, official_results)
    for record in records:
        record["source_pdf"] = source_pdf
        record["lv1_prompted_llm_enabled"] = llm_enabled

    chunk_by_id = {_chunk_id(chunk): chunk for chunk in chunks}
    raw_records = [
        lf_output_to_record(output, chunk_by_id.get(str(output.metadata.get("chunk_id")), {}), source_pdf=source_pdf)
        for output in raw_outputs
    ]
    write_jsonl(raw_records, lf_output_path)
    write_jsonl(records, output_path)

    positive_chunks = {
        str(record.get("chunk_id"))
        for record in records
        if bool(record.get("present")) and str(record.get("status")) in {"accepted", "weak"}
    }
    summary = {
        "source_pdf": source_pdf,
        "chunks_path": str(chunks_path),
        "lv1_path": str(output_path),
        "lf_path": str(lf_output_path),
        "chunks": len(chunks),
        "labels": labels,
        "lf_names": [lf.name for lf in lfs],
        "lf_outputs": len(raw_records),
        "lv1_records": len(records),
        "lv1_positive_chunks": len(positive_chunks),
        "prompted_llm_enabled": llm_enabled,
        "official_snorkel": official_snorkel_preflight(),
        "official_snorkel_fusion": official_diagnostics,
    }
    return summary


def _default_output_path(chunks_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{chunks_path.name.removesuffix('.chunk.json')}.chunk_label_result.jsonl"


def _default_lf_path(chunks_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{chunks_path.name.removesuffix('.chunk.json')}.lv1_lf_outputs.jsonl"


def run_lv1_for_directory(
    *,
    chunks_dir: str | Path,
    output_dir: str | Path,
    weak_config_path: str | Path | None = "configs/weak_supervision.yaml",
    llm_config_path: str | Path | None = "configs/llm.yaml",
    enable_prompted_llm: bool = False,
    disable_prompted_llm: bool = False,
    summary_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run Lv1 chunk labeling for every ``*.chunk.json`` file in a directory."""

    chunks_root = Path(chunks_dir)
    output_root = Path(output_dir)
    summaries: list[dict[str, Any]] = []
    for chunks_path in sorted(chunks_root.glob("*.chunk.json")):
        summaries.append(
            run_lv1_for_file(
                chunks_path=chunks_path,
                output_path=_default_output_path(chunks_path, output_root),
                lf_output_path=_default_lf_path(chunks_path, output_root),
                weak_config_path=weak_config_path,
                llm_config_path=llm_config_path,
                enable_prompted_llm=enable_prompted_llm,
                disable_prompted_llm=disable_prompted_llm,
            )
        )
    if summary_path is not None:
        write_json(summaries, summary_path)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lv1 chunk-label weak supervision.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--chunks", help="Single chunk JSON file.")
    input_group.add_argument("--chunks-dir", help="Directory containing *.chunk.json files.")
    parser.add_argument("--output", help="Single output *.chunk_label_result.jsonl file.")
    parser.add_argument("--lf-output", help="Single output *.lv1_lf_outputs.jsonl file.")
    parser.add_argument("--output-dir", default="data/weak_signals", help="Directory for Lv1 outputs.")
    parser.add_argument("--summary", help="Optional pipeline summary JSON path.")
    parser.add_argument("--config", default="configs/weak_supervision.yaml", help="Weak supervision config YAML.")
    parser.add_argument("--llm-config", default="configs/llm.yaml", help="Teacher LLM config YAML.")
    parser.add_argument("--enable-prompted-llm", action="store_true", help="Enable Lv1 Teacher LLM LF.")
    parser.add_argument("--disable-prompted-llm", action="store_true", help="Disable Lv1 Teacher LLM LF.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.enable_prompted_llm and args.disable_prompted_llm:
        raise SystemExit("--enable-prompted-llm and --disable-prompted-llm are mutually exclusive")

    if args.chunks:
        if not args.output or not args.lf_output:
            raise SystemExit("--chunks mode requires --output and --lf-output")
        summary = run_lv1_for_file(
            chunks_path=args.chunks,
            output_path=args.output,
            lf_output_path=args.lf_output,
            weak_config_path=args.config,
            llm_config_path=args.llm_config,
            enable_prompted_llm=args.enable_prompted_llm,
            disable_prompted_llm=args.disable_prompted_llm,
        )
        if args.summary:
            write_json(summary, args.summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
        return

    summaries = run_lv1_for_directory(
        chunks_dir=args.chunks_dir,
        output_dir=args.output_dir,
        weak_config_path=args.config,
        llm_config_path=args.llm_config,
        enable_prompted_llm=args.enable_prompted_llm,
        disable_prompted_llm=args.disable_prompted_llm,
        summary_path=args.summary,
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
