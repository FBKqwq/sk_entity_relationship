"""Build Lv2 entity label decisions from entity_base candidates."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any

from src.weak_supervision.common.graph_schema import ACTIVE_ENTITY_LABELS
from src.weak_supervision.common.labels import ABSTAIN, NEGATIVE
from src.weak_supervision.common.lf_output import LFOutput
from src.weak_supervision.common.llm_prompt_registry import LV2_ENTITY_PROMPT_SPECS
from src.weak_supervision.common.official_snorkel_runner import run_official_binary_label_models
from src.weak_supervision_lv2.entity_label_matrix import analyze_lf_outputs
from src.weak_supervision_lv2.lfs.lf_entity_context_window import EntityContextWindowLF
from src.weak_supervision_lv2.lfs.lf_entity_dictionary import EntitySuggestedTypeLF
from src.weak_supervision_lv2.lfs.lf_entity_prompted_llm import EntityPromptedLLMLF
from src.weak_supervision_lv2.lfs.lf_entity_section_prior import EntitySectionPriorLF
from src.weak_supervision_lv2.lfs.lf_entity_surface_pattern import EntitySurfacePatternLF
from src.utils.io import read_yaml

DEFAULT_ACCEPT_THRESHOLD = 0.65
DEFAULT_REVIEW_THRESHOLD = 0.40
DEFAULT_MIN_TOP2_GAP = 0.15
LOW_RECALL_ACCEPT_RATE = 0.10
LLM_PROMPT_LF_PREFIX = "lv2_entity_prompted_llm_"
LLM_CANDIDATE_PROTECTION_MIN_PROMPTS = 3
LLM_CANDIDATE_PROTECTION_MIN_CONFIDENCE = 0.80
PLAN_EXECUTION_PATTERNS = (
    r"\b\d+(\.\d+)?\s*(mg|g|ml|μg|ug)\b",
    r"(每日|每周|每月|每晚|每次|1次|2次|3次|bid|tid|qd|q[0-9]+h)",
    r"(口服|注射|静脉|肌内|皮下|玻璃体内|关节腔内|鞘内)",
    r"(冲击治疗|序贯|逐渐减量|减少.*剂量|短期抗凝|随访|疗程)",
    r"(甲泼尼龙|泼尼松|秋水仙碱|环磷酰胺|硫唑嘌呤|英夫利西单抗|阿达木单抗|曲安奈德)",
)


def _llm_batching_config(config_path: str | None = None) -> dict[str, Any]:
    if config_path is None:
        return {}
    try:
        config = read_yaml(config_path)
    except Exception:  # noqa: BLE001 - batching config must not break Lv2.
        return {}
    batching = config.get("llm_batching", {})
    return batching if isinstance(batching, dict) else {}


def default_entity_lfs(config_path: str | None = None) -> list[Any]:
    """Return the default Lv2 LF set."""

    batching = _llm_batching_config(config_path)
    return [
        EntitySuggestedTypeLF(),
        EntitySurfacePatternLF(),
        EntityContextWindowLF(),
        EntitySectionPriorLF(),
        *[
            EntityPromptedLLMLF(
                prompt_spec=spec,
                config_path=config_path,
                batch_size=int(batching.get("lv2_entity_batch_size", batching.get("entity_batch_size", 20))),
                max_batch_chars=int(batching.get("max_chars_per_batch", 24000)),
                retry_missing_items=bool(batching.get("retry_missing_items", True)),
            )
            for spec in LV2_ENTITY_PROMPT_SPECS
        ],
    ]


def _apply_lv2_lf(lf: Any, entity: dict[str, Any], chunk: dict[str, Any], labels: list[str]) -> dict[str, LFOutput]:
    if hasattr(lf, "apply_all"):
        return lf.apply_all(entity, chunk, labels)
    return {label: lf.apply(entity, chunk, label) for label in labels}


def _entity_key(entity: dict[str, Any]) -> str:
    return str(entity.get("entity_id") or hash(str(entity)))


def _chunk_id(chunk: dict[str, Any]) -> str:
    return str(chunk.get("chunk_id") or chunk.get("id") or "")


def chunk_by_id(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_chunk_id(chunk): chunk for chunk in chunks}


def evidence_is_located(entity: dict[str, Any], chunk: dict[str, Any] | None) -> bool:
    """Return whether candidate evidence can be anchored to the source chunk."""

    if chunk is None:
        return False
    text = str(chunk.get("text") or "")
    evidence = str(entity.get("evidence_text") or "").strip()
    name = str(entity.get("name") or "").strip()
    if evidence and evidence in text:
        return True
    if name and name in text:
        return True
    normalized_text = _normalize_anchor_text(text)
    return bool(
        evidence
        and _normalize_anchor_text(evidence) in normalized_text
        or name
        and _normalize_anchor_text(name) in normalized_text
    )


def _normalize_anchor_text(value: str) -> str:
    """Normalize PDF layout noise for evidence back-link checks."""

    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("α", "a").replace("Α", "A")
    normalized = normalized.replace("β", "b").replace("Β", "B")
    normalized = normalized.replace("γ", "r").replace("Γ", "R")
    return re.sub(r"\s+", "", normalized).lower()


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    exp_scores = {label: math.exp(value - max_score) for label, value in scores.items()}
    total = sum(exp_scores.values()) or 1.0
    return {label: value / total for label, value in exp_scores.items()}


def _score_label(outputs: list[LFOutput]) -> float:
    score = 0.0
    for output in outputs:
        if output.vote == ABSTAIN:
            continue
        if output.vote == NEGATIVE:
            score -= max(0.0, output.confidence) * 0.6
            continue
        score += max(0.0, output.confidence)
    return score


def _llm_candidate_type_protection(
    result: dict[str, Any],
    *,
    labels: set[str],
) -> dict[str, Any] | None:
    """Return an accepted candidate-type override when all LLM Prompt LFs agree."""

    candidate_type = str(result.get("candidate_entity_type") or "")
    if candidate_type not in labels or not result.get("evidence_located"):
        return None
    positive_candidate_lfs: list[dict[str, Any]] = []
    positive_other_lfs: list[dict[str, Any]] = []
    for trace in result.get("lf_trace", []):
        lf_name = str(trace.get("lf_name") or "")
        if not lf_name.startswith(LLM_PROMPT_LF_PREFIX):
            continue
        vote = int(trace.get("vote") or ABSTAIN)
        confidence = float(trace.get("confidence") or 0.0)
        if vote <= 0 or confidence < LLM_CANDIDATE_PROTECTION_MIN_CONFIDENCE:
            continue
        if trace.get("label") == candidate_type:
            positive_candidate_lfs.append(trace)
        else:
            positive_other_lfs.append(trace)
    prompt_names = {
        str((trace.get("metadata") or {}).get("prompt_name") or trace.get("lf_name"))
        for trace in positive_candidate_lfs
    }
    if len(prompt_names) < LLM_CANDIDATE_PROTECTION_MIN_PROMPTS:
        return None
    if positive_other_lfs:
        return None
    avg_confidence = sum(float(trace.get("confidence") or 0.0) for trace in positive_candidate_lfs) / len(positive_candidate_lfs)
    return {
        "final_entity_type": candidate_type,
        "lv2_probability": round(max(avg_confidence, float(result.get("lv2_probability") or 0.0)), 6),
        "top2_gap": round(max(0.2, float(result.get("top2_gap") or 0.0)), 6),
        "status": "accepted",
        "conflict_reasons": [],
        "fusion_backend": "official_snorkel_label_model_with_llm_candidate_protection",
        "source_lfs": [trace["lf_name"] for trace in positive_candidate_lfs],
        "llm_candidate_type_protected": True,
    }


def _plan_execution_override(result: dict[str, Any]) -> dict[str, Any] | None:
    """Promote clearly executable medication/procedure/follow-up items to Plan."""

    current_type = str(result.get("final_entity_type") or "")
    candidate_type = str(result.get("candidate_entity_type") or "")
    if current_type not in {"treatments", "plans"} and candidate_type not in {"treatments", "plans"}:
        return None
    text = f"{result.get('name') or ''}\n{result.get('evidence_text') or ''}"
    matched = [pattern for pattern in PLAN_EXECUTION_PATTERNS if re.search(pattern, text, flags=re.I)]
    if not matched:
        return None
    confidence = max(float(result.get("lv2_probability") or 0.0), 0.72)
    return {
        "final_entity_type": "plans",
        "lv2_probability": round(confidence, 6),
        "top2_gap": round(max(float(result.get("top2_gap") or 0.0), 0.2), 6),
        "status": "accepted" if result.get("evidence_located") else result.get("status"),
        "conflict_reasons": [] if result.get("evidence_located") else result.get("conflict_reasons", []),
        "fusion_backend": f"{result.get('fusion_backend') or 'local_fusion'}_with_plan_execution_override",
        "source_lfs": [*list(result.get("source_lfs") or []), "lv2_plan_execution_boundary"],
        "plan_execution_override": True,
        "plan_execution_patterns": matched[:5],
    }


def _status_for_candidate(
    *,
    entity: dict[str, Any],
    final_label: str,
    probability: float,
    top2_gap: float,
    evidence_located: bool,
    accept_threshold: float,
    review_threshold: float,
    min_top2_gap: float,
    labels: set[str],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not str(entity.get("name") or "").strip():
        return "rejected", ["empty_name"]
    if not str(entity.get("evidence_text") or "").strip():
        return "rejected", ["empty_evidence"]
    if not evidence_located:
        return "rejected", ["evidence_not_located"]
    if final_label not in labels:
        return "rejected", ["schema_out_of_scope"]
    if probability >= accept_threshold and top2_gap >= min_top2_gap:
        return "accepted", []
    if probability < review_threshold:
        return "rejected", ["low_probability"]
    if top2_gap < min_top2_gap:
        reasons.append("type_conflict_top2_gap")
    if probability < accept_threshold:
        reasons.append("review_probability_band")
    return "review", reasons


def build_entity_label_results(
    entities: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    labels: list[str] | None = None,
    accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    min_top2_gap: float = DEFAULT_MIN_TOP2_GAP,
    lfs: list[Any] | None = None,
    config_path: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run Lv2 entity type fusion and return label results/conflicts/report."""

    active_labels = list(labels or ACTIVE_ENTITY_LABELS)
    active_label_set = set(active_labels)
    local_lfs = lfs or default_entity_lfs(config_path=config_path)
    chunks_by_id = chunk_by_id(chunks)
    results: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    all_outputs: list[LFOutput] = []
    candidate_counts = Counter()
    official_index: dict[tuple[str, str, str], LFOutput] = {}
    entity_chunk_pairs = [
        (entity, chunks_by_id.get(str(entity.get("chunk_id") or "")) or {})
        for entity in entities
    ]
    batched_lf_outputs: dict[str, dict[str, dict[str, LFOutput]]] = {}
    for lf in local_lfs:
        lf_name = getattr(lf, "name", "")
        if lf_name and hasattr(lf, "apply_batch"):
            batched_lf_outputs[lf_name] = lf.apply_batch(entity_chunk_pairs, active_labels)

    for entity in entities:
        candidate_type = str(entity.get("entity_type") or "")
        if candidate_type in active_label_set:
            candidate_counts[candidate_type] += 1
        chunk = chunks_by_id.get(str(entity.get("chunk_id") or ""))
        outputs_by_label: dict[str, list[LFOutput]] = {label: [] for label in active_labels}
        entity_outputs: list[LFOutput] = []
        entity_key = _entity_key(entity)
        for lf in local_lfs:
            lf_name = getattr(lf, "name", "")
            if lf_name in batched_lf_outputs:
                outputs_for_lf = batched_lf_outputs[lf_name].get(entity_key)
                if outputs_for_lf is None:
                    outputs_for_lf = _apply_lv2_lf(lf, entity, chunk or {}, active_labels)
            else:
                outputs_for_lf = _apply_lv2_lf(lf, entity, chunk or {}, active_labels)
            for label in active_labels:
                output = outputs_for_lf[label]
                outputs_by_label[label].append(output)
                entity_outputs.append(output)
                all_outputs.append(output)
                entity_id = str(entity.get("entity_id") or "")
                official_index[(entity_id, label, output.lf_name)] = output
        for label in active_labels:
            label_outputs = outputs_by_label[label]
        scores = {label: _score_label(outputs) for label, outputs in outputs_by_label.items()}
        probabilities = _softmax(scores)
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        final_label = ranked[0][0] if ranked else candidate_type
        probability = ranked[0][1] if ranked else 0.0
        second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
        top2_gap = probability - second_probability
        located = evidence_is_located(entity, chunk)
        status, reasons = _status_for_candidate(
            entity=entity,
            final_label=final_label,
            probability=probability,
            top2_gap=top2_gap,
            evidence_located=located,
            accept_threshold=accept_threshold,
            review_threshold=review_threshold,
            min_top2_gap=min_top2_gap,
            labels=active_label_set,
        )
        source_lfs = [
            output.lf_name
            for output in outputs_by_label.get(final_label, [])
            if output.vote != ABSTAIN
        ]
        result = {
            "entity_id": entity.get("entity_id"),
            "document_id": entity.get("document_id"),
            "chunk_id": entity.get("chunk_id"),
            "section_title": entity.get("section_title"),
            "section_path": entity.get("section_path", []),
            "candidate_entity_type": candidate_type,
            "final_entity_type": final_label,
            "name": entity.get("name"),
            "content": entity.get("content") or entity.get("name"),
            "evidence_text": entity.get("evidence_text"),
            "candidate_confidence": float(entity.get("confidence") or 0.0),
            "lv2_probability": round(probability, 6),
            "top2_gap": round(top2_gap, 6),
            "probabilities": {label: round(value, 6) for label, value in probabilities.items()},
            "source_lfs": source_lfs,
            "evidence_located": located,
            "status": status,
            "conflict_reasons": reasons,
            "lf_trace": [
                {
                    "lf_name": output.lf_name,
                    "label": output.label,
                    "vote": output.vote,
                    "confidence": output.confidence,
                    "metadata": output.metadata,
                }
                for output in entity_outputs
            ],
        }
        results.append(result)
        if status != "accepted":
            conflicts.append(
                {
                    "entity_id": entity.get("entity_id"),
                    "name": entity.get("name"),
                    "candidate_entity_type": candidate_type,
                    "final_entity_type": final_label,
                    "status": status,
                    "reasons": reasons,
                    "lv2_probability": round(probability, 6),
                    "top2_gap": round(top2_gap, 6),
                    "chunk_id": entity.get("chunk_id"),
                }
            )

    official_binary_results, official_diagnostics = run_official_binary_label_models(
        rows=entities,
        labels=active_labels,
        outputs_by_row_label_lf=official_index,
        row_id_field="entity_id",
        lf_names=sorted({output.lf_name for output in all_outputs}),
        thresholds={label: accept_threshold for label in active_labels},
    )
    if official_binary_results:
        conflicts = []
        for result in results:
            entity_id = str(result.get("entity_id") or "")
            label_scores = {
                label: official_binary_results.get((entity_id, label), {}).get("official_label_model_probability", 0.0)
                for label in active_labels
            }
            if not label_scores:
                continue
            total = sum(float(value) for value in label_scores.values()) or 1.0
            probabilities = {label: float(value) / total for label, value in label_scores.items()}
            ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            final_label = ranked[0][0]
            probability = ranked[0][1]
            second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
            top2_gap = probability - second_probability
            entity = next((item for item in entities if str(item.get("entity_id") or "") == entity_id), {})
            status, reasons = _status_for_candidate(
                entity=entity,
                final_label=final_label,
                probability=probability,
                top2_gap=top2_gap,
                evidence_located=bool(result.get("evidence_located")),
                accept_threshold=accept_threshold,
                review_threshold=review_threshold,
                min_top2_gap=min_top2_gap,
                labels=active_label_set,
            )
            result.update(
                {
                    "final_entity_type": final_label,
                    "lv2_probability": round(probability, 6),
                    "top2_gap": round(top2_gap, 6),
                    "probabilities": {label: round(value, 6) for label, value in probabilities.items()},
                    "one_vs_rest_probabilities": {
                        label: round(float(value), 6) for label, value in label_scores.items()
                    },
                    "status": status,
                    "conflict_reasons": reasons,
                    "fusion_backend": "official_snorkel_label_model",
                    "official_fit_status": official_binary_results.get((entity_id, final_label), {}).get("official_fit_status"),
                    "source_lfs": [
                        trace["lf_name"]
                        for trace in result.get("lf_trace", [])
                        if trace.get("label") == final_label and trace.get("vote") != ABSTAIN
                    ],
                }
            )
            protected = _llm_candidate_type_protection(result, labels=active_label_set)
            if protected is not None:
                final_label = str(protected["final_entity_type"])
                probability = float(protected["lv2_probability"])
                top2_gap = float(protected["top2_gap"])
                status = str(protected["status"])
                reasons = []
                result.update(protected)
            plan_override = _plan_execution_override(result)
            if plan_override is not None:
                final_label = str(plan_override["final_entity_type"])
                probability = float(plan_override["lv2_probability"])
                top2_gap = float(plan_override["top2_gap"])
                status = str(plan_override["status"])
                reasons = list(plan_override.get("conflict_reasons") or [])
                result.update(plan_override)
            if status != "accepted":
                conflicts.append(
                    {
                        "entity_id": result.get("entity_id"),
                        "name": result.get("name"),
                        "candidate_entity_type": result.get("candidate_entity_type"),
                        "final_entity_type": final_label,
                        "status": status,
                        "reasons": reasons,
                        "lv2_probability": round(probability, 6),
                        "top2_gap": round(top2_gap, 6),
                        "chunk_id": result.get("chunk_id"),
                    }
                )

    per_label = defaultdict(lambda: Counter({"candidates": 0, "accepted": 0, "review": 0, "rejected": 0}))
    for label, count in candidate_counts.items():
        per_label[label]["candidates"] = count
    for result in results:
        per_label[str(result.get("final_entity_type") or "")][str(result.get("status") or "rejected")] += 1

    label_rows: dict[str, dict[str, Any]] = {}
    for label, counts in sorted(per_label.items()):
        candidates = counts["candidates"]
        accepted = counts["accepted"]
        label_rows[label] = {
            "candidates": candidates,
            "accepted": accepted,
            "review": counts["review"],
            "rejected": counts["rejected"],
            "accepted_rate": round(accepted / candidates, 4) if candidates else 0.0,
            "low_recall_warning": bool(candidates and accepted / candidates < LOW_RECALL_ACCEPT_RATE),
        }
    report = {
        "entities": len(entities),
        "results": len(results),
        "accepted": sum(1 for row in results if row["status"] == "accepted"),
        "review": sum(1 for row in results if row["status"] == "review"),
        "rejected": sum(1 for row in results if row["status"] == "rejected"),
        "thresholds": {
            "accepted": accept_threshold,
            "review": review_threshold,
            "min_top2_gap": min_top2_gap,
        },
        "labels": label_rows,
        "lf_analysis": analyze_lf_outputs(all_outputs),
        "official_snorkel_fusion": official_diagnostics,
    }
    return results, conflicts, report
