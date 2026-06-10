"""Official Snorkel label-model adapters with local trace preservation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from src.weak_supervision.common.labels import ABSTAIN, NEGATIVE
from src.weak_supervision.common.lf_output import LFOutput

SNORKEL_ABSTAIN = -1
SNORKEL_NEGATIVE = 0
SNORKEL_POSITIVE = 1


def _to_snorkel_vote(output: LFOutput | None) -> int:
    if output is None or output.vote == ABSTAIN:
        return SNORKEL_ABSTAIN
    if output.vote == NEGATIVE:
        return SNORKEL_NEGATIVE
    if output.vote > 0:
        return SNORKEL_POSITIVE
    return SNORKEL_ABSTAIN


def _analysis_to_dict(analysis: Any) -> dict[str, Any]:
    try:
        frame = analysis.lf_summary()
    except Exception:  # noqa: BLE001 - diagnostics must not block the pipeline.
        return {"available": False}
    rows: dict[str, Any] = {}
    for lf_name, row in frame.to_dict(orient="index").items():
        rows[str(lf_name)] = {
            str(key): _json_safe_analysis_value(value)
            for key, value in row.items()
        }
    return {"available": True, "lf_summary": rows}


def _json_safe_analysis_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        if pd.isna(value):
            return None
    except Exception:  # noqa: BLE001
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_official_binary_label_models(
    *,
    rows: list[dict[str, Any]],
    labels: list[str],
    outputs_by_row_label_lf: dict[tuple[str, str, str], LFOutput],
    row_id_field: str,
    lf_names: list[str],
    thresholds: dict[str, float] | None = None,
    n_epochs: int = 200,
    seed: int = 13,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """Run official Snorkel one-vs-rest LabelModel for each label.

    The official matrix only carries votes. Evidence, count and metadata stay
    in the caller's LFOutput rows and are joined back by row id / label / LF.
    """

    try:
        from snorkel.labeling import LFAnalysis, LabelingFunction, PandasLFApplier
        from snorkel.labeling.model import LabelModel, MajorityLabelVoter
    except Exception as exc:  # noqa: BLE001 - caller can fall back locally.
        return {}, {
            "available": False,
            "reason": f"official_snorkel_import_failed:{exc}",
        }

    frame = pd.DataFrame(
        [
            {
                "__row_id": str(row.get(row_id_field) or row.get("id") or index),
                **row,
            }
            for index, row in enumerate(rows)
        ]
    )
    if frame.empty:
        return {}, {"available": True, "labels": {}, "reason": "empty_rows"}

    thresholds = thresholds or {}
    results: dict[tuple[str, str], dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {
        "available": True,
        "mode": "official_snorkel_label_model",
        "labels": {},
    }

    for label in labels:
        official_lfs = []
        for lf_name in lf_names:
            def vote_fn(row: pd.Series, *, _lf_name: str = lf_name, _label: str = label) -> int:
                row_id = str(row.get("__row_id") or "")
                return _to_snorkel_vote(outputs_by_row_label_lf.get((row_id, _label, _lf_name)))

            official_lfs.append(LabelingFunction(name=lf_name, f=vote_fn))

        applier = PandasLFApplier(lfs=official_lfs)
        label_matrix = applier.apply(df=frame)
        analysis = LFAnalysis(L=label_matrix, lfs=official_lfs)
        majority = MajorityLabelVoter(cardinality=2)
        majority_probs = majority.predict_proba(L=label_matrix)
        label_model_probs = majority_probs
        fit_status = "majority_only"
        fit_reason = ""
        if label_matrix.shape[1] >= 2 and np.any(label_matrix != SNORKEL_ABSTAIN):
            try:
                model = LabelModel(cardinality=2, verbose=False)
                model.fit(L_train=label_matrix, n_epochs=n_epochs, seed=seed, log_freq=0)
                label_model_probs = model.predict_proba(L=label_matrix)
                fit_status = "label_model"
            except Exception as exc:  # noqa: BLE001 - use majority baseline if model fit is ill-posed.
                fit_status = "majority_fallback"
                fit_reason = str(exc)

        threshold = float(thresholds.get(label, 0.35))
        weak_threshold = max(0.05, threshold * 0.75)
        for row_index, row in frame.iterrows():
            row_id = str(row["__row_id"])
            positive_probability = float(label_model_probs[row_index][SNORKEL_POSITIVE])
            majority_probability = float(majority_probs[row_index][SNORKEL_POSITIVE])
            present = positive_probability >= threshold
            status = "accepted" if present else ("weak" if positive_probability >= weak_threshold else "rejected")
            results[(row_id, label)] = {
                "present": present,
                "status": status,
                "confidence": round(positive_probability, 6),
                "vote_score": round(positive_probability, 6),
                "threshold": threshold,
                "official_label_model_probability": round(positive_probability, 6),
                "official_majority_probability": round(majority_probability, 6),
                "official_fit_status": fit_status,
            }

        diagnostics["labels"][label] = {
            "matrix_shape": list(label_matrix.shape),
            "coverage_rows": int(np.sum(np.any(label_matrix != SNORKEL_ABSTAIN, axis=1))),
            "analysis": _analysis_to_dict(analysis),
            "fit_status": fit_status,
            "fit_reason": fit_reason,
        }

    return results, diagnostics


def run_official_multiclass_label_model(
    *,
    rows: list[dict[str, Any]],
    labels: list[str],
    outputs_by_row_label_lf: dict[tuple[str, str, str], LFOutput],
    row_id_field: str,
    lf_names: list[str],
    n_epochs: int = 200,
    seed: int = 13,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Run official Snorkel multiclass LabelModel for entity typing."""

    try:
        from snorkel.labeling import LFAnalysis, LabelingFunction, PandasLFApplier
        from snorkel.labeling.model import LabelModel, MajorityLabelVoter
    except Exception as exc:  # noqa: BLE001
        return {}, {
            "available": False,
            "reason": f"official_snorkel_import_failed:{exc}",
        }

    frame = pd.DataFrame(
        [
            {
                "__row_id": str(row.get(row_id_field) or row.get("id") or index),
                **row,
            }
            for index, row in enumerate(rows)
        ]
    )
    if frame.empty:
        return {}, {"available": True, "mode": "official_snorkel_multiclass", "reason": "empty_rows"}

    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    official_lfs = []
    for lf_name in lf_names:
        def vote_fn(row: pd.Series, *, _lf_name: str = lf_name) -> int:
            row_id = str(row.get("__row_id") or "")
            positive: list[tuple[float, str]] = []
            for label in labels:
                output = outputs_by_row_label_lf.get((row_id, label, _lf_name))
                if output is not None and output.vote > 0:
                    positive.append((max(0.0, output.confidence), label))
            if not positive:
                return SNORKEL_ABSTAIN
            positive.sort(reverse=True)
            return label_to_id[positive[0][1]]

        official_lfs.append(LabelingFunction(name=lf_name, f=vote_fn))

    applier = PandasLFApplier(lfs=official_lfs)
    label_matrix = applier.apply(df=frame)
    analysis = LFAnalysis(L=label_matrix, lfs=official_lfs)
    majority = MajorityLabelVoter(cardinality=len(labels))
    majority_probs = majority.predict_proba(L=label_matrix)
    label_model_probs = majority_probs
    fit_status = "majority_only"
    fit_reason = ""
    if label_matrix.shape[1] >= 2 and np.any(label_matrix != SNORKEL_ABSTAIN):
        try:
            model = LabelModel(cardinality=len(labels), verbose=False)
            model.fit(L_train=label_matrix, n_epochs=n_epochs, seed=seed, log_freq=0)
            label_model_probs = model.predict_proba(L=label_matrix)
            fit_status = "label_model"
        except Exception as exc:  # noqa: BLE001
            fit_status = "majority_fallback"
            fit_reason = str(exc)

    results: dict[str, dict[str, Any]] = {}
    for row_index, row in frame.iterrows():
        row_id = str(row["__row_id"])
        probs = {
            id_to_label[index]: float(label_model_probs[row_index][index])
            for index in range(len(labels))
        }
        majority_row = {
            id_to_label[index]: float(majority_probs[row_index][index])
            for index in range(len(labels))
        }
        ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        final_label = ranked[0][0]
        probability = ranked[0][1]
        second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
        results[row_id] = {
            "final_label": final_label,
            "probability": round(probability, 6),
            "top2_gap": round(probability - second_probability, 6),
            "probabilities": {label: round(value, 6) for label, value in probs.items()},
            "majority_probabilities": {label: round(value, 6) for label, value in majority_row.items()},
            "official_fit_status": fit_status,
        }

    diagnostics = {
        "available": True,
        "mode": "official_snorkel_multiclass_label_model",
        "matrix_shape": list(label_matrix.shape),
        "coverage_rows": int(np.sum(np.any(label_matrix != SNORKEL_ABSTAIN, axis=1))),
        "labels": labels,
        "analysis": _analysis_to_dict(analysis),
        "fit_status": fit_status,
        "fit_reason": fit_reason,
    }
    return results, diagnostics


def group_lf_outputs_by_row_label_lf(
    outputs: list[LFOutput],
    *,
    row_id_metadata_key: str = "chunk_id",
) -> dict[tuple[str, str, str], LFOutput]:
    """Index LFOutput rows for official Snorkel wrapper lookup."""

    grouped: dict[tuple[str, str, str], LFOutput] = {}
    for output in outputs:
        row_id = str(output.metadata.get(row_id_metadata_key) or "")
        if not row_id:
            continue
        grouped[(row_id, output.label, output.lf_name)] = output
    return grouped


def lf_names_by_label(outputs: list[LFOutput]) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for output in outputs:
        grouped[output.label].add(output.lf_name)
    return {label: sorted(names) for label, names in grouped.items()}
