"""Chunk prompted-LLM LF for Lv1."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.utils.llm_client import chat_completion_text
from src.weak_supervision.common.lf_output import EvidenceSpan, LFOutput
from src.weak_supervision_lv1.chunk_lf_base import ChunkLabelingFunction
from src.weak_supervision_lv1.lfs._chunk_lf_utils import (
    abstain,
    chunk_text,
    find_literal_spans,
    make_output,
    merge_evidence,
)

LLM_SYSTEM_PROMPT = """你是医学专家共识知识图谱的弱监督标注函数。
任务：对同一个 chunk 同时判断所有给定实体类型是否出现，并预测每类实体数量。
只输出 JSON，不要输出解释性正文。证据必须逐字来自 chunk 原文。"""

LLM_USER_TEMPLATE = """请对以下 chunk 做 Lv1 多标签标注。

当前 Prompt 视角:
{prompt_focus}

额外判定要求:
{prompt_instruction}

实体类型:
{labels}

输出 JSON 对象，key 必须是实体类型，value 格式:
{{
  "present": true/false,
  "count": 整数,
  "confidence": 0到1之间的小数,
  "evidence": ["原文证据1", "原文证据2"],
  "reason": "不超过30字"
}}

chunk:
{text}
"""

LLM_BATCH_USER_TEMPLATE = """请对以下多个 chunk 做 Lv1 多标签标注。

当前 Prompt 视角:
{prompt_focus}

额外判定要求:
{prompt_instruction}

实体类型:
{labels}

输出 JSON 对象，格式必须为:
{{
  "results": [
    {{
      "chunk_id": "输入中的 chunk_id",
      "labels": {{
        "实体类型": {{
          "present": true/false,
          "count": 整数,
          "confidence": 0到1之间的小数,
          "evidence": ["原文证据1", "原文证据2"],
          "reason": "不超过30字"
        }}
      }}
    }}
  ]
}}

要求：每个 chunk_id 必须返回一条结果；证据必须逐字来自对应 chunk 原文。

chunks:
{chunks_payload}
"""

LLMCallable = Callable[..., dict[str, Any]]


def _extract_json_object(text: str) -> dict[str, Any] | None:
    clean = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.S)
    if fence_match:
        clean = fence_match.group(1)
    else:
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            clean = clean[start : end + 1]
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


@dataclass
class ChunkPromptedLLMLF(ChunkLabelingFunction):
    """Optional Teacher LLM LF that returns all chunk labels in one call."""

    enabled: bool = False
    config_path: str | Path | None = None
    name: str = "lv1_chunk_prompted_llm"
    prompt_name: str = "general"
    prompt_focus: str = "综合判断 chunk 中是否出现各实体类型。"
    prompt_instruction: str = "优先依据原文直接证据；不确定时 present=false 或降低 confidence。"
    min_confidence: float = 0.5
    max_chunk_chars: int = 5000
    batch_size: int = 10
    max_batch_chars: int = 24000
    retry_missing_items: bool = True
    llm_func: LLMCallable = chat_completion_text
    _cache: dict[tuple[str, tuple[str, ...]], dict[str, LFOutput]] = field(default_factory=dict)

    def apply_batch(
        self,
        chunks: list[dict[str, Any]],
        labels: list[str],
    ) -> dict[str, dict[str, LFOutput]]:
        """Return prompted-LLM votes for multiple chunks in one LLM request."""

        outputs: dict[str, dict[str, LFOutput]] = {}
        pending: list[tuple[str, dict[str, Any], str]] = []
        for chunk in chunks:
            text = chunk_text(chunk)
            chunk_key = self._chunk_cache_key(chunk, text)
            cache_key = (chunk_key, tuple(labels))
            if cache_key in self._cache:
                outputs[chunk_key] = self._cache[cache_key]
                continue
            if not self.enabled:
                row = {label: abstain(self.name, label, "llm_disabled") for label in labels}
                self._cache[cache_key] = row
                outputs[chunk_key] = row
                continue
            if not text:
                row = {label: abstain(self.name, label, "missing_text") for label in labels}
                self._cache[cache_key] = row
                outputs[chunk_key] = row
                continue
            pending.append((chunk_key, chunk, text))

        for batch in self._iter_batches(pending):
            batch_outputs = self._call_batch(batch, labels)
            for chunk_key, chunk, _text in batch:
                cache_key = (chunk_key, tuple(labels))
                row = batch_outputs.get(chunk_key)
                if row is None and self.retry_missing_items:
                    row = self.apply_all(chunk, labels)
                if row is None:
                    row = {label: abstain(self.name, label, "batch_missing_item") for label in labels}
                self._cache[cache_key] = row
                outputs[chunk_key] = row
        return outputs

    def apply_all(self, chunk: dict[str, Any], labels: list[str]) -> dict[str, LFOutput]:
        if not self.enabled:
            return {label: abstain(self.name, label, "llm_disabled") for label in labels}

        text = chunk_text(chunk)
        if not text:
            return {label: abstain(self.name, label, "missing_text") for label in labels}

        cache_key = (self._chunk_cache_key(chunk, text), tuple(labels))
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = LLM_USER_TEMPLATE.format(
            prompt_focus=self.prompt_focus,
            prompt_instruction=self.prompt_instruction,
            labels="\n".join(f"- {label}" for label in labels),
            text=text[: self.max_chunk_chars],
        )
        try:
            result = self.llm_func(
                prompt,
                system_prompt=LLM_SYSTEM_PROMPT,
                config_path=self.config_path,
            )
        except Exception as exc:  # noqa: BLE001 - LF must fail closed to abstain.
            outputs = {label: abstain(self.name, label, f"llm_error:{exc}") for label in labels}
            self._cache[cache_key] = outputs
            return outputs

        if result.get("status") != "ok":
            outputs = {
                label: abstain(self.name, label, str(result.get("reason") or "llm_unavailable"))
                for label in labels
            }
            self._cache[cache_key] = outputs
            return outputs

        parsed = _extract_json_object(str(result.get("text") or ""))
        if parsed is None:
            outputs = {label: abstain(self.name, label, "invalid_json") for label in labels}
            self._cache[cache_key] = outputs
            return outputs

        outputs = {
            label: self._output_from_payload(label, parsed.get(label), text, result.get("model"))
            for label in labels
        }
        self._cache[cache_key] = outputs
        return outputs

    def _output_from_payload(
        self,
        label: str,
        payload: Any,
        text: str,
        model: Any,
    ) -> LFOutput:
        if not isinstance(payload, dict):
            return abstain(self.name, label, "missing_label_payload")

        present = bool(payload.get("present", False))
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        try:
            count = int(payload.get("count", 0))
        except (TypeError, ValueError):
            count = 0

        raw_evidence = payload.get("evidence", [])
        if not isinstance(raw_evidence, list):
            raw_evidence = []
        evidence_groups: list[list[EvidenceSpan]] = []
        missing_evidence: list[str] = []
        for item in raw_evidence:
            evidence_text = str(item).strip()
            if not evidence_text:
                continue
            spans = find_literal_spans(text, evidence_text, max_spans=2)
            if spans:
                evidence_groups.append(spans)
            else:
                missing_evidence.append(evidence_text)

        evidence = merge_evidence(*evidence_groups)
        if not present or confidence < self.min_confidence:
            return make_output(
                self.name,
                label,
                metadata={
                    "reason": payload.get("reason") or "llm_negative_or_low_confidence",
                    "model": model,
                    "prompt_name": self.prompt_name,
                    "prompt_focus": self.prompt_focus,
                    "raw_present": present,
                    "raw_confidence": confidence,
                    "missing_evidence": missing_evidence,
                },
            )
        return make_output(
            self.name,
            label,
            vote=1,
            confidence=confidence,
            evidence=evidence,
            count=max(count, len(evidence), len(missing_evidence)),
            metadata={
                "reason": payload.get("reason"),
                "model": model,
                "prompt_name": self.prompt_name,
                "prompt_focus": self.prompt_focus,
                "missing_evidence": missing_evidence,
                "evidence_span_validated": bool(evidence),
            },
        )

    def _iter_batches(
        self,
        items: list[tuple[str, dict[str, Any], str]],
    ) -> list[list[tuple[str, dict[str, Any], str]]]:
        batches: list[list[tuple[str, dict[str, Any], str]]] = []
        current: list[tuple[str, dict[str, Any], str]] = []
        current_chars = 0
        for item in items:
            item_chars = min(len(item[2]), self.max_chunk_chars)
            if current and (
                len(current) >= max(1, self.batch_size)
                or current_chars + item_chars > max(self.max_chunk_chars, self.max_batch_chars)
            ):
                batches.append(current)
                current = []
                current_chars = 0
            current.append(item)
            current_chars += item_chars
        if current:
            batches.append(current)
        return batches

    def _call_batch(
        self,
        batch: list[tuple[str, dict[str, Any], str]],
        labels: list[str],
    ) -> dict[str, dict[str, LFOutput]]:
        chunks_payload = [
            {
                "chunk_id": chunk_key,
                "source_chunk_id": chunk.get("chunk_id") or chunk.get("id"),
                "section_title": chunk.get("section_title"),
                "section_path": chunk.get("section_path", []),
                "text": text[: self.max_chunk_chars],
            }
            for chunk_key, chunk, text in batch
        ]
        prompt = LLM_BATCH_USER_TEMPLATE.format(
            prompt_focus=self.prompt_focus,
            prompt_instruction=self.prompt_instruction,
            labels="\n".join(f"- {label}" for label in labels),
            chunks_payload=json.dumps(chunks_payload, ensure_ascii=False, indent=2),
        )
        try:
            result = self.llm_func(
                prompt,
                system_prompt=LLM_SYSTEM_PROMPT,
                config_path=self.config_path,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                chunk_key: {label: abstain(self.name, label, f"llm_error:{exc}") for label in labels}
                for chunk_key, _, _ in batch
            }
        if result.get("status") != "ok":
            reason = str(result.get("reason") or "llm_unavailable")
            return {
                chunk_key: {label: abstain(self.name, label, reason) for label in labels}
                for chunk_key, _, _ in batch
            }
        parsed = _extract_json_object(str(result.get("text") or ""))
        if parsed is None:
            return {
                chunk_key: {label: abstain(self.name, label, "invalid_json") for label in labels}
                for chunk_key, _, _ in batch
            }
        raw_results = parsed.get("results", [])
        if not isinstance(raw_results, list):
            raw_results = []
        by_chunk = {str(item.get("chunk_id") or ""): item for item in raw_results if isinstance(item, dict)}
        text_by_key = {chunk_key: text for chunk_key, _, text in batch}
        outputs: dict[str, dict[str, LFOutput]] = {}
        for chunk_key, _, _ in batch:
            item = by_chunk.get(chunk_key)
            if not isinstance(item, dict):
                continue
            label_payloads = item.get("labels", {})
            if not isinstance(label_payloads, dict):
                label_payloads = item
            outputs[chunk_key] = {
                label: self._output_from_payload(label, label_payloads.get(label), text_by_key[chunk_key], result.get("model"))
                for label in labels
            }
        return outputs

    def _chunk_cache_key(self, chunk: dict[str, Any], text: str) -> str:
        for key in ("chunk_id", "id"):
            if chunk.get(key):
                return str(chunk[key])
        return f"anon:{hash(text)}"

    def apply(self, chunk: dict[str, Any], label: str) -> LFOutput:
        """Return prompted-LLM vote for one label."""

        return self.apply_all(chunk, [label])[label]
