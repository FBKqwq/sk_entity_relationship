"""Prompted LLM Lv2 entity typing labeling functions."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.utils.llm_client import chat_completion_text
from src.weak_supervision.common.lf_output import LFOutput
from src.weak_supervision.common.labels import ABSTAIN
from src.weak_supervision.common.llm_prompt_registry import PromptLFSpec
from src.weak_supervision_lv2.entity_lf_base import EntityLabelingFunction

LLMCallable = Callable[..., dict[str, Any]]

LV2_ENTITY_SYSTEM_PROMPT = """你是医学专家共识知识图谱的 Lv2 弱监督实体定型 Labeling Function。
任务：根据候选实体、证据和 chunk 上下文，判断候选是否属于给定实体类型。
只输出严格 JSON；不能生成新实体；不能改变候选名称；不确定时 present=false 或降低 confidence。"""

LV2_ENTITY_USER_TEMPLATE = """请对候选实体做 Lv2 类型弱标注。

当前 Prompt 视角:
{prompt_focus}

额外判定要求:
{prompt_instruction}

实体类型:
{labels}

候选实体:
{entity_payload}

chunk 上下文:
{chunk_payload}

输出 JSON 对象，key 必须是实体类型，value 格式:
{{
  "present": true/false,
  "confidence": 0到1之间的小数,
  "evidence": "支持该类型的最小原文证据；没有则为空字符串",
  "reason": "不超过40字"
}}
"""

LV2_ENTITY_BATCH_USER_TEMPLATE = """请对以下多个候选实体做 Lv2 类型弱标注。

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
      "entity_id": "输入中的 entity_id",
      "labels": {{
        "实体类型": {{
          "present": true/false,
          "confidence": 0到1之间的小数,
          "evidence": "支持该类型的最小原文证据；没有则为空字符串",
          "reason": "不超过40字"
        }}
      }}
    }}
  ]
}}

要求：每个 entity_id 必须返回一条结果；不能生成新实体；不能改变候选名称；证据必须来自对应 chunk 上下文。

items:
{items_payload}
"""


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


def _normalize_anchor_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("α", "a").replace("Α", "A")
    normalized = normalized.replace("β", "b").replace("Β", "B")
    normalized = normalized.replace("γ", "r").replace("Γ", "R")
    return re.sub(r"\s+", "", normalized).lower()


def _text_contains_anchor(text: str, anchor: str) -> bool:
    if not anchor:
        return False
    return anchor in text or _normalize_anchor_text(anchor) in _normalize_anchor_text(text)


@dataclass
class EntityPromptedLLMLF(EntityLabelingFunction):
    """Prompted-LLM LF that votes on one entity against all Lv2 labels."""

    prompt_spec: PromptLFSpec
    enabled: bool = True
    config_path: str | Path | None = None
    min_confidence: float = 0.5
    max_chunk_chars: int = 2500
    batch_size: int = 20
    max_batch_chars: int = 24000
    retry_missing_items: bool = True
    llm_func: LLMCallable = chat_completion_text
    _cache: dict[tuple[str, tuple[str, ...]], dict[str, LFOutput]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"lv2_entity_prompted_llm_{self.prompt_spec.name}"

    def apply_batch(
        self,
        entity_chunk_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
        labels: list[str],
    ) -> dict[str, dict[str, LFOutput]]:
        """Return prompted-LLM votes for multiple entities in one LLM request."""

        outputs: dict[str, dict[str, LFOutput]] = {}
        pending: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []
        for entity, chunk in entity_chunk_pairs:
            entity_id = self._entity_cache_key(entity)
            cache_key = (entity_id, tuple(labels))
            if cache_key in self._cache:
                outputs[entity_id] = self._cache[cache_key]
                continue
            if not self.enabled:
                row = {label: self._abstain(entity, label, "llm_disabled") for label in labels}
                self._cache[cache_key] = row
                outputs[entity_id] = row
                continue
            text = str(chunk.get("text") or "")
            if not str(entity.get("entity_id") or "") or not str(entity.get("name") or "").strip() or not text:
                row = {label: self._abstain(entity, label, "missing_entity_or_context") for label in labels}
                self._cache[cache_key] = row
                outputs[entity_id] = row
                continue
            pending.append((entity_id, entity, chunk, text))

        for batch in self._iter_batches(pending):
            batch_outputs = self._call_batch(batch, labels)
            for entity_id, entity, chunk, _text in batch:
                cache_key = (entity_id, tuple(labels))
                row = batch_outputs.get(entity_id)
                if row is None and self.retry_missing_items:
                    row = self.apply_all(entity, chunk, labels)
                if row is None:
                    row = {label: self._abstain(entity, label, "batch_missing_item") for label in labels}
                self._cache[cache_key] = row
                outputs[entity_id] = row
        return outputs

    def apply_all(self, entity: dict[str, Any], chunk: dict[str, Any], labels: list[str]) -> dict[str, LFOutput]:
        if not self.enabled:
            return {label: self._abstain(entity, label, "llm_disabled") for label in labels}
        entity_id = self._entity_cache_key(entity)
        cache_key = (entity_id, tuple(labels))
        if cache_key in self._cache:
            return self._cache[cache_key]
        text = str(chunk.get("text") or "")
        if not entity_id or not str(entity.get("name") or "").strip() or not text:
            outputs = {label: self._abstain(entity, label, "missing_entity_or_context") for label in labels}
            self._cache[cache_key] = outputs
            return outputs
        prompt = LV2_ENTITY_USER_TEMPLATE.format(
            prompt_focus=self.prompt_spec.focus,
            prompt_instruction=self.prompt_spec.instruction,
            labels="\n".join(f"- {label}" for label in labels),
            entity_payload=json.dumps(
                {
                    "entity_id": entity.get("entity_id"),
                    "candidate_entity_type": entity.get("entity_type"),
                    "name": entity.get("name"),
                    "content": entity.get("content") or entity.get("name"),
                    "evidence_text": entity.get("evidence_text"),
                    "candidate_properties": entity.get("candidate_properties", {}),
                    "section_title": entity.get("section_title"),
                    "section_path": entity.get("section_path", []),
                },
                ensure_ascii=False,
                indent=2,
            ),
            chunk_payload=json.dumps(
                {
                    "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
                    "section_title": chunk.get("section_title"),
                    "section_path": chunk.get("section_path", []),
                    "text": text[: self.max_chunk_chars],
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        try:
            result = self.llm_func(
                prompt,
                system_prompt=LV2_ENTITY_SYSTEM_PROMPT,
                config_path=self.config_path,
            )
        except Exception as exc:  # noqa: BLE001
            outputs = {label: self._abstain(entity, label, f"llm_error:{exc}") for label in labels}
            self._cache[cache_key] = outputs
            return outputs
        if result.get("status") != "ok":
            outputs = {
                label: self._abstain(entity, label, str(result.get("reason") or "llm_unavailable"))
                for label in labels
            }
            self._cache[cache_key] = outputs
            return outputs
        parsed = _extract_json_object(str(result.get("text") or ""))
        if parsed is None:
            outputs = {label: self._abstain(entity, label, "invalid_json") for label in labels}
            self._cache[cache_key] = outputs
            return outputs
        outputs = {
            label: self._output_from_payload(entity, chunk, label, parsed.get(label), result.get("model"))
            for label in labels
        }
        self._cache[cache_key] = outputs
        return outputs

    def apply(self, entity: dict[str, Any], chunk: dict[str, Any], label: str) -> LFOutput:
        return self.apply_all(entity, chunk, [label])[label]

    def _output_from_payload(
        self,
        entity: dict[str, Any],
        chunk: dict[str, Any],
        label: str,
        payload: Any,
        model: Any,
    ) -> LFOutput:
        if not isinstance(payload, dict):
            return self._abstain(entity, label, "missing_label_payload")
        present = bool(payload.get("present", False))
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        evidence = str(payload.get("evidence") or "").strip()
        text = str(chunk.get("text") or "")
        evidence_located = _text_contains_anchor(text, evidence)
        metadata = {
            "entity_id": entity.get("entity_id"),
            "reason": payload.get("reason"),
            "model": model,
            "prompt_name": self.prompt_spec.name,
            "prompt_focus": self.prompt_spec.focus,
            "raw_present": present,
            "raw_confidence": confidence,
            "evidence_text": evidence,
            "evidence_located": evidence_located,
        }
        if not present or confidence < self.min_confidence:
            metadata["reason"] = metadata["reason"] or "llm_negative_or_low_confidence"
            return LFOutput(self.name, label, vote=ABSTAIN, confidence=confidence, metadata=metadata)
        return LFOutput(self.name, label, vote=1, confidence=confidence, count=1, metadata=metadata)

    def _entity_cache_key(self, entity: dict[str, Any]) -> str:
        return str(entity.get("entity_id") or hash(str(entity)))

    def _iter_batches(
        self,
        items: list[tuple[str, dict[str, Any], dict[str, Any], str]],
    ) -> list[list[tuple[str, dict[str, Any], dict[str, Any], str]]]:
        batches: list[list[tuple[str, dict[str, Any], dict[str, Any], str]]] = []
        current: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []
        current_chars = 0
        for item in items:
            entity_text = json.dumps(self._entity_payload(item[1]), ensure_ascii=False)
            item_chars = len(entity_text) + min(len(item[3]), self.max_chunk_chars)
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

    def _entity_payload(self, entity: dict[str, Any]) -> dict[str, Any]:
        return {
            "entity_id": entity.get("entity_id"),
            "candidate_entity_type": entity.get("entity_type"),
            "name": entity.get("name"),
            "content": entity.get("content") or entity.get("name"),
            "evidence_text": entity.get("evidence_text"),
            "candidate_properties": entity.get("candidate_properties", {}),
            "section_title": entity.get("section_title"),
            "section_path": entity.get("section_path", []),
        }

    def _call_batch(
        self,
        batch: list[tuple[str, dict[str, Any], dict[str, Any], str]],
        labels: list[str],
    ) -> dict[str, dict[str, LFOutput]]:
        items_payload = [
            {
                "entity": self._entity_payload(entity),
                "chunk": {
                    "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
                    "section_title": chunk.get("section_title"),
                    "section_path": chunk.get("section_path", []),
                    "text": text[: self.max_chunk_chars],
                },
            }
            for _entity_id, entity, chunk, text in batch
        ]
        prompt = LV2_ENTITY_BATCH_USER_TEMPLATE.format(
            prompt_focus=self.prompt_spec.focus,
            prompt_instruction=self.prompt_spec.instruction,
            labels="\n".join(f"- {label}" for label in labels),
            items_payload=json.dumps(items_payload, ensure_ascii=False, indent=2),
        )
        try:
            result = self.llm_func(
                prompt,
                system_prompt=LV2_ENTITY_SYSTEM_PROMPT,
                config_path=self.config_path,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                entity_id: {label: self._abstain(entity, label, f"llm_error:{exc}") for label in labels}
                for entity_id, entity, _chunk, _text in batch
            }
        if result.get("status") != "ok":
            reason = str(result.get("reason") or "llm_unavailable")
            return {
                entity_id: {label: self._abstain(entity, label, reason) for label in labels}
                for entity_id, entity, _chunk, _text in batch
            }
        parsed = _extract_json_object(str(result.get("text") or ""))
        if parsed is None:
            return {
                entity_id: {label: self._abstain(entity, label, "invalid_json") for label in labels}
                for entity_id, entity, _chunk, _text in batch
            }
        raw_results = parsed.get("results", [])
        if not isinstance(raw_results, list):
            raw_results = []
        by_entity = {str(item.get("entity_id") or ""): item for item in raw_results if isinstance(item, dict)}
        outputs: dict[str, dict[str, LFOutput]] = {}
        for entity_id, entity, chunk, _text in batch:
            item = by_entity.get(entity_id)
            if not isinstance(item, dict):
                continue
            label_payloads = item.get("labels", {})
            if not isinstance(label_payloads, dict):
                label_payloads = item
            outputs[entity_id] = {
                label: self._output_from_payload(entity, chunk, label, label_payloads.get(label), result.get("model"))
                for label in labels
            }
        return outputs

    def _abstain(self, entity: dict[str, Any], label: str, reason: str) -> LFOutput:
        return LFOutput(
            self.name,
            label,
            vote=ABSTAIN,
            metadata={
                "entity_id": entity.get("entity_id"),
                "reason": reason,
                "prompt_name": self.prompt_spec.name,
                "prompt_focus": self.prompt_spec.focus,
            },
        )
