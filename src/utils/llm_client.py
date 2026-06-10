"""OpenAI-compatible LLM 客户端封装。"""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.utils.io import read_yaml

ModelKey = Literal["model_name", "TR_model_name", "OCR_model_name"]


@dataclass(frozen=True)
class TeacherLLMConfig:
    """Teacher LLM 运行配置。"""

    enabled: bool
    provider: str
    model_name: str
    tr_model_name: str
    ocr_model_name: str
    api_base: str
    api_key: str | None
    timeout: int
    max_retries: int
    temperature: float
    max_tokens: int
    enable_thinking: bool
    thinking_budget: int


def _default_llm_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "llm.yaml"


def load_teacher_llm_config(config_path: str | Path | None = None) -> TeacherLLMConfig:
    """从 llm.yaml 读取 Teacher LLM 配置。"""
    raw = read_yaml(config_path or _default_llm_config_path()).get("teacher_llm", {})
    if not isinstance(raw, dict):
        raise ValueError("teacher_llm 配置必须是对象。")

    api_key_env = str(raw.get("api_key_env", "DASHSCOPE_API_KEY"))
    api_key = os.getenv(api_key_env) or raw.get("api_key")
    return TeacherLLMConfig(
        enabled=bool(raw.get("enabled", False)),
        provider=str(raw.get("provider", "openai_compatible")),
        model_name=str(raw.get("model_name", "")),
        tr_model_name=str(raw.get("TR_model_name", raw.get("model_name", ""))),
        ocr_model_name=str(raw.get("OCR_model_name", raw.get("model_name", ""))),
        api_base=str(raw.get("api_base", "")),
        api_key=str(api_key) if api_key else None,
        timeout=int(raw.get("timeout", 60)),
        max_retries=int(raw.get("max_retries", 3)),
        temperature=float(raw.get("temperature", 0.0)),
        max_tokens=int(raw.get("max_tokens", 4096)),
        enable_thinking=bool(raw.get("enable_thinking", True)),
        thinking_budget=int(raw.get("thinking_budget", 81920)),
    )


def is_llm_available(config: TeacherLLMConfig) -> bool:
    """判断配置是否允许实际调用模型。"""
    return bool(config.enabled and config.api_key and config.api_base)


def _model_from_key(config: TeacherLLMConfig, model_key: ModelKey) -> str:
    if model_key == "TR_model_name":
        return config.tr_model_name
    if model_key == "OCR_model_name":
        return config.ocr_model_name
    return config.model_name


def create_openai_client(config: TeacherLLMConfig) -> Any:
    """创建 OpenAI-compatible 客户端。"""
    if not is_llm_available(config):
        raise RuntimeError("Teacher LLM 未启用或缺少 api_key/api_base。")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("调用 Teacher LLM 需要安装 openai 包。") from exc
    return OpenAI(api_key=config.api_key, base_url=config.api_base, timeout=config.timeout)


def _call_with_retry(config: TeacherLLMConfig, func: Any) -> Any:
    """按配置执行重试。"""
    last_error: Exception | None = None
    for attempt in range(max(1, config.max_retries)):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - 需要保留原始异常用于最后抛出
            last_error = exc
            if attempt + 1 >= max(1, config.max_retries):
                break
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Teacher LLM 调用失败: {last_error}") from last_error


def chat_completion_text(
    prompt: str,
    *,
    system_prompt: str | None = None,
    model_key: ModelKey = "model_name",
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """调用文本模型并返回结构化结果。"""
    config = load_teacher_llm_config(config_path)
    if not is_llm_available(config):
        return {"status": "disabled", "text": "", "reason": "Teacher LLM 未启用或缺少 API Key。"}

    try:
        client = create_openai_client(config)
    except RuntimeError as exc:
        return {"status": "disabled", "text": "", "reason": str(exc)}
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    model = _model_from_key(config, model_key)
    extra_body = {"enable_thinking": config.enable_thinking}

    def _request() -> Any:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body=extra_body,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    response = _call_with_retry(config, _request)
    text = response.choices[0].message.content or ""
    return {"status": "ok", "text": text.strip(), "model": model}


def image_path_to_data_url(image_path: str | Path) -> str:
    """将本地图片转换为 data URL。"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片文件不存在: {path}")
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def vision_completion_text(
    image_url: str,
    prompt: str,
    *,
    config_path: str | Path | None = None,
    stream: bool = True,
) -> dict[str, Any]:
    """调用图表/图片识别模型并返回回答文本。"""
    config = load_teacher_llm_config(config_path)
    if not is_llm_available(config):
        return {"status": "disabled", "text": "", "reason": "Teacher LLM 未启用或缺少 API Key。"}

    try:
        client = create_openai_client(config)
    except RuntimeError as exc:
        return {"status": "disabled", "text": "", "reason": str(exc)}
    model = _model_from_key(config, "OCR_model_name")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    extra_body = {"enable_thinking": config.enable_thinking, "thinking_budget": config.thinking_budget}

    def _request() -> Any:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            extra_body=extra_body,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    response = _call_with_retry(config, _request)
    if not stream:
        text = response.choices[0].message.content or ""
        return {"status": "ok", "text": text.strip(), "reasoning_content": "", "model": model}

    reasoning_content = ""
    answer_content = ""
    for chunk in response:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            reasoning_content += reasoning
            continue
        content = getattr(delta, "content", None)
        if content:
            answer_content += content
    return {
        "status": "ok",
        "text": answer_content.strip(),
        "reasoning_content": reasoning_content.strip(),
        "model": model,
    }
