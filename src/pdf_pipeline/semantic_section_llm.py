"""语义章节识别阶段的 LLM 复筛：仅返回噪声/翻译区间，由本地拼接正文。"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from src.utils.llm_client import chat_completion_text, is_llm_available, load_teacher_llm_config

PAGE_BREAK_PATTERN = re.compile(r"\n<<<PAGE_BREAK:(\d+)>>>\n")

SYSTEM_PROMPT = (
    "你是医学 PDF 文本预处理标注助手。用户将提供单一大字符串 `marked`："
    "已初步去噪的页文本按顺序拼接，页与页之间用精确子串 \\n<<<PAGE_BREAK:N>>>\\n 分隔（N 为页码，从 2 起）。\n"
    "你的任务：只输出一个 JSON 对象，**禁止**输出全文、禁止输出 cleaned_text。\n"
    "对 `marked` 使用与 Python 切片一致的 **0 起算、左闭右开** 区间 [start, end)。\n\n"
    "字段：\n"
    "1）noise_spans：需整段删除的无效版面区间（作者与单位罗列、学会署名、通信作者及邮箱等）。"
    "每项：start（int）, end（int）, reason（简体中文，可选）。\n"
    "2）translation_spans：需译为简体中文的英文片段区间。每项：start, end, translated_text（str，"
    "仅替换该区间内原文，不得夹带区间外文字）。\n\n"
    "硬约束：\n"
    "- 0 <= start < end <= len(marked)。len(marked) 在用户消息首行给出。\n"
    "- noise_spans 与 translation_spans 中所有区间 **两两不重叠**（无交集）。\n"
    "- 任一区间不得与任一 <<<PAGE_BREAK:N>>> 分隔子串（含两侧换行）在字符上相交；"
    "即不得删除或改写页分隔符。\n"
    "- 除上述删除与替换外，不得对正文做概括或重写；translation_spans 仅覆盖英文为主的连续片段。\n"
    "- 若无某类修改，对应数组为 []。\n"
    "- 只输出合法 JSON，不要 Markdown 代码围栏，不要额外解释。"
)

USER_TEMPLATE = (
    "以下为 marked 全文（含页分隔符）。首行单独给出字符总数，请勿在 JSON 中重复正文。\n"
    "len(marked) = {length}\n"
    "禁止覆盖的页分隔符区间 forbidden_spans = {forbidden_spans}\n"
    "任何 noise_spans / translation_spans 都不得与 forbidden_spans 相交。\n\n"
    "重要：输出必须很短，只能是 JSON 对象；不要输出 marked 原文，不要输出 cleaned_text，不要解释。\n"
    "若无需要翻译的英文正文片段，translation_spans 必须为 []。\n\n"
    "{body}\n\n"
    "请输出 JSON，仅包含字段 noise_spans、translation_spans（定义见系统说明）。"
)

ALLOWED_OUTPUT_KEYS = {"noise_spans", "translation_spans"}


def _pre_log(msg: str) -> None:
    """语义复筛子步骤控制台输出（与 scripts 中 [进度] 前缀一致）。"""
    print(f"[进度] 语义复筛: {msg}", flush=True)


def _join_pages_with_markers(page_text: dict[int, str]) -> str:
    """将多页文本用固定分隔符拼接，便于与模型约定同一字符坐标系。"""
    numbers = sorted(page_text)
    if not numbers:
        return ""
    parts: list[str] = [page_text[numbers[0]]]
    for page_number in numbers[1:]:
        parts.append(f"\n<<<PAGE_BREAK:{page_number}>>>\n")
        parts.append(page_text[page_number])
    return "".join(parts)


def _split_pages_from_markers(cleaned: str, expected_pages: list[int]) -> dict[int, str] | None:
    """从拼接串拆回页码 -> 文本；分隔符缺失或页码不一致时返回 None。"""
    if not expected_pages:
        return {}
    parts = PAGE_BREAK_PATTERN.split(cleaned)
    if len(parts) == 1 and len(expected_pages) > 1:
        return None
    out: dict[int, str] = {}
    out[expected_pages[0]] = parts[0].strip()
    index = 1
    while index < len(parts):
        page_num_s = parts[index]
        body = parts[index + 1] if index + 1 < len(parts) else ""
        try:
            page_num = int(page_num_s)
        except ValueError:
            return None
        out[page_num] = body.strip()
        index += 2
    if set(out.keys()) != set(expected_pages):
        return None
    return out


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _page_break_spans(marked: str, expected_pages: list[int]) -> list[tuple[int, int]]:
    """返回每个页分隔符子串在 marked 中的 [start, end) 区间（禁止与用户区间相交）。"""
    spans: list[tuple[int, int]] = []
    for page_number in expected_pages[1:]:
        needle = f"\n<<<PAGE_BREAK:{page_number}>>>\n"
        offset = 0
        while True:
            pos = marked.find(needle, offset)
            if pos < 0:
                break
            spans.append((pos, pos + len(needle)))
            offset = pos + 1
    return spans


def _format_forbidden_spans(spans: list[tuple[int, int]]) -> str:
    """将禁止区间格式化为紧凑 JSON，直接放入 prompt 供模型避让。"""
    return json.dumps([{"start": s, "end": e} for s, e in spans], ensure_ascii=False)


def _looks_like_contract_violation(raw: str, marked_len: int, max_output_chars: int) -> str:
    """识别模型是否明显违反“仅输出区间 JSON”的契约。"""
    if len(raw) > max_output_chars:
        return (
            f"模型输出过长（{len(raw)} 字符，限制 {max_output_chars}），"
            "疑似回显正文、cleaned_text 或额外解释。"
        )
    if "cleaned_text" in raw:
        return "模型返回了 cleaned_text 字段，违反仅返回区间 JSON 的约束。"
    if marked_len > 0 and len(raw) > max(8000, int(marked_len * 1.2)):
        return f"模型输出长度异常（{len(raw)} 字符，marked 长度 {marked_len}），疑似未按区间格式返回。"
    return ""


def _validate_output_keys(parsed: dict[str, Any]) -> str:
    extra = sorted(str(k) for k in parsed.keys() if k not in ALLOWED_OUTPUT_KEYS)
    if extra:
        return f"模型 JSON 包含未允许字段: {', '.join(extra)}。"
    return ""


def _intervals_disjoint(intervals: list[tuple[int, int]]) -> bool:
    intervals = sorted(intervals)
    for i in range(len(intervals) - 1):
        if intervals[i][1] > intervals[i + 1][0]:
            return False
    return True


def _interval_hits_protected(s: int, e: int, protected: list[tuple[int, int]]) -> bool:
    for ps, pe in protected:
        if not (e <= ps or s >= pe):
            return True
    return False


def _has_safe_delete_boundaries(marked: str, s: int, e: int) -> bool:
    """噪声删除区间必须落在空白/行边界上，避免从正文词中间挖掉内容。"""
    left_ok = s == 0 or marked[s - 1].isspace()
    right_ok = e == len(marked) or marked[e].isspace()
    return left_ok and right_ok


def _build_ops_from_parsed(
    parsed: dict[str, Any],
    marked: str,
    protected: list[tuple[int, int]],
) -> tuple[list[dict[str, Any]], str]:
    """
    校验并生成操作列表。每项 op 为 delete 或 replace，按 start 降序可由本地安全套用。
    成功返回 (ops, "")；失败返回 ([], 错误原因)。
    """
    raw_noise = parsed.get("noise_spans")
    raw_trans = parsed.get("translation_spans")
    if raw_noise is not None and not isinstance(raw_noise, list):
        return [], "noise_spans 必须是数组。"
    if raw_trans is not None and not isinstance(raw_trans, list):
        return [], "translation_spans 必须是数组。"
    noise_list = raw_noise if isinstance(raw_noise, list) else []
    trans_list = raw_trans if isinstance(raw_trans, list) else []
    marked_len = len(marked)

    ops: list[dict[str, Any]] = []
    intervals: list[tuple[int, int]] = []

    for item in noise_list:
        if not isinstance(item, dict):
            continue
        try:
            s, e = int(item["start"]), int(item["end"])
        except (KeyError, TypeError, ValueError):
            return [], "noise_spans 某项缺少整数 start/end。"
        if not (0 <= s < e <= marked_len):
            return [], f"noise_spans 区间非法: [{s}, {e})，marked 长度 {marked_len}。"
        if _interval_hits_protected(s, e, protected):
            return [], f"noise_spans 与页分隔符冲突: [{s}, {e})。"
        if not _has_safe_delete_boundaries(marked, s, e):
            return [], f"noise_spans 区间疑似截断正文字符: [{s}, {e})。"
        ops.append({"op": "delete", "start": s, "end": e, "reason": str(item.get("reason", ""))})
        intervals.append((s, e))

    for item in trans_list:
        if not isinstance(item, dict):
            continue
        try:
            s, e = int(item["start"]), int(item["end"])
        except (KeyError, TypeError, ValueError):
            return [], "translation_spans 某项缺少整数 start/end。"
        if not (0 <= s < e <= marked_len):
            return [], f"translation_spans 区间非法: [{s}, {e})，marked 长度 {marked_len}。"
        if _interval_hits_protected(s, e, protected):
            return [], f"translation_spans 与页分隔符冲突: [{s}, {e})。"
        text = item.get("translated_text", item.get("text", ""))
        if text is None or str(text).strip() == "":
            return [], f"translation_spans 区间 [{s},{e}) 缺少非空 translated_text。"
        ops.append({"op": "replace", "start": s, "end": e, "text": str(text)})
        intervals.append((s, e))

    if not _intervals_disjoint(intervals):
        return [], "noise_spans 与 translation_spans 之间存在重叠区间。"

    ops.sort(key=lambda o: int(o["start"]), reverse=True)
    return ops, ""


def _apply_span_ops(marked: str, ops: list[dict[str, Any]]) -> str:
    work = marked
    for op in ops:
        s, e = int(op["start"]), int(op["end"])
        if op["op"] == "delete":
            work = work[:s] + work[e:]
        else:
            work = work[:s] + str(op["text"]) + work[e:]
    return work


def _noise_fragments_report(marked: str, noise_list: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in noise_list:
        if not isinstance(item, dict):
            continue
        try:
            s, e = int(item["start"]), int(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        s = max(0, s)
        e = min(len(marked), e)
        if s >= e:
            continue
        excerpt = marked[s:e][:200]
        out.append(
            {
                "label": "噪声片段",
                "start": s,
                "end": e,
                "excerpt": excerpt,
                "reason": str(item.get("reason", "")),
            }
        )
    return out


def apply_semantic_section_llm_prescreen(
    pages: list[dict[str, Any]],
    *,
    pdf_config: dict[str, Any],
    llm_config_path: str | Path,
) -> dict[str, Any]:
    """
    在去噪页文本上调用 LLM，仅获取噪声删除区间与英文替换译文，再在本地拼接后写回 pages[*]['text']。

    返回报告字典（可写入 chunk JSON 根字段或单独落盘）。
    """
    cfg = pdf_config.get("semantic_section_llm") or {}
    report: dict[str, Any] = {
        "enabled": bool(cfg.get("enabled", False)),
        "status": "skipped",
        "mode": "span_ops_v1",
        "noise_fragments": [],
        "reason": "",
    }
    if not report["enabled"]:
        report["reason"] = "semantic_section_llm.enabled 为 false。"
        return report

    teacher = load_teacher_llm_config(llm_config_path)
    if not is_llm_available(teacher):
        report["status"] = "disabled"
        report["reason"] = "Teacher LLM 未启用或缺少 API Key，跳过复筛。"
        _pre_log("Teacher LLM 不可用，跳过复筛。")
        return report

    page_text = {int(p["page_number"]): str(p.get("text", "")) for p in pages}
    expected_pages = sorted(page_text)
    _pre_log(f"开始拼接页文本（共 {len(expected_pages)} 页）…")
    marked = _join_pages_with_markers(page_text)
    raw_len = len(marked)
    max_chars = int(cfg.get("max_input_chars", 120_000))
    truncated = False
    if raw_len > max_chars:
        marked = marked[:max_chars]
        truncated = True
        _pre_log(f"输入长度 {raw_len} 超过 max_input_chars={max_chars}，已截断为 {len(marked)} 字符再送模型。")
    else:
        _pre_log(f"拼接完成，送入模型字符数约 {raw_len}（未截断）。")

    protected = _page_break_spans(marked, expected_pages)
    prompt = USER_TEMPLATE.format(
        length=len(marked),
        forbidden_spans=_format_forbidden_spans(protected),
        body=marked,
    )
    prompt_chars = len(prompt)
    _pre_log(
        f"即将调用 chat 模型（仅返回区间 JSON，prompt 约 {prompt_chars} 字符）；"
        "等待期间每 30 秒打印心跳。"
    )

    stop_heartbeat = threading.Event()

    def _heartbeat() -> None:
        waited = 0
        while not stop_heartbeat.wait(30.0):
            waited += 30
            _pre_log(f"仍在等待 LLM 响应…（已累计约 {waited}s）")

    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()
    t0 = time.monotonic()
    try:
        result = chat_completion_text(
            prompt,
            system_prompt=SYSTEM_PROMPT,
            model_key="model_name",
            config_path=llm_config_path,
        )
    finally:
        stop_heartbeat.set()
        hb.join(timeout=2.0)

    elapsed = time.monotonic() - t0
    _pre_log(f"LLM HTTP 调用结束（耗时约 {elapsed:.1f}s），status={result.get('status')!s}。")

    if result.get("status") != "ok" or not str(result.get("text", "")).strip():
        report["status"] = "failed"
        report["reason"] = str(result.get("reason", "模型无有效返回。"))
        _pre_log(f"模型返回异常，reason={report['reason'][:200]}")
        return report

    raw_out = str(result["text"])
    _pre_log(f"开始解析 JSON（模型输出约 {len(raw_out)} 字符）…")
    max_output_chars = int(cfg.get("max_output_chars", 12000))
    contract_err = _looks_like_contract_violation(raw_out, len(marked), max_output_chars)
    if contract_err:
        report["status"] = "failed"
        report["reason"] = contract_err
        _pre_log(f"模型输出契约校验失败: {contract_err}")
        return report

    parsed = _extract_json_object(raw_out)
    if not parsed:
        report["status"] = "failed"
        report["reason"] = "无法解析模型 JSON。"
        _pre_log("JSON 解析失败。")
        return report

    key_err = _validate_output_keys(parsed)
    if key_err:
        report["status"] = "failed"
        report["reason"] = key_err
        _pre_log(f"模型输出字段校验失败: {key_err}")
        return report

    noise_list = parsed.get("noise_spans") if isinstance(parsed.get("noise_spans"), list) else []
    report["noise_fragments"] = _noise_fragments_report(marked, noise_list)

    ops, err = _build_ops_from_parsed(parsed, marked, protected)
    if err:
        report["status"] = "failed"
        report["reason"] = err
        _pre_log(f"区间校验失败: {err}")
        return report

    _pre_log(
        f"区间合法：noise_spans={sum(1 for o in ops if o['op']=='delete')} 段，"
        f"translation_spans={sum(1 for o in ops if o['op']=='replace')} 段；正在本地套用…"
    )
    edited = _apply_span_ops(marked, ops)
    split_map = _split_pages_from_markers(edited, expected_pages)
    if split_map is None:
        report["status"] = "failed"
        report["reason"] = "套用区间后页分隔符与页码集合不一致，已跳过写回。"
        _pre_log("套用后分隔符校验失败，跳过写回 pages。")
        return report

    _pre_log("正在将编辑结果写回各页 pages[*].text …")
    for page in pages:
        pn = int(page["page_number"])
        page["text"] = split_map.get(pn, str(page.get("text", "")))
        page.setdefault("meta", {})["semantic_section_llm"] = True

    report["status"] = "ok"
    report["truncated_input"] = truncated
    report["model"] = result.get("model")
    report["noise_fragment_count"] = len(report["noise_fragments"])
    report["translation_span_count"] = sum(1 for o in ops if o["op"] == "replace")
    report["noise_span_count"] = sum(1 for o in ops if o["op"] == "delete")
    _pre_log(
        f"复筛成功：写回 {len(expected_pages)} 页；删除 {report['noise_span_count']} 段、"
        f"替换译文 {report['translation_span_count']} 段；noise_fragments 记录 {report['noise_fragment_count']} 条。"
    )
    return report
