"""Probe tool name rules by systematically registering tools with diverse names.

This script sends minimal tool definitions to a target LLM provider and records
whether the API accepts the tool name, how it is normalized, and if the model
can subsequently call the tool. The implementation follows the "exp0" design
for surveying name restrictions without relying on provider documentation.

The experiment keeps API usage under 50 calls and performs requests
concurrently as required by repository guidelines. Model credentials are
loaded from ``config.toml`` via the shared ``load_models`` helper.
"""

from __future__ import annotations

import asyncio
import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import json

from openai import AsyncOpenAI, NotFoundError
try:
    from volcenginesdkarkruntime import Ark  # 火山引擎 Ark SDK（可选）
except Exception:  # pragma: no cover
    Ark = None  # type: ignore

from experiments.config import load_models
from mcp_elo.experiment import ExperimentLogger, ResultWriter, run_for_models

# Load model configurations from ``config.toml``
MODELS = load_models()

# Limit total external calls to avoid excessive usage.
MAX_CALLS = 500

# Base directories for logs and results
LOG_DIR = Path(__file__).resolve().parent / "logs" / "exp0"
RESULT_DIR = Path(__file__).resolve().parent / "results" / "exp0"


@dataclass
class ProbeResult:
    """Structured record for a single probe."""
    name: str
    category: str
    api_status: str
    visible_name: str | None
    can_call: bool
    normalized: str | None
    notes: str | None = None


# ---------------------------------------------------------------------------
# Test case generation
# ---------------------------------------------------------------------------

def _length_cases() -> List[str]:
    base = "a"
    lengths = [1, 2, 16, 32, 63, 64, 65, 127, 128, 255, 256]
    return [base * n for n in lengths]


def _charset_cases() -> List[str]:
    return [
        "tool", "tool_1", "tool-1", "Tool123",
        "my.tool", "my/tool", "my:tool", "my@tool", "my+tool",
        "my,tool", "my?tool", "my#tool", "my%tool",
        "my tool", " tool", "tool ",
    ]


def _unicode_cases() -> List[str]:
    return [
        "工具", "café", "cafe\u0301", "ｔｏｏｌ", "🛠️tool",
        "δοκιμή", "tool\u200bname", "tool\u200dname", "tool\u2060name", "Straße",
    ]


def _directional_cases() -> List[str]:
    return ["abc\u200Fdef", "\u202Eabc"]


def _boundary_cases() -> List[str]:
    return ["_start", "-start", "end_", "end-", "0start", ".", ".hidden"]


def _reserved_cases() -> List[str]:
    return [
        "system","assistant","user","function","tool","default","null",
        "none","model","server","read","write","search","call","exec","shell",
    ]


def _path_cases() -> List[str]:
    return ["server.tool", "server/tool", "ns:tool", "../tool", "./tool"]


def _control_cases() -> List[str]:
    return ["tl\nname", "tl\rname", "tl\tname"]


# Collisions represented as tuples of (name_a, name_b)
COLLISION_CASES = [
    ("dup", "dup"),
    ("Tool", "tool"),
    ("café", "cafe\u0301"),
    ("name", "na\u200bme"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_base_url(base_url: Optional[str]) -> Optional[str]:
    """Ensure the base_url ends with exactly one '/v1' if provided."""
    if not base_url:
        return None
    u = base_url.rstrip("/")
    if not re.search(r"/v\d+$", u):
        u += "/v1"
    return u


def _is_volc(base_url: Optional[str]) -> bool:
    """Heuristic to detect Volcengine Ark compatible endpoint."""
    if not base_url:
        return False
    s = base_url.lower()
    return ("volc" in s) or ("ark" in s) or ("doubao" in s)


def _make_tool_nested(name: str) -> dict:
    # OpenAI/通用 Responses “嵌套 function”形态
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "test",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _make_tool_toplevel(name: str) -> dict:
    # 顶层 name 形态（部分 Responses 校验器支持）
    return {
        "type": "function",
        "name": name,
        "description": "test",
        "parameters": {"type": "object", "properties": {}},
        "function": {
            "name": name,
            "description": "test",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _make_tool_minimal(name: str) -> dict:
    # 最小形态：不包含 "function" 字段（用于某些 Ark 端点）
    return {
        "type": "function",
        "name": name,
        "description": "test",
        "parameters": {"type": "object", "properties": {}},
    }


def _looks_like_missing_tools_name_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "tools[0].name" in msg
        or "missing required parameter" in msg.lower()
        or "missing_required_parameter" in msg.lower()
    )


def _looks_like_unknown_function_field_error(exc: Exception) -> bool:
    return 'unknown field "function"' in str(exc).lower()


def _looks_like_missing_tools_function_error(exc: Exception) -> bool:
    m = str(exc).lower()
    return ("tools.function" in m) or ("missingparameter" in m and "function" in m)


def _to_dict(obj: Any) -> Dict[str, Any]:
    """Best-effort conversion of SDK object to plain dict for parsing."""
    if obj is None:
        return {}
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            pass
    dd = getattr(obj, "dict", None)
    if callable(dd):
        try:
            return dd()
        except Exception:
            pass
    tj = getattr(obj, "to_json", None)
    if callable(tj):
        try:
            return json.loads(tj())
        except Exception:
            pass
    try:
        return json.loads(str(obj))
    except Exception:
        pass
    try:
        return obj.__dict__
    except Exception:
        return {}


def _extract_tool_calls_from_responses(resp: Any) -> List[Dict[str, Any]]:
    """
    Parse tool calls from Responses API result, covering both:
      • Newer format: resp.output[*] can directly be a function/tool call
        (e.g., {'type': 'function_call', 'name': '...', ...})
      • Legacy format: resp.output[*].content[*] contains entries with
        type in {'tool_call', 'function_call'}.
    """
    data = _to_dict(resp)
    results: List[Dict[str, Any]] = []

    output = data.get("output") or []
    for item in output:
        if not isinstance(item, dict):
            try:
                item = _to_dict(item)
            except Exception:
                item = {}

        # 1) New Responses shape: the output item itself is a function/tool call
        t = (item.get("type") or "").lower()
        if t in ("function_call", "tool_call"):
            results.append(item)

        # 2) Legacy/alternative shape: scan nested content
        contents = item.get("content") or []
        for c in contents:
            if not isinstance(c, dict):
                try:
                    c = _to_dict(c)
                except Exception:
                    c = {}
            ct = (c.get("type") or "").lower()
            if ct in ("tool_call", "function_call"):
                results.append(c)

    return results


def _visible_name_from_tool_call(call: Dict[str, Any]) -> Optional[str]:
    """
    Extract the displayed tool/function name from a parsed call object.
    Common providers put it at:
      • call['name']                              (OpenAI Responses function_call)
      • call['function']['name']                  (nested function object)
      • call['tool_name']                         (some gateways)
    """
    if not isinstance(call, dict):
        try:
            call = _to_dict(call)
        except Exception:
            return None

    # 1) Most common in new Responses: top-level name
    name = call.get("name")
    if isinstance(name, str) and name.strip():
        return name

    # 2) Nested function object
    fn = call.get("function")
    if isinstance(fn, dict):
        fn_name = fn.get("name")
        if isinstance(fn_name, str) and fn_name.strip():
            return fn_name

    # 3) Some providers/gateways use tool_name
    tool_name = call.get("tool_name")
    if isinstance(tool_name, str) and tool_name.strip():
        return tool_name

    return None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class OpenAIAdapter:
    """Minimal adapter around OpenAI Responses API with robust fallbacks."""

    def __init__(self, api_key: str, model_name: str, base_url: str | None = None):
        base_url = _normalize_base_url(base_url)
        self.model_name = model_name
        self.is_volc = _is_volc(base_url)
        if self.is_volc and Ark is not None:
            self.client = Ark(api_key=api_key, base_url=base_url)
        else:
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def _responses_create(self, prompt: str, tool: dict) -> Any:
        return await self.client.responses.create(
            model=self.model_name,
            input=prompt,
            tools=[tool],
        )

    async def _chat_completions_create(self, prompt: str, tool: dict) -> Any:
        return await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            tools=[tool],
            tool_choice="auto",
        )

    async def probe_single(self, name: str) -> Dict[str, Any]:
        """
        Send a single-tool request and capture behavior.

        Ark/Doubao:
          1) 先尝试“嵌套 function”形态（满足 `tools.function` 的端点）
          2) 若报 `unknown field "function"`，降级为“最小形态”（无 function）
          3) 两者都失败再兜底 Chat Completions，同样套用上述顺序

        非 Ark:
          嵌套 -> 顶层重试 -> Chat Completions 兜底
        """
        prompt = f"If you can see a tool named '{name}', call it once and return nothing else."

        # ---- Ark / Doubao 专用路径：优先“嵌套 function”，必要时降级 ----
        if self.is_volc:
            tool_nested = _make_tool_nested(name)
            try:
                resp = await self._responses_create(prompt, tool_nested)
                calls = _extract_tool_calls_from_responses(resp)
                visible = _visible_name_from_tool_call(calls[0]) if calls else None
                return {
                    "api_status": "ok",
                    "visible_name": visible,
                    "can_call": bool(calls),
                    "normalized": visible if (visible and visible != name) else None,
                }
            except Exception as exc1:
                # 如果端点不认识 "function" 字段，则改用最小形态
                if _looks_like_unknown_function_field_error(exc1):
                    try:
                        tool_min = _make_tool_minimal(name)
                        resp = await self._responses_create(prompt, tool_min)
                        calls = _extract_tool_calls_from_responses(resp)
                        visible = _visible_name_from_tool_call(calls[0]) if calls else None
                        return {
                            "api_status": "ok",
                            "visible_name": visible,
                            "can_call": bool(calls),
                            "normalized": visible if (visible and visible != name) else None,
                            "notes": "fallback:minimal",
                        }
                    except Exception as exc2:
                        # 兜底到 Chat Completions，同样套用顺序
                        try:
                            completion = await self._chat_completions_create(prompt, tool_nested)
                            calls = getattr(completion.choices[0].message, "tool_calls", None) or []
                            visible = None
                            if calls:
                                first = calls[0]
                                fn = getattr(first, "function", None)
                                visible = getattr(fn, "name", None) if fn else None
                            return {
                                "api_status": "ok",
                                "visible_name": visible,
                                "can_call": bool(calls),
                                "normalized": visible if (visible and visible != name) else None,
                                "notes": f"responses_err:{type(exc2).__name__}",
                            }
                        except Exception as exc3:
                            # 再不行就用最小形态试一次 Chat
                            try:
                                tool_min = _make_tool_minimal(name)
                                completion = await self._chat_completions_create(prompt, tool_min)
                                calls = getattr(completion.choices[0].message, "tool_calls", None) or []
                                visible = None
                                if calls:
                                    first = calls[0]
                                    fn = getattr(first, "function", None)
                                    visible = getattr(fn, "name", None) if fn else None
                                return {
                                    "api_status": "ok",
                                    "visible_name": visible,
                                    "can_call": bool(calls),
                                    "normalized": visible if (visible and visible != name) else None,
                                    "notes": f"fallback:minimal_chat ({type(exc3).__name__})",
                                }
                            except Exception as exc4:
                                return {
                                    "api_status": str(exc4),
                                    "visible_name": None,
                                    "can_call": False,
                                    "normalized": None,
                                    "notes": f"ark_chain_fail:{type(exc4).__name__}",
                                }
                # 如果是缺少 tools.function，说明端点需要嵌套形态——我们已经试过；直接报错返回细节
                if _looks_like_missing_tools_function_error(exc1):
                    return {
                        "api_status": str(exc1),
                        "visible_name": None,
                        "can_call": False,
                        "normalized": None,
                        "notes": "ark_requires_nested_function",
                    }
                # 其他错误：兜底 Chat（先 nested，再 minimal）
                try:
                    completion = await self._chat_completions_create(prompt, tool_nested)
                    calls = getattr(completion.choices[0].message, "tool_calls", None) or []
                    visible = None
                    if calls:
                        first = calls[0]
                        fn = getattr(first, "function", None)
                        visible = getattr(fn, "name", None) if fn else None
                    return {
                        "api_status": "ok",
                        "visible_name": visible,
                        "can_call": bool(calls),
                        "normalized": visible if (visible and visible != name) else None,
                        "notes": f"ark_resp_err:{type(exc1).__name__}",
                    }
                except Exception as exc_chat1:
                    try:
                        tool_min = _make_tool_minimal(name)
                        completion = await self._chat_completions_create(prompt, tool_min)
                        calls = getattr(completion.choices[0].message, "tool_calls", None) or []
                        visible = None
                        if calls:
                            first = calls[0]
                            fn = getattr(first, "function", None)
                            visible = getattr(fn, "name", None) if fn else None
                        return {
                            "api_status": "ok",
                            "visible_name": visible,
                            "can_call": bool(calls),
                            "normalized": visible if (visible and visible != name) else None,
                            "notes": f"ark_chat_minimal:{type(exc_chat1).__name__}",
                        }
                    except Exception as exc_chat2:
                        return {
                            "api_status": str(exc_chat2),
                            "visible_name": None,
                            "can_call": False,
                            "normalized": None,
                            "notes": f"ark_chat_fail:{type(exc_chat2).__name__}",
                        }

        # ---- 非 Ark：常规路径 ----
        tool_nested = _make_tool_nested(name)
        try:
            resp = await self._responses_create(prompt, tool_nested)
            calls = _extract_tool_calls_from_responses(resp)
            visible = _visible_name_from_tool_call(calls[0]) if calls else None
            return {
                "api_status": "ok",
                "visible_name": visible,
                "can_call": bool(calls),
                "normalized": visible if (visible and visible != name) else None,
            }
        except NotFoundError as exc_not_found:
            try:
                completion = await self._chat_completions_create(prompt, tool_nested)
                calls = getattr(completion.choices[0].message, "tool_calls", None) or []
                visible = None
                if calls:
                    first = calls[0]
                    fn = getattr(first, "function", None)
                    visible = getattr(fn, "name", None) if fn else None
                return {
                    "api_status": "ok",
                    "visible_name": visible,
                    "can_call": bool(calls),
                    "normalized": visible if (visible and visible != name) else None,
                }
            except Exception as exc2:
                return {
                    "api_status": str(exc2),
                    "visible_name": None,
                    "can_call": False,
                    "normalized": None,
                    "notes": type(exc2).__name__,
                }
        except Exception as exc1:
            if _looks_like_missing_tools_name_error(exc1):
                try:
                    tool_top = _make_tool_toplevel(name)
                    resp = await self._responses_create(prompt, tool_top)
                    calls = _extract_tool_calls_from_responses(resp)
                    visible = _visible_name_from_tool_call(calls[0]) if calls else None
                    return {
                        "api_status": "ok",
                        "visible_name": visible,
                        "can_call": bool(calls),
                        "normalized": visible if (visible and visible != name) else None,
                    }
                except Exception as exc2:
                    try:
                        completion = await self._chat_completions_create(prompt, tool_nested)
                        calls = getattr(completion.choices[0].message, "tool_calls", None) or []
                        visible = None
                        if calls:
                            first = calls[0]
                            fn = getattr(first, "function", None)
                            visible = getattr(fn, "name", None) if fn else None
                        return {
                            "api_status": "ok",
                            "visible_name": visible,
                            "can_call": bool(calls),
                            "normalized": visible if (visible and visible != name) else None,
                        }
                    except Exception as exc3:
                        return {
                            "api_status": str(exc3),
                            "visible_name": None,
                            "can_call": False,
                            "normalized": None,
                            "notes": type(exc3).__name__,
                        }
            else:
                return {
                    "api_status": str(exc1),
                    "visible_name": None,
                    "can_call": False,
                    "normalized": None,
                    "notes": type(exc1).__name__,
                }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def generate_cases() -> List[tuple[str, str]]:
    cases: List[tuple[str, str]] = []
    for category, func in [
        ("length", _length_cases),
        ("charset", _charset_cases),
        ("unicode", _unicode_cases),
        ("directional", _directional_cases),
        ("boundary", _boundary_cases),
        ("reserved", _reserved_cases),
        ("path", _path_cases),
        ("control", _control_cases),
    ]:
        cases.extend((category, name) for name in func())
    return cases[:MAX_CALLS]


async def probe_all(
    adapter: OpenAIAdapter,
    cases: List[tuple[str, str]],
    logger: ExperimentLogger,
) -> List[ProbeResult]:
    sem = asyncio.Semaphore(1)

    async def worker(category: str, name: str) -> ProbeResult:
        async with sem:
            result = await adapter.probe_single(name)
            params = {"model": adapter.model_name, "category": category, "name": name}
            logger.log("name_probe", params, result)
            time.sleep(2)
            return ProbeResult(name=name, category=category, **result)

    tasks = [asyncio.create_task(worker(cat, name)) for cat, name in cases]
    return await asyncio.gather(*tasks)


def write_results(results: List[ProbeResult], model_name: str) -> None:
    out_dir = RESULT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "name_probe.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "name", "category", "api_status", "visible_name",
                "can_call", "normalized", "notes",
            ],
        )
        writer.writeheader()
        for res in results:
            writer.writerow(asdict(res))


async def main() -> None:
    cases = generate_cases()
    logger = ExperimentLogger(LOG_DIR)

    async def worker(m: dict) -> List[Dict[str, Any]]:
        adapter = OpenAIAdapter(
            api_key=m["api_key"],
            model_name=m["name"],
            base_url=m.get("base_url"),
        )
        results = await probe_all(adapter, cases, logger)
        write_results(results, m["name"])
        return [asdict(r) for r in results]

    results = await run_for_models(MODELS, worker)
    ResultWriter(RESULT_DIR).write("name_probe", results)


if __name__ == "__main__":
    asyncio.run(main())
