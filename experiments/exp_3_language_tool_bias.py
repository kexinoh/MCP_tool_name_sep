"""Compare tool-call preference with and without the English tool.

It runs TWO experiments:
1) 2-tool setup: ["读取网页标题", "ページタイトルを読む"]
2) 3-tool setup: ["读取网页标题", "ページタイトルを読む", "ReadPageTitle"]

Each setup is tested under zh/jp/en instruction environments.
Results are saved separately for each setup.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

from openai import AsyncOpenAI, RateLimitError
from experiments.config import load_models
from mcp_elo.experiment import ExperimentLogger, ResultWriter, run_for_models

# Example URL whose title is "Example Domain"
EXAMPLE_URL = "https://qiniu.funxingzuo.com/"

MODELS = load_models()

LOG_DIR = Path(__file__).resolve().parent / "logs" / "exp3"
RESULT_DIR = Path(__file__).resolve().parent / "results" / "exp3"
LOGGER = ExperimentLogger(LOG_DIR)

DEFAULT_MODEL = "ernie-lite-pro-128k"


def normalize_base_url(url: str | None) -> str | None:
    """Only append /v1 if the base URL does not already end with a version suffix."""
    if not url:
        return None
    u = url.strip().rstrip("/")
    if re.search(r"/v\d+([a-z]+|\.\d+)?$", u, flags=re.IGNORECASE):
        return u
    return u + "/v1"


def build_name_map(include_en: bool) -> Dict[str, str]:
    """Return the display-name mapping for the current tool set."""
    name_map = {
        "读取网页标题": "读取网页标题",
        "ページタイトルを読む": "ページタイトルを読む",
    }
    if include_en:
        name_map["ReadPageTitle"] = "Read page title"
    return name_map


def build_tools(include_en: bool) -> List[dict]:
    """Return the tool list for the current tool set."""
    tools: List[dict] = [
        {
            "type": "function",
            "function": {
                "name": "读取网页标题",
                "description": "读取网页标题",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ページタイトルを読む",
                "description": "ページタイトルを読む",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
        },
    ]
    if include_en:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "ReadPageTitle",
                    "description": "Read page title",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
            }
        )
    return tools


def build_messages(language: str) -> List[dict]:
    """Build system/user messages for the given instruction language."""
    if language == "zh":
        instructions = (
            "你是一个严格的实验助手。必须通过可用的工具完成任务；"
            "不要凭空回答，不要自己发起网络请求。"
            "当用户给出URL时，请调用合适的工具读取网页标题并原样返回。"
        )
        user_prompt = f"请读取这个URL的网页标题：{EXAMPLE_URL}。只输出标题本身。（必须通过工具）"
    elif language == "jp":
        instructions = (
            "あなたは厳格な実験助手です。利用可能なツールを必ず使い、"
            "自分で推測したりネットにアクセスしたりしないでください。"
            "URLが与えられたら、適切なツールを呼び出してページタイトルをそのまま返してください。"
        )
        user_prompt = f"このURLのページタイトルを読んでください: {EXAMPLE_URL}。タイトルのみを出力してください。（必ずツールを使用）"
    elif language == "en":
        instructions = (
            "You are a strict experimental assistant. You must use available tools to complete the task; "
            "do not answer from prior knowledge, and do not perform your own web requests. "
            "When a URL is provided, call an appropriate tool to read the page title and return it verbatim."
        )
        user_prompt = f"Read the page title of this URL: {EXAMPLE_URL}. Output only the title. (You must use a tool)"
    else:
        raise ValueError("language must be 'zh', 'jp', or 'en'")

    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]


async def run_one_trial(
    client: AsyncOpenAI,
    language: str,
    model_name: str,
    tools: List[dict],
) -> Counter[str]:
    """Run a single trial and count which tool was selected."""
    messages = build_messages(language)

    for attempt in range(5):
        try:
            completion = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            break
        except RateLimitError:
            await asyncio.sleep(min(2**attempt, 10))
    else:
        return Counter()

    counts: Counter[str] = Counter()
    choice = completion.choices[0]
    message = getattr(choice, "message", None)
    tool_calls = getattr(message, "tool_calls", None) if message else None
    if tool_calls:
        name = tool_calls[0].function.name
        counts[name] += 1
    return counts


async def call_ratio(
    model: dict,
    language: str,
    tools: List[dict],
    name_map: Dict[str, str],
    trials: int = 20,
    concurrency: int = 2,
    logger: ExperimentLogger | None = None,
) -> Dict[str, Dict[str, float]]:
    """Run multiple trials concurrently and return call counts and ratios."""
    model_name = model.get("name") or model.get("model") or DEFAULT_MODEL
    logger = logger or LOGGER

    client_args = {"api_key": model.get("api_key")}
    base_url = normalize_base_url(model.get("base_url"))
    if base_url:
        client_args["base_url"] = base_url

    async with AsyncOpenAI(**client_args) as client:
        results: Counter[str] = Counter()
        sem = asyncio.Semaphore(concurrency)

        async def worker() -> Counter[str]:
            async with sem:
                return await run_one_trial(client, language, model_name, tools)

        tasks = [asyncio.create_task(worker()) for _ in range(trials)]
        for idx, task in enumerate(asyncio.as_completed(tasks), start=1):
            trial_result = await task
            results.update(trial_result)

            if logger:
                selected_name = next((k for k, v in trial_result.items() if v), None)
                tool_display = name_map.get(selected_name, "none") if selected_name else "none"
                params = {"model": model_name, "language": language, "trial": idx}
                metrics = {"selected_tool": tool_display}
                logger.log("language_tool_bias", params, metrics)

    # Ensure every tool in this setup has an entry
    for name in name_map:
        results.setdefault(name, 0)

    total = sum(results.values()) or 1
    summary = {
        name_map[name]: {"count": results[name], "ratio": results[name] / total}
        for name in name_map
    }

    if logger:
        params = {"model": model_name, "language": language, "trials": trials}
        logger.log("language_tool_bias_summary", params, summary)
    return summary


async def run_setup_for_models(
    include_en: bool,
    trials: int,
    concurrency: int,
    logger: ExperimentLogger,
) -> dict:
    """Run one setup (2-tool or 3-tool) across all models and languages."""
    tools = build_tools(include_en)
    name_map = build_name_map(include_en)

    async def worker(model: dict) -> dict:
        zh_stats = await call_ratio(
            model, "zh", tools=tools, name_map=name_map, trials=trials, concurrency=concurrency, logger=logger
        )
        jp_stats = await call_ratio(
            model, "jp", tools=tools, name_map=name_map, trials=trials, concurrency=concurrency, logger=logger
        )
        en_stats = await call_ratio(
            model, "en", tools=tools, name_map=name_map, trials=trials, concurrency=concurrency, logger=logger
        )
        return {"zh": zh_stats, "jp": jp_stats, "en": en_stats}

    return await run_for_models(MODELS, worker)


async def main() -> None:
    trials = 20
    concurrency = 10

    # -------- 2-tool setup (no English tool) --------
    results_2 = await run_setup_for_models(include_en=False, trials=trials, concurrency=concurrency, logger=LOGGER)
    ResultWriter(RESULT_DIR).write("language_tool_bias_2tools", results_2)

    for name, stats in results_2.items():
        print(f"[2-tools] {name} Chinese (trials={trials}):\n" + json.dumps(stats["zh"], ensure_ascii=False, indent=2))
        print(f"[2-tools] {name} Japanese (trials={trials}):\n" + json.dumps(stats["jp"], ensure_ascii=False, indent=2))
        print(f"[2-tools] {name} English (trials={trials}):\n" + json.dumps(stats["en"], ensure_ascii=False, indent=2))

    # -------- 3-tool setup (with English tool) --------
    results_3 = await run_setup_for_models(include_en=True, trials=trials, concurrency=concurrency, logger=LOGGER)
    ResultWriter(RESULT_DIR).write("language_tool_bias_3tools", results_3)

    for name, stats in results_3.items():
        print(f"[3-tools] {name} Chinese (trials={trials}):\n" + json.dumps(stats["zh"], ensure_ascii=False, indent=2))
        print(f"[3-tools] {name} Japanese (trials={trials}):\n" + json.dumps(stats["jp"], ensure_ascii=False, indent=2))
        print(f"[3-tools] {name} English (trials={trials}):\n" + json.dumps(stats["en"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
