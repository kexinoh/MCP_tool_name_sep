"""Name style bias experiment: compare how server naming conventions affect call preference.

修复点：
- 不再在 run_experiment 内修改全局列表；改为基于常量生成局部名称。
- 确保每个模型使用相同的一组参赛名称与对阵组合（可复现实验）。
- 防止重复后缀拼接（幂等）。
"""

from __future__ import annotations

import asyncio
import os
import random
from itertools import product, combinations
from pathlib import Path
from typing import Iterable, List, Tuple

import httpx

from mcp_elo.api import app
from mcp_elo.experiment import ExperimentLogger, ResultWriter, run_for_models
from mcp_elo.elo import EloRatingSystem
from experiments.config import load_models

LOG_DIR = Path(__file__).resolve().parent / "logs" / "exp2"
RESULT_DIR = Path(__file__).resolve().parent / "results" / "exp2"

MODELS = load_models()
DEFAULT_MODEL = MODELS[0]["name"] if MODELS else ""

MAX_API_CALLS = int(os.getenv("MAX_API_CALLS", "500"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "20250825"))

# ---- 常量：原始名称（不带后缀），不要修改这三组 ----
BIG_COMPANIES_BASE = [
    "GoogleSearch",
    "MicrosoftAzure",
    "AmazonBedrock",
    "TencentCloud",
    "BaiduResearch",
    "HuaweiCloud",
]

DEVELOPER_COMPANIES_BASE = [
    "OpenAIModel",
    "AnthropicClaude",
    "DeepMindModel",
    "CohereCommand",
    "MistralAI",
    "StabilityAI",
    "KIMI",
    "KIMIAI",
]

GENERIC_NAMES_BASE = [
    "system",
    "assistant",
    "service",
    "worker",
    "agent",
    "bot",
]


def with_suffix_once(names: Iterable[str], suffix: str = "_read_url") -> List[str]:
    """对每个名字最多追加一次后缀（幂等）。"""
    out = []
    for n in names:
        out.append(n if n.endswith(suffix) else f"{n}{suffix}")
    return out


def build_players_and_pairs(
    suffix: str = "_read_url",
    limit_pairs: int | None = MAX_API_CALLS,
    seed: int = RANDOM_SEED,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """基于常量基表，构造一次性的参赛名单与对阵组合（全模型共用，保证可复现）。"""
    big = with_suffix_once(BIG_COMPANIES_BASE, suffix)
    dev = with_suffix_once(DEVELOPER_COMPANIES_BASE, suffix)
    gen = with_suffix_once(GENERIC_NAMES_BASE, suffix)

    # 参赛选手（去重保持顺序）
    players = []
    for lst in (big, dev, gen):
        for n in lst:
            if n not in players:
                players.append(n)

    # 对阵：跨组 + 组内
    pairs: List[Tuple[str, str]] = []
    pairs += list(product(big, gen))
    pairs += list(product(dev, gen))
    pairs += list(product(big, dev))
    pairs += list(combinations(big, 2))
    pairs += list(combinations(dev, 2))
    pairs += list(combinations(gen, 2))

    rng = random.Random(seed)
    rng.shuffle(pairs)

    if limit_pairs is not None:
        pairs = pairs[:limit_pairs]

    return players, pairs


async def run_experiment(
    model_name: str = DEFAULT_MODEL,
    players: List[str] | None = None,
    pairs: List[Tuple[str, str]] | None = None,
    concurrency: int = 30,
) -> dict:
    """按给定的参赛名单和对阵组合运行一次实验。"""

    # 允许外部传入固定的 players/pairs，从而保证跨模型复现
    if players is None or pairs is None:
        players, pairs = build_players_and_pairs()

    logger = ExperimentLogger(LOG_DIR)
    rating = EloRatingSystem()

    for name in players:
        rating.add_player(name)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 注册模型
        for m in MODELS:
            await client.post(
                "/models/register",
                json={"name": m["name"], "base_url": m["base_url"], "api_key": m["api_key"]},
            )

        sem = asyncio.Semaphore(concurrency)

        async def one_match(name_a: str, name_b: str) -> tuple[str, str, float]:
            async with sem:
                payload = {"name_a": name_a, "name_b": name_b, "model": model_name}
                resp = await client.post("/diff/ratio", json=payload)
                ratio = resp.json()["ratio"]
                return name_a, name_b, ratio

        tasks = [asyncio.create_task(one_match(a, b)) for a, b in pairs]
        for t in asyncio.as_completed(tasks):
            name_a, name_b, ratio = await t
            rating.record_match(name_a, name_b, ratio)

            params = {"name_a": name_a, "name_b": name_b}
            metrics = {
                "call_ratio": ratio,
                "rating_a": rating.ratings[name_a],
                "rating_b": rating.ratings[name_b],
            }
            logger.log("name_bias", params, metrics)
            print(
                f"{model_name} {name_a} vs {name_b}: ratio={ratio:.3f} "
                f"-> ratings: {rating.ratings[name_a]:.1f} / {rating.ratings[name_b]:.1f}"
            )

    ranking = dict(
        sorted(rating.ratings.items(), key=lambda item: item[1], reverse=True)
    )

    print(f"\nFinal rankings for {model_name}:")
    for name, score in ranking.items():
        print(f"{name}: {score:.1f}")
    return {"ranking": ranking}


async def main() -> None:
    # 关键：预先固定参赛选手与对阵组合，供所有模型共用
    players, pairs = build_players_and_pairs()

    async def worker(m: dict) -> dict:
        print(f"Running model: {m['name']}")
        return await run_experiment(model_name=m["name"], players=players, pairs=pairs)

    results = await run_for_models(MODELS, worker)
    ResultWriter(RESULT_DIR).write("name_bias", results)


if __name__ == "__main__":
    asyncio.run(main())
