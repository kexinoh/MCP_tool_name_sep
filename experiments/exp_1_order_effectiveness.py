"""
Experiment: Does tool order alone affect selection probability?

We run paired AB/BA trials for each name pair and apply McNemar's test.
Key: Treat success as "selected A (the same name)" rather than "selected the first".
This naturally controls for inherent name bias.

- /diff/ratio returns:
    1.0 -> first name selected in that call
    0.0 -> second name selected
    0.5 -> no tool call; skip this paired trial
- For each paired AB/BA:
    x_ab = 1 if A selected when A is first (AB), else 0
    x_ba = 1 if A selected when A is second (BA), else 0  [note: BA returns 1 means B selected]
- McNemar counts:
    n10 = #(x_ab=1, x_ba=0): putting A first helps A
    n01 = #(x_ab=0, x_ba=1): putting A first hurts A
- Per-pair we report:
    OR_first = (n10+0.5)/(n01+0.5), 95% CI (normal approx on logOR), exact McNemar p
    flip rate, ΔP = P(select A | A first) - P(select A | A second)
- Global pooled OR/CI/p via summing n10/n01 across pairs.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Dict, List, Tuple

import httpx

from mcp_elo.api import app
from mcp_elo.experiment import ExperimentLogger, ResultWriter, run_for_models
from experiments.config import load_models

# ------------------ Config ------------------

LOG_DIR = Path(__file__).resolve().parent / "logs" / "exp1"
RESULT_DIR = Path(__file__).resolve().parent / "results" / "exp1"

# Load model configurations
MODELS = load_models()

DEFAULT_MODEL = MODELS[0]["name"] if MODELS else ""

# Pairs of tool names (A,B)
NAME_PAIRS: List[Tuple[str, str]] = [
    ("GoogleSearch_read_url", "AmazonBedrock_read_url"),
    ("assistant_read_url", "system_read_url"),
    ("OpenAI_read_url", "Mistral_read_url"),
    ("bot_read_url", "agent_read_url"),
    ("service_read_url", "worker_read_url"),
]

# ------------------ Stats helpers ------------------

def mcnemar_or_and_ci(n10: int, n01: int) -> tuple[float, float, float]:
    """Odds ratio and ~95% CI with Haldane–Anscombe correction."""
    b, c = n10, n01
    or_hat = (b + 0.5) / (c + 0.5)
    se = math.sqrt(1.0 / (b + 0.5) + 1.0 / (c + 0.5))
    lo = math.exp(math.log(or_hat) - 1.96 * se)
    hi = math.exp(math.log(or_hat) + 1.96 * se)
    return or_hat, lo, hi

def mcnemar_exact_p(n10: int, n01: int) -> float:
    """Exact two-sided p-value via binomial tail on discordant pairs."""
    from math import comb
    b, c = n10, n01
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)

# ------------------ API call ------------------

async def one_call_ratio(
    client: httpx.AsyncClient,
    name_a: str,
    name_b: str,
    *,
    model: str | None = None,
    logits: Dict[str, float] | None = None,
) -> float:
    """Call /diff/ratio once; expect 1.0/0.0 or 0.5 (skip)."""
    if (model is None) == (logits is None):
        raise ValueError("Provide either a model name or logits")

    payload = {"name_a": name_a, "name_b": name_b, "inner_trials": 1}
    if model:
        payload["model"] = model
    else:
        payload["logits"] = logits

    resp = await client.post("/diff/ratio", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return float(data["ratio"])

# ------------------ Experiment core ------------------

async def run_experiment(
    model_name: str = DEFAULT_MODEL, trials: int = 60, *, concurrency: int = 20
) -> dict:
    """
    Run paired AB/BA trials with bounded concurrency.
    For each pair:
      - compute McNemar n10/n01, OR, CI, exact p
      - diagnostics: flip rate, ΔP(A|first) - P(A|second), ties
    Also compute pooled (global) n10/n01 and report global OR/CI/p.
    """
    logger = ExperimentLogger(LOG_DIR)
    transport = httpx.ASGITransport(app=app)

    pair_records: list[dict[str, float | int]] = []
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for m in MODELS:
            await client.post(
                "/models/register",
                json={"name": m["name"], "base_url": m["base_url"], "api_key": m["api_key"]},
            )

        global_n10 = 0
        global_n01 = 0

        print(
            f"Per-pair McNemar test (does placing A first increase A being selected) "
            f"[model={model_name}]\n"
        )
        for name_a, name_b in NAME_PAIRS:
            n10 = n01 = 0
            sum_xab = 0  # # of times A selected when A is first
            sum_xba = 0  # # of times A selected when A is second
            tie_11 = tie_00 = 0
            skipped = 0

            sem = asyncio.Semaphore(concurrency)

            async def paired_trial() -> tuple[float, float]:
                async with sem:
                    # AB then BA (order doesn't matter as long as paired)
                    return await asyncio.gather(
                        one_call_ratio(client, name_a, name_b, model=model_name),
                        one_call_ratio(client, name_b, name_a, model=model_name),
                    )

            results = await asyncio.gather(*(paired_trial() for _ in range(trials)))

            valid_pairs = 0
            for r_ab, r_ba in results:
                # Skip if either side had no tool call
                if r_ab == 0.5 or r_ba == 0.5:
                    skipped += 1
                    continue

                valid_pairs += 1

                # Success = selected A (NOT "selected first")
                x_ab = 1 if r_ab == 1.0 else 0           # AB: 1 means A selected
                x_ba = 1 if r_ba == 0.0 else 0           # BA: 1 means A selected (since 1.0 means B)

                sum_xab += x_ab
                sum_xba += x_ba

                if x_ab == 1 and x_ba == 0:
                    n10 += 1         # putting A first helped A
                elif x_ab == 0 and x_ba == 1:
                    n01 += 1         # putting A first hurt A
                elif x_ab == 1 and x_ba == 1:
                    tie_11 += 1      # both orders select A
                else:
                    tie_00 += 1      # both orders select B

            # Stats per pair
            or_hat, ci_lo, ci_hi = mcnemar_or_and_ci(n10, n01)
            pval = mcnemar_exact_p(n10, n01)

            # Diagnostics
            flips = n10 + n01
            flip_rate = flips / max(valid_pairs, 1)
            p_A_first  = sum_xab / max(valid_pairs, 1)
            p_A_second = sum_xba / max(valid_pairs, 1)
            delta_p = p_A_first - p_A_second  # probability lift for A when placed first

            logger.log(
                "order_effect_binary",
                {"A": name_a, "B": name_b},
                {
                    "valid_pairs": valid_pairs,
                    "skipped_pairs": skipped,
                    "n10": n10,
                    "n01": n01,
                    "ties_11_both_A": tie_11,
                    "ties_00_both_B": tie_00,
                    "OR_first": or_hat,
                    "CI_lo": ci_lo,
                    "CI_hi": ci_hi,
                    "p": pval,
                    "flip_rate": flip_rate,
                    "p_A_first": p_A_first,
                    "p_A_second": p_A_second,
                    "delta_p": delta_p,
                },
            )

            pair_records.append(
                {
                    "A": name_a,
                    "B": name_b,
                    "valid_pairs": valid_pairs,
                    "skipped": skipped,
                    "n10": n10,
                    "n01": n01,
                    "OR_first": or_hat,
                    "CI_lo": ci_lo,
                    "CI_hi": ci_hi,
                    "p": pval,
                    "flip_rate": flip_rate,
                    "p_A_first": p_A_first,
                    "p_A_second": p_A_second,
                    "delta_p": delta_p,
                }
            )

            sig = " **significant**" if (pval < 0.05 and not (ci_lo <= 1 <= ci_hi)) else ""
            print(
                f"{model_name} {name_a} vs {name_b}:\n"
                f"  valid_pairs={valid_pairs}, skipped={skipped}, discordant pairs n10={n10}, n01={n01}\n"
                f"  OR_first={or_hat:.3f} 95%CI[{ci_lo:.3f}, {ci_hi:.3f}], p={pval:.3g}{sig}\n"
                f"  Interpretation: placing A first increases the chance A is chosen by ΔP≈{delta_p:.3f} "
                f"(P_first={p_A_first:.3f} vs P_second={p_A_second:.3f})\n"
                f"  Diagnostics: flip_rate={flip_rate:.3f} (higher means order affects choice), both_pick_A={tie_11}, both_pick_B={tie_00}\n"
            )

            global_n10 += n10
            global_n01 += n01

        # Global pooled McNemar / Mantel–Haenszel common OR
        or_g, lo_g, hi_g = mcnemar_or_and_ci(global_n10, global_n01)
        p_g = mcnemar_exact_p(global_n10, global_n01)
        print(
            f"Global order effect (pooled McNemar / common odds ratio) [model={model_name}]:"
        )
        print(
            f"  Total discordant pairs: n10={global_n10}, n01={global_n01}\n"
            f"  Common OR_first={or_g:.3f} 95%CI[{lo_g:.3f}, {hi_g:.3f}], p={p_g:.3g}"
            f"{' **significant**' if (p_g < 0.05 and not (lo_g <= 1 <= hi_g)) else ''}\n"
            "  Interpretation: OR_first>1 means placing the name first favors its selection; <1 means the opposite; a CI crossing 1 implies no statistical significance."
        )

    return {
        "pairs": pair_records,
        "global": {
            "n10": global_n10,
            "n01": global_n01,
            "OR_first": or_g,
            "CI_lo": lo_g,
            "CI_hi": hi_g,
            "p": p_g,
        },
    }

# ------------------ Runner ------------------

async def main() -> None:
    async def worker(m: dict) -> dict:
        print(f"Running model: {m['name']}")
        return await run_experiment(model_name=m["name"], trials=60, concurrency=20)

    results = await run_for_models(MODELS, worker)
    ResultWriter(RESULT_DIR).write("order_effect", results)


if __name__ == "__main__":
    asyncio.run(main())
