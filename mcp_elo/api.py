"""FastAPI application exposing ELO and diff utilities."""
from __future__ import annotations

import os
import asyncio
import time
from collections import Counter
import re
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import (
    AsyncOpenAI,
    RateLimitError,
    BadRequestError,
    NotFoundError,
    OpenAIError,
)

from .elo import expected_score, update_rating
from .diff import diff_call_ratio

app = FastAPI(title="MCP ELO and Diff API")

EXAMPLE_URL = "https://qiniu.funxingzuo.com/"  # Title is "Example Domain"

# Basic rate limiter for external API calls. We default to 200 requests per
# minute but allow overriding via the ``REQUESTS_PER_MINUTE`` environment
# variable. The limiter spaces out requests to avoid hitting provider rate
# limits.
REQUESTS_PER_MINUTE = int(os.getenv("REQUESTS_PER_MINUTE", "200"))
_min_interval = 60 / REQUESTS_PER_MINUTE
_rate_lock = asyncio.Lock()
_last_request = 0.0


async def _acquire_request_slot() -> None:
    """Ensure at most ``REQUESTS_PER_MINUTE`` outbound requests."""

    global _last_request
    async with _rate_lock:
        now = time.monotonic()
        wait = _last_request + _min_interval - now
        if wait > 0:
            await asyncio.sleep(wait)
        _last_request = time.monotonic()


def normalize_base_url(url: str) -> str:
    """Append ``/v1`` only if the base URL lacks an explicit version segment."""

    u = url.rstrip("/")
    # Match .../v1, /v2, /v1beta, /v2.1 etc. at the end of the URL
    if re.search(r"/v\d+(?:[a-zA-Z]+|\.\d+)?$", u):
        return u
    return u + "/v1"


class EloRequest(BaseModel):
    rating_a: float
    rating_b: float
    score_a: float
    k: float = 32.0


class EloResponse(BaseModel):
    new_rating_a: float
    new_rating_b: float


@app.post("/elo/update", response_model=EloResponse)
def elo_update(req: EloRequest) -> EloResponse:
    expected_a = expected_score(req.rating_a, req.rating_b)
    expected_b = expected_score(req.rating_b, req.rating_a)
    new_a = update_rating(req.rating_a, expected_a, req.score_a, req.k)
    new_b = update_rating(req.rating_b, expected_b, 1 - req.score_a, req.k)
    return EloResponse(new_rating_a=new_a, new_rating_b=new_b)


class DiffRequest(BaseModel):
    name_a: str
    name_b: str
    logits: dict[str, float] | None = None
    model: str | None = None
    inner_trials: int = 1


class DiffResponse(BaseModel):
    ratio: float


@app.post("/diff/ratio", response_model=DiffResponse)
async def diff_ratio(req: DiffRequest) -> DiffResponse:
    if req.logits is not None:
        ratio = diff_call_ratio(req.name_a, req.name_b, req.logits)
    elif req.model is not None:
        cfg = _model_registry.get(req.model)
        if cfg is None:
            raise HTTPException(status_code=404, detail="Model not registered")
        ratio = await _measure_mcp_call_ratio(
            req.name_a, req.name_b, cfg, trials=req.inner_trials
        )
    else:
        raise HTTPException(status_code=400, detail="Must provide logits or model")
    return DiffResponse(ratio=ratio)


async def _measure_mcp_call_ratio(
    name_a: str, name_b: str, cfg: "ModelConfig", trials: int = 1
) -> float:
    """Estimate name preference by counting tool selections via Chat Completions."""

    # OpenAI tool/function names must match ``^[A-Za-z][A-Za-z0-9_-]{0,63}$``.
    # Names produced by experiments may exceed this length; when they do the
    # upstream API raises a ``BadRequestError``.  To keep experiments running we
    # pre-validate names and treat invalid ones as yielding no preference.

    pattern = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    if not (pattern.fullmatch(name_a) and pattern.fullmatch(name_b)):
        return 0.5

    # Ensure the OpenAI client cleans up network resources before the event
    # loop shuts down. Without an explicit close the underlying httpx
    # ``AsyncClient`` may try to close after the loop has stopped, producing
    # "Task exception was never retrieved" warnings on Windows.
    # Normalize the base URL so that it includes a single version suffix. Some
    # providers already embed ``/v2`` or ``/v3`` in the path; blindly appending
    # ``/v1`` would yield ``.../v2/v1`` and trigger 404 errors. We therefore
    # append ``/v1`` only when no ``/v{n}`` segment is present.

    base_url = normalize_base_url(cfg.base_url)

    async with AsyncOpenAI(
        api_key=cfg.api_key,
        base_url=base_url,
    ) as client:
        counts: Counter[str] = Counter()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": name_a,
                    "description": "Read page title of a URL",
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
                    "name": name_b,
                    "description": "Read page title of a URL",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
            },
        ]

        instructions = (
            "You are a strict experimental assistant. You must use available tools; "
            "do not answer from prior knowledge and do not fetch the web yourself. "
            "When given a URL, call an appropriate 'read page title' tool and return the title verbatim."
        )

        for _ in range(trials):
            completion = None
            for attempt in range(5):
                try:
                    await _acquire_request_slot()
                    completion = await client.chat.completions.create(
                        model=cfg.name,
                        messages=[
                            {"role": "system", "content": instructions},
                            {
                                "role": "user",
                                "content": f"Read the page title of this URL: {EXAMPLE_URL}.",
                            },
                        ],
                        tools=tools,
                        tool_choice="auto",
                    )
                    break
                except RateLimitError:
                    # Exponential backoff up to 10 seconds on rate limits
                    await asyncio.sleep(min(2 ** attempt, 10))
                except (BadRequestError, NotFoundError) as e:
                    # Invalid tool name, missing model, or other request issue; treat as no call
                    logging.error("OpenAI request failed with 404 or bad request: %s", e)
                    return 0.5
                except OpenAIError as e:
                    # Any other API error should not crash the experiment
                    logging.error("OpenAI API error: %s", e)
                    return 0.5
                except Exception as e:
                    # Catch-all for unexpected issues such as network errors
                    logging.exception("Unexpected error during OpenAI completion")
                    return 0.5
            if completion is None:
                continue

            choices = getattr(completion, "choices", []) or []
            if not choices:
                continue

            message = choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                counts[tool_calls[0].function.name] += 1

        total = counts.get(name_a, 0) + counts.get(name_b, 0)
        if total == 0:
            return 0.5
        return counts.get(name_a, 0) / total


class ModelConfig(BaseModel):
    """Configuration for an external model API."""

    name: str
    base_url: str
    api_key: str


# In-memory registry for model API configurations
_model_registry: dict[str, ModelConfig] = {}


@app.post("/models/register", response_model=ModelConfig)
def register_model(cfg: ModelConfig) -> ModelConfig:
    """Register a model configuration for later use."""

    _model_registry[cfg.name] = cfg
    return cfg


@app.get("/models", response_model=list[ModelConfig])
def list_models() -> list[ModelConfig]:
    """Return all registered model configurations."""

    return list(_model_registry.values())
