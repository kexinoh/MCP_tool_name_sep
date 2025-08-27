"""Utilities for logging experiment parameters and metrics."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any
import asyncio
import json


@dataclass
class ExperimentRecord:
    """Single experiment run record."""

    params: dict[str, Any]
    metrics: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ExperimentLogger:
    """Log experiment runs to JSON files.

    Each call to :meth:`log` writes a ``.json`` file containing the
    experiment parameters and metrics along with a UTC timestamp.
    """

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, name: str, params: dict[str, Any], metrics: dict[str, Any]) -> Path:
        """Persist a new experiment record.

        Parameters
        ----------
        name:
            Base name for the log file. The timestamp will be appended.
        params, metrics:
            Dictionaries describing the run configuration and results.
        """
        record = ExperimentRecord(params=params, metrics=metrics)
        safe_time = record.timestamp.replace(":", "-")
        path = self.log_dir / f"{name}_{safe_time}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(record), fh, indent=2, ensure_ascii=False)
        return path


class ResultWriter:
    """Persist aggregated experiment results.

    Each call to :meth:`write` outputs a timestamped JSON file to avoid
    overwriting previous results when rerunning experiments. Files are
    stored under a dedicated directory for each experiment.
    """

    def __init__(self, result_dir: str | Path):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, data: dict[str, Any]) -> Path:
        """Write ``data`` to a timestamped ``<name>_<time>.json`` file."""
        timestamp = datetime.utcnow().isoformat().replace(":", "-")
        path = self.result_dir / f"{name}_{timestamp}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        return path


async def run_for_models(models: list[dict[str, Any]], worker):
    """Run a coroutine ``worker`` for each model concurrently.

    Parameters
    ----------
    models:
        List of model configuration dictionaries.
    worker:
        Callable accepting a single model dict and returning awaitable
        result for that model.

    Returns
    -------
    Dict[str, Any]
        Mapping of model name to the result returned by ``worker``.
    """
    tasks = {m["name"]: asyncio.create_task(worker(m)) for m in models}
    results: dict[str, Any] = {}
    for name, task in tasks.items():
        results[name] = await task
    return results
