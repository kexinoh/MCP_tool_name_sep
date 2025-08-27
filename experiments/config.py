from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Dict, List, Any

from mcp_elo.config_parser import parse_typed_value


def load_models(config_path: str | Path | None = None) -> List[Dict[str, Any]]:
    """Load model configurations from a TOML file.

    Parameters
    ----------
    config_path:
        Optional path to the configuration file. If not provided, the function
        looks for ``config.toml`` at the repository root.
    """
    if config_path is None:
        # Repository root is two levels above this file
        config_path = Path(__file__).resolve().parents[1] / "config.toml"
    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    # Coerce any typed values like "integer=42" within model configs
    models = data.get("models", [])
    for model in models:
        for key, val in list(model.items()):
            model[key] = parse_typed_value(val)
    enabled = data.get("enable_model")
    if enabled:
        models = [m for m in models if m.get("name") in enabled]
    return models
