"""Utilities for parsing typed values from configuration strings."""
from __future__ import annotations
from typing import Any, Callable

# Parsers for supported type tags.
_TYPE_PARSERS: dict[str, Callable[[str], Any]] = {
    "string": lambda v: v,
    "float": float,
    "bool": lambda v: v.lower() in {"1", "true", "yes", "on"},
    # New support for integer types
    "integer": int,
}


def parse_typed_value(value: Any) -> Any:
    """Parse values of the form ``"<type>=<value>"``.

    Parameters
    ----------
    value:
        The value to parse. If it is not a string or does not follow the
        ``type=data`` pattern, it is returned unchanged.
    """
    if not isinstance(value, str) or "=" not in value:
        return value
    type_tag, _, data = value.partition("=")
    parser = _TYPE_PARSERS.get(type_tag)
    if parser is None:
        return value
    return parser(data)
