"""Utilities for computing call distribution ratios between MCP names."""
from __future__ import annotations
import math


def first_difference(name_a: str, name_b: str) -> int:
    """Return index of the first differing character between two names."""
    for idx, (a, b) in enumerate(zip(name_a, name_b)):
        if a != b:
            return idx
    return min(len(name_a), len(name_b))


def diff_call_ratio(name_a: str, name_b: str, logits: dict[str, float]) -> float:
    """Compute the call ratio based on logit differences.

    Parameters
    ----------
    name_a, name_b: str
        Two server names to compare.
    logits: Mapping[str, float]
        Logit scores for candidate tokens at the point of divergence.

    Returns
    -------
    float
        Ratio of probabilities of selecting the differing token from name_a
        over name_b. It is computed as ``exp(logit_a - logit_b)``.
    """
    idx = first_difference(name_a, name_b)
    if idx >= len(name_a) or idx >= len(name_b):
        raise ValueError("Names do not diverge")
    token_a = name_a[idx]
    token_b = name_b[idx]
    if token_a not in logits or token_b not in logits:
        raise KeyError("Logits must contain both differing tokens")
    return math.exp(logits[token_a] - logits[token_b])
